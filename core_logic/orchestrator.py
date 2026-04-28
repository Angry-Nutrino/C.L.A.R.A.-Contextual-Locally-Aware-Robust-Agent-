import asyncio
from datetime import datetime, timezone
from .event_queue import EventQueue, make_event, Event
from .task_graph import TaskGraph, Task
from .session_logger import slog
from .conflict import ConflictDetector, ArbitrationEngine


class Orchestrator:
    """
    Continuous decision engine. Runs as a persistent asyncio background task.
    Consumes events from EventQueue, manages TaskGraph state, and dispatches
    ReAct workers. Never blocks. Never executes tool logic directly.
    """

    def __init__(
        self,
        agent,
        event_queue: EventQueue,
        task_graph: TaskGraph,
        tracer: "Tracer" = None,
    ):
        self._agent = agent
        self._event_queue = event_queue
        self._task_graph = task_graph
        self._running: bool = False
        self._loop_task: asyncio.Task | None = None
        self._active_workers: dict = {}  # task_id → asyncio.Task
        self._interrupt: bool = False
        self._conflict_detector  = ConflictDetector()
        self._arbitration_engine = ArbitrationEngine()
        self._tracer = tracer
        self._broadcast_fn = None  # injected by api.py after startup — avoids circular import

    def _trace(self, event: str, **fields) -> None:
        if self._tracer:
            self._tracer.emit(event, **fields)

    async def _broadcast_task(self, state: str, task) -> None:
        """Broadcast task state change to all connected WebSocket clients.
        _broadcast_fn is injected by api.py at startup to avoid circular imports."""
        try:
            if self._broadcast_fn:
                await self._broadcast_fn(
                    task_id=task.id,
                    goal=task.goal,
                    state=state,
                    priority=task.priority,
                    source=task.origin,
                )
        except Exception as e:
            slog.warning(f"   [Broadcast] task_event failed: {e}")

    # ------------------------------------------------------------------ public

    async def start(self) -> None:
        """Launch the orchestrator loop as a background asyncio task."""
        if self._running:
            slog.warning("[Orchestrator] Already running — start() ignored.")
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._loop())
        slog.info("[Orchestrator] Started.")

    async def stop(self) -> None:
        """Cancel the loop and all active workers, then close the TaskGraph."""
        self._running = False

        # Cancel all active workers
        if self._active_workers:
            slog.info(f"[Orchestrator] Cancelling {len(self._active_workers)} active worker(s)...")
            for worker_task in self._active_workers.values():
                worker_task.cancel()
            await asyncio.gather(*self._active_workers.values(), return_exceptions=True)
            self._active_workers.clear()

        # Cancel the loop task
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

        self._task_graph.close()
        if self._tracer:
            self._tracer.close()
        slog.info("[Orchestrator] Stopped. TaskGraph closed.")

    async def submit_user_event(self, text: str, image_data=None, on_step_update=None) -> str:
        """
        Entry point for the WebSocket handler. Creates a response future,
        emits a user_input event, and awaits the future resolved by the worker.
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        event = make_event(
            type="user_input",
            payload={
                "text": text,
                "image_data": image_data,
                "on_step_update": on_step_update,
                "response_future": future,
            },
            priority=1.0,
            source="user",
        )
        await self._event_queue.emit(event)
        return await future  # waits for _run_worker to resolve it

    # ------------------------------------------------------------------ private

    async def _loop(self) -> None:
        """Main tick. Runs until self._running is False."""
        slog.info("[Orchestrator] Loop started.")
        # Dispatch any tasks that were pending before the loop started
        # (crash recovery tasks, or tasks added before start() was called).
        await self._dispatch_ready_tasks()
        while self._running:
            try:
                events = await self._event_queue.drain_blocking(timeout=0.1)
                self._trace(
                    "orchestrator_tick",
                    events_drained=len(events),
                    ready_tasks=len(self._task_graph.get_ready_tasks()),
                    active_workers=len(self._active_workers),
                    pending_tasks=len(self._task_graph.get_tasks_by_state("pending")),
                    paused_tasks=len(self._task_graph.get_tasks_by_state("paused")),
                )
                await self._ingest_events(events)
                await self._dispatch_ready_tasks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                slog.error(f"[Orchestrator] Loop error: {e}")
                # Never let a loop error kill the orchestrator
                await asyncio.sleep(0.1)
        slog.info("[Orchestrator] Loop exited.")

    async def _ingest_events(self, events: list) -> None:
        """Process each event and update the TaskGraph accordingly."""
        for event in events:
            try:
                if event.type == "user_input":
                    await self._handle_user_input(event)

                elif event.type == "task_completed":
                    task_id = event.payload["task_id"]
                    result = event.payload["result"]
                    # Grab future BEFORE update_state evicts the task from memory
                    task = self._task_graph.get_task(task_id)
                    future = task.context.get("response_future") if task else None
                    self._task_graph.update_state(task_id, "completed")
                    if future and not future.done():
                        future.set_result(result)
                    # Resume any paused background tasks now that the foreground is clear
                    await self._resume_paused_tasks()

                elif event.type == "task_failed":
                    task_id = event.payload["task_id"]
                    error   = event.payload["error"]
                    await self._handle_task_failure(task_id, error)

                elif event.type == "system_trigger":
                    task_id = event.payload.get("task_id")
                    trigger = event.payload.get("trigger", "unknown")
                    if task_id:
                        tg_task = self._task_graph.get_task(task_id)
                        if tg_task and tg_task.state == "pending":
                            slog.info(
                                f"[Orchestrator] system_trigger '{trigger}' → "
                                f"activating Task {task_id[:8]}"
                            )
                            # Task already exists in SQLite (write-ahead).
                            # _dispatch_ready_tasks will pick it up on the next tick.
                        elif tg_task and tg_task.state in ("completed", "failed", "invalidated"):
                            # Normal for background tasks that completed and re-fired —
                            # the scheduler creates a new task on the next cycle.
                            pass
                        else:
                            slog.debug(
                                f"[Orchestrator] system_trigger '{trigger}' "
                                f"— task_id '{task_id}' already completed (normal for background tasks)."
                            )
                    else:
                        slog.warning(
                            f"[Orchestrator] system_trigger received with no task_id: "
                            f"{event.payload}"
                        )

                elif event.type == "error":
                    slog.error(
                        f"[Orchestrator] error event from {event.payload.get('source', '?')}: "
                        f"{event.payload.get('message', '')}"
                    )

                self._trace(
                    "event_ingested",
                    event_type=event.type,
                    event_id=event.id,
                    priority=event.priority,
                    source=event.source,
                )
            except Exception as e:
                slog.error(f"[Orchestrator] Failed to ingest event {event.type}: {e}")

    async def _handle_user_input(self, event: Event) -> None:
        """Create a Task from a user_input event and inject the non-serializable runtime objects."""
        payload = event.payload
        text = payload.get("text", "")
        image_data = payload.get("image_data")
        on_step_update = payload.get("on_step_update")
        future = payload.get("response_future")

        # Add task with only serializable context (image_data is base64 str — safe)
        task = self._task_graph.add_task(
            goal=text,
            priority=1.0,
            reversibility="reversible",
            dependencies=[],
            context={"text": text, "image_data": image_data},
            origin="user",
        )

        # Inject non-serializable runtime objects directly into in-memory context.
        # These are never persisted to SQLite — futures and callbacks are transient.
        task.context["on_step_update"] = on_step_update
        task.context["response_future"] = future
        await self._broadcast_task("pending", task)

        self._trace(
            "task_created",
            task_id=task.id,
            goal=task.goal[:80],
            origin=task.origin,
            priority=task.priority,
            reversibility=task.reversibility,
        )

        # If another user task is already running, queue this one behind it.
        # Background tasks (origin=system) run in parallel and are unaffected.
        running_user_tasks = [
            t for t in self._task_graph.get_tasks_by_state("running")
            if t.origin == "user"
        ]
        if running_user_tasks:
            task.priority = 0.95
            slog.info(
                f"[Orchestrator] User task queued behind running task "
                f"{running_user_tasks[0].id[:8]} — priority set to 0.95"
            )

        # Signal interrupt — pause lower-priority running work before dispatching
        self._interrupt = True
        await self._check_and_pause_lower_priority(new_task_priority=1.0)
        self._interrupt = False

        slog.info(f"[Orchestrator] user_input → Task {task.id[:8]} created: {text[:60]}")

    async def _dispatch_ready_tasks(self) -> None:
        """Select pending tasks whose dependencies are met and launch workers."""
        if self._interrupt:
            return  # Do not dispatch during active interrupt handling
        ready   = self._task_graph.get_ready_tasks()
        running = self._task_graph.get_all_active()

        for task in ready:
            if task.id in self._active_workers:
                continue  # already running

            # Conflict check before dispatch
            conflicts = self._conflict_detector.check(task, running)
            result    = self._arbitration_engine.arbitrate(task, conflicts, running)

            self._trace(
                "dispatch_decision",
                task_id=task.id,
                goal=task.goal[:80],
                decision=result.decision,
                conflicts=[
                    {
                        "type": c.type,
                        "task_b": c.task_b[:8],
                        "reason": c.reason,
                        "severity": c.severity,
                    }
                    for c in conflicts
                ],
                reason=result.reason,
            )

            if result.decision == "dispatch":
                slog.info(
                    f"[Orchestrator] Dispatching task {task.id[:8]} | "
                    f"Arbitration: {result.reason}"
                )
                self._task_graph.update_state(task.id, "active")
                self._trace(
                    "task_state_change",
                    task_id=task.id,
                    goal=task.goal[:80],
                    origin=task.origin,
                    priority=task.priority,
                    from_state="pending",
                    to_state="active",
                )
                worker = asyncio.create_task(self._run_worker(task))
                self._active_workers[task.id] = worker
                running.append(task)  # treat as running for subsequent checks in same tick

            elif result.decision == "defer":
                slog.info(
                    f"[Orchestrator] Deferring task {task.id[:8]} | "
                    f"Reason: {result.reason}"
                )
                # Log deferral to episodic memory
                try:
                    self._agent.log_system_episode(
                        f"[TASK DEFERRED] '{task.goal[:60]}' deferred — "
                        f"{result.reason[:100]}"
                    )
                except Exception:
                    pass
                # Task stays pending — re-evaluated next tick

            elif result.decision == "reorder":
                # Reserved for Phase 7 — treat as defer for now
                slog.info(
                    f"[Orchestrator] Reorder (deferred) task {task.id[:8]} | "
                    f"Reason: {result.reason}"
                )

            elif result.decision == "notify_user":
                slog.warning(
                    f"[Orchestrator] Conflict — notifying user for task "
                    f"{task.id[:8]}: {result.reason}"
                )
                await self._notify_user_conflict(task, result.reason)

    async def _notify_user_conflict(self, task: Task, reason: str) -> None:
        """
        Resolves the task's response_future with a conflict explanation
        so the user receives a response rather than silence.
        Transitions the task to invalidated — it will not be retried.
        """
        future = task.context.get("response_future")
        if future and not future.done():
            future.set_result(
                f"I wasn't able to start that right now — another task is "
                f"currently using a conflicting resource. {reason} "
                f"Please try again in a moment."
            )

        # Log conflict to episodic memory
        try:
            self._agent.log_system_episode(
                f"[TASK CONFLICT] '{task.goal[:60]}' could not start — "
                f"{reason[:100]}"
            )
        except Exception:
            pass

        try:
            self._task_graph.update_state(task.id, "invalidated")
        except Exception as e:
            slog.error(
                f"[Orchestrator] Failed to invalidate conflicted task "
                f"{task.id[:8]}: {e}"
            )

    async def _handle_task_failure(self, task_id: str, error: str) -> None:
        """
        On task failure: check retry count. If under limit, create a new
        task with failure summary attached. If limit reached, notify user.
        """
        MAX_ATTEMPTS = 3

        task = self._task_graph.get_task(task_id)
        if task is None:
            return

        attempt = task.context.get("attempt", 1)
        future  = task.context.get("response_future")

        self._task_graph.update_state(task_id, "failed")

        if attempt >= MAX_ATTEMPTS:
            slog.warning(
                f"[Orchestrator] Task {task_id[:8]} failed after "
                f"{attempt} attempts. Notifying user."
            )
            if future and not future.done():
                future.set_result(
                    f"I was unable to complete this after {attempt} attempts. "
                    f"Last error: {error}"
                )
            try:
                self._agent.log_system_episode(
                    f"[TASK FAILED] '{task.goal[:60]}' failed after "
                    f"{attempt} attempts: {error[:100]}"
                )
            except Exception:
                pass
            return

        # Build failure summary for retry context
        failure_summary = {
            "reason": error,
            "attempt": attempt,
            "original_goal": task.goal,
            "suggested_adjustment": (
                "Try a different approach or break into smaller steps."
            ),
        }

        slog.info(
            f"[Orchestrator] Task {task_id[:8]} failed (attempt {attempt}). "
            f"Retrying with failure context..."
        )

        retry_context = {**task.context, "failure_summary": failure_summary,
                         "attempt": attempt + 1}

        self._task_graph.add_task(
            goal=task.goal,
            priority=task.priority,
            reversibility=task.reversibility,
            dependencies=[],
            context=retry_context,
            origin=task.origin,
        )

        try:
            self._agent.log_system_episode(
                f"[TASK RETRY] '{task.goal[:60]}' retrying "
                f"(attempt {attempt + 1}/{MAX_ATTEMPTS})"
            )
        except Exception:
            pass

    async def _check_and_pause_lower_priority(self, new_task_priority: float) -> None:
        """
        Pause all currently running/active tasks with priority lower than
        new_task_priority. Saves a checkpoint into their context before pausing.
        Cancels their worker asyncio.Tasks.
        """
        for task_id, worker_task in list(self._active_workers.items()):
            tg_task = self._task_graph.get_task(task_id)
            if tg_task is None:
                continue
            if tg_task.state not in ("running", "active"):
                continue
            if tg_task.priority >= new_task_priority:
                continue  # same or higher priority — do not pause

            slog.info(
                f"[Orchestrator] Pausing task {task_id[:8]} "
                f"(priority {tg_task.priority:.2f}) for higher-priority interrupt."
            )

            checkpoint = {
                "interrupted_at": datetime.now(timezone.utc).isoformat(),
                "reason": "higher_priority_interrupt",
            }
            try:
                self._task_graph.pause_task(task_id, checkpoint)
            except Exception as e:
                slog.error(f"[Orchestrator] Failed to pause task {task_id[:8]}: {e}")
                continue

            # Cancel the asyncio worker so it actually stops executing
            worker_task.cancel()
            self._active_workers.pop(task_id, None)

            # Log pause to episodic memory for CLARA's passive awareness
            try:
                self._agent.log_system_episode(
                    f"[TASK PAUSED] '{tg_task.goal[:60]}' paused — "
                    f"higher-priority user request took over."
                )
            except Exception:
                pass

            self._trace(
                "interrupt",
                paused_task_id=task_id[:8],
                paused_task_goal=tg_task.goal[:80],
                triggered_by="higher_priority_user_input",
            )
            worker_task.cancel()
            try:
                await worker_task
            except (asyncio.CancelledError, Exception):
                pass
            self._active_workers.pop(task_id, None)

    async def _resume_paused_tasks(self) -> None:
        """
        Called after a user task completes. Re-evaluates all paused tasks:
        - Tasks without a response_future → transition back to pending for re-dispatch.
        - Tasks with a response_future (user tasks, edge case) → skip.

        Phase 4 relevance check is simple: all background paused tasks are resumed.
        Phase 6 (Arbitration) will add smarter relevance evaluation.
        """
        paused = self._task_graph.get_paused_tasks()
        for task in paused:
            # Skip tasks that carry a response_future — these are user tasks
            if task.context.get("response_future") is not None:
                continue

            slog.info(
                f"[Orchestrator] Resuming paused task {task.id[:8]}: {task.goal[:50]}"
            )
            try:
                self._task_graph.update_state(task.id, "active")
            except Exception as e:
                slog.error(f"[Orchestrator] Failed to resume task {task.id[:8]}: {e}")

    async def _run_worker(self, task: Task) -> None:
        """Execute a task via the ReAct loop (user) or background worker (system)."""
        try:
            self._task_graph.update_state(task.id, "running")
            await self._broadcast_task("running", task)
            self._trace(
                "worker_start",
                task_id=task.id[:8],
                goal=task.goal[:80],
                origin=task.origin,
            )
            ctx = task.context

            if task.origin == "system":
                trigger = ctx.get("trigger", "unknown")

                # Known lightweight triggers use the fast background dispatch path.
                # Unknown or complex triggers go through the full intelligence pipeline.
                SIMPLE_TRIGGERS = {
                    "memory_maintenance", "context_warmup", "health_check",
                    "file_change", "memory_growth", "interaction_density",
                    "rag_rebuild",
                }

                if trigger in SIMPLE_TRIGGERS:
                    from .background_tasks import run_background_task
                    result = await run_background_task(
                        trigger_name=trigger,
                        agent=self._agent,
                        task_graph=self._task_graph,
                        ctx=ctx,
                    )
                else:
                    # Complex system task — full intelligence pipeline
                    goal = task.goal.replace(
                        "[BACKGROUND] ", ""
                    ).replace("[ENVIRONMENT] ", "")
                    result = await self._agent.process_request(
                        query=goal,
                        source="system",
                        task_context=ctx,
                    )

                # Log autonomous action to episodic memory
                if result:
                    try:
                        summary = (
                            f"[AUTONOMOUS] "
                            f"{task.goal.replace('[BACKGROUND] ', '').replace('[ENVIRONMENT] ', '')}"
                            f": {result[:200]}"
                        )
                        self._agent.log_system_episode(summary)
                        slog.info("[Orchestrator] Autonomous action logged.")
                    except Exception as e:
                        slog.error(f"[Orchestrator] Episodic log failed: {e}")
            else:
                # User task — full pipeline via process_request
                result = await self._agent.process_request(
                    query=ctx["text"],
                    image_data=ctx.get("image_data"),
                    on_step_update=ctx.get("on_step_update"),
                    source="user",
                    task_context=ctx,
                )

            self._trace(
                "worker_complete",
                task_id=task.id[:8],
                goal=task.goal[:80],
                origin=task.origin,
                result_preview=str(result)[:100] if result else "",
            )
            await self._broadcast_task("completed", task)
            await self._event_queue.emit(make_event(
                type="task_completed",
                payload={"task_id": task.id, "result": result},
                priority=0.6,
                source="worker",
            ))
        except Exception as e:
            slog.error(f"[Orchestrator] Worker failed for task {task.id[:8]}: {e}")
            await self._broadcast_task("failed", task)
            self._trace(
                "worker_failed",
                task_id=task.id[:8],
                goal=task.goal[:80],
                error=str(e)[:200],
            )
            await self._event_queue.emit(make_event(
                type="task_failed",
                payload={"task_id": task.id, "error": str(e)},
                priority=0.8,
                source="worker",
            ))
        finally:
            self._active_workers.pop(task.id, None)
