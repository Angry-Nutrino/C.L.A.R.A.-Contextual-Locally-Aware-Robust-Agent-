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
        self._active_workers: dict = {}   # task_id → asyncio.Task
        self._task_resources: dict = {}   # task_id → {"reads": set, "writes": set}
        self._conflict_detector  = ConflictDetector()
        self._arbitration_engine = ArbitrationEngine()
        self._tracer = tracer
        self._broadcast_fn = None  # injected by api.py after startup — avoids circular import
        self._send_message_fn = None  # injected by api.py — general WS message push (Brief 35 proactive delivery)

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
                    message_id=task.context.get("message_id", ""),
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

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running or pending user task by task_id.
        Returns True if cancelled, False if not found or already terminal.
        Resolves the response_future with a cancellation message so the
        WebSocket handler gets a response and the client isn't left hanging.
        """
        task = self._task_graph.get_task(task_id)
        if task is None:
            slog.warning(f"[Orchestrator] cancel_task: task {task_id[:8]} not found.")
            return False

        if task.state in ("completed", "failed", "invalidated"):
            slog.debug(f"[Orchestrator] cancel_task: task {task_id[:8]} already terminal ({task.state}).")
            return False

        slog.info(f"[Orchestrator] Cancelling task {task_id[:8]}: {task.goal[:60]}")

        # Cancel the asyncio worker if running
        worker = self._active_workers.pop(task_id, None)
        if worker and not worker.done():
            worker.cancel()
            try:
                await worker
            except (asyncio.CancelledError, Exception):
                pass

        # Resolve the future so the WS handler doesn't hang
        future = task.context.get("response_future")
        if future and not future.done():
            future.set_result("Cancelled.")

        # Transition to invalidated (reuses existing terminal state)
        try:
            self._task_graph.update_state(task_id, "invalidated")
        except Exception as e:
            slog.error(f"[Orchestrator] cancel_task state update failed: {e}")

        await self._broadcast_task("failed", task)  # 'failed' renders red in task board

        try:
            self._agent.log_system_episode(
                f"[TASK CANCELLED] '{task.goal[:60]}' cancelled by user."
            )
        except Exception:
            pass

        return True

    async def submit_user_event(
        self, text: str, image_data=None, file_data=None,
        on_step_update=None, on_interpreted=None,
        message_id: str = None, memory_mode: str = "full", return_trace: bool = False,
        channel: str = "interface",
    ) -> str:
        """
        Entry point for the WebSocket handler. Creates a response future,
        emits a user_input event, and awaits the future resolved by the worker.

        memory_mode (test harness only; real callers use the "full" default):
          "full"      — normal persistence.
          "ephemeral" — transient recent_exchanges only, NO permanent episodic/vault
                        (coherence drill — needs within-dialogue recall without polluting).
          "none"      — skip ALL memory writes (L1-L5 harness — full isolation).
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        event = make_event(
            type="user_input",
            payload={
                "text": text,
                "image_data": image_data,
                "file_data": file_data,
                "message_id": message_id,
                "on_step_update": on_step_update,
                "on_interpreted": on_interpreted,
                "response_future": future,
                "memory_mode": memory_mode,
                "return_trace": return_trace,
                "channel": channel,   # Brief 43.3 — source tag for the persistent console (interface/telegram/voice/harness)
            },
            priority=1.0,
            source="user",
        )
        await self._event_queue.emit(event)
        return await future  # waits for _run_worker to resolve it

    # ------------------------------------------------------------------ private

    async def _loop(self) -> None:
        """Main tick. Runs until self._running is False."""
        import time as _time
        slog.info("[Orchestrator] Loop started.")
        # Dispatch any tasks that were pending before the loop started
        # (crash recovery tasks, or tasks added before start() was called).
        await self._dispatch_ready_tasks()
        # Tick-trace gating (Brief 36 A-12): the loop idles at ~10 ticks/sec and an
        # unconditional trace per tick accumulated 653 MB of identical idle records.
        # Emit only when the tick DID something (events drained / counts changed),
        # plus a 60s heartbeat so liveness stays visible in the trace.
        _last_tick_state = None
        _last_tick_ts = 0.0
        while self._running:
            try:
                events = await self._event_queue.drain_blocking(timeout=0.1)
                _state = (
                    len(self._task_graph.get_ready_tasks()),
                    len(self._active_workers),
                    len(self._task_graph.get_tasks_by_state("pending")),
                    len(self._task_graph.get_tasks_by_state("paused")),
                )
                _now = _time.monotonic()
                if events or _state != _last_tick_state or _now - _last_tick_ts >= 60.0:
                    self._trace(
                        "orchestrator_tick",
                        events_drained=len(events),
                        ready_tasks=_state[0],
                        active_workers=_state[1],
                        pending_tasks=_state[2],
                        paused_tasks=_state[3],
                    )
                    _last_tick_state = _state
                    _last_tick_ts = _now
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
                        # Brief 32: if a ReAct trace was captured (return_trace), resolve with a
                        # {response, react_trace} dict so /query can hand it to Layer-2 diagnosis.
                        # Normal callers (WS, Telegram) never set return_trace → bare string, unchanged.
                        _trace = event.payload.get("react_trace")
                        if _trace is not None:
                            future.set_result({"response": result, "react_trace": _trace})
                        else:
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
        file_data = payload.get("file_data")
        message_id = payload.get("message_id")
        on_step_update = payload.get("on_step_update")
        on_interpreted = payload.get("on_interpreted")
        future = payload.get("response_future")
        memory_mode = payload.get("memory_mode", "full")
        return_trace = payload.get("return_trace", False)
        channel = payload.get("channel", "interface")

        # Add task with only serializable context (image_data/file_data are base64 str — safe)
        task = self._task_graph.add_task(
            goal=text,
            priority=1.0,
            reversibility="reversible",
            dependencies=[],
            context={"text": text, "image_data": image_data, "file_data": file_data,
                     "message_id": message_id, "memory_mode": memory_mode,
                     "return_trace": return_trace, "channel": channel},
            origin="user",
        )

        # Inject non-serializable runtime objects directly into in-memory context.
        # These are never persisted to SQLite — futures and callbacks are transient.
        task.context["task_id"] = task.id
        task.context["on_step_update"] = on_step_update
        task.context["on_interpreted"] = on_interpreted
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

        slog.info(f"[Orchestrator] user_input → Task {task.id[:8]} created: {text[:60]}")

    async def _dispatch_ready_tasks(self) -> None:
        """Select pending tasks whose dependencies are met and launch workers."""
        ready   = self._task_graph.get_ready_tasks()
        running = self._task_graph.get_all_active()

        for task in ready:
            if task.id in self._active_workers:
                continue  # already running

            # Conflict check before dispatch — pass live resource ledger so the
            # detector sees what running tasks are actually touching right now.
            conflicts = self._conflict_detector.check(task, running, live_resources=self._task_resources)
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

        # Runtime objects (response_future, callbacks) ride along IN-MEMORY so the
        # retry can still stream to and answer the waiting client; TaskGraph._persist
        # sanitizes them out of the SQLite write (Brief 36 A-7 — previously this
        # add_task crashed on json.dumps(future) and the user's future hung forever).
        retry_context = {**task.context, "failure_summary": failure_summary,
                         "attempt": attempt + 1}
        retry_context.pop("active_tasks_context", None)  # stale snapshot — _run_worker rebuilds it

        retry_task = self._task_graph.add_task(
            goal=task.goal,
            priority=task.priority,
            reversibility=task.reversibility,
            dependencies=[],
            context=retry_context,
            origin=task.origin,
        )
        # The inherited task_id belongs to the FAILED task — refresh so the resource
        # ledger and trace events key on the retry's own id.
        retry_task.context["task_id"] = retry_task.id

        try:
            self._agent.log_system_episode(
                f"[TASK RETRY] '{task.goal[:60]}' retrying "
                f"(attempt {attempt + 1}/{MAX_ATTEMPTS})"
            )
        except Exception:
            pass

    # NOTE (Brief 36 A-14): the Phase-4 interrupt pauser (_check_and_pause_lower_priority)
    # was removed 2026-06-10 — it was never called from anywhere (zero pauses in the entire
    # log history), and the runtime model that actually shipped is plain concurrency: user
    # tasks and sub-second background observers run as parallel asyncio workers. The pause
    # machinery will be REBUILT properly (cooperative cancellation at ReAct-turn boundaries
    # with a real checkpoint) when Ambient Awareness introduces long-running autonomous tasks
    # worth preempting. TaskGraph.pause_task + the paused state remain available for that.

    async def _resume_paused_tasks(self) -> None:
        """
        Called after a user task completes. Re-evaluates all paused tasks:
        - Tasks without a response_future → transition back to PENDING so
          _dispatch_ready_tasks launches a worker on the next tick. (Resuming to
          "active" left the task workerless forever — Brief 36 A-14.)
        - Tasks with a response_future (user tasks, edge case) → skip.
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
                self._task_graph.update_state(task.id, "pending")
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

                # Log autonomous action to episodic memory — but NOT routine heartbeats.
                # health_check / memory_maintenance / context_warmup fire every 2-10 min;
                # logging each result made 468 of 1028 episodes (45%) permanent noise that
                # retrieval filters out anyway (Brief 36 B-20). A routine trigger only earns
                # an episode when something actually HAPPENED (repair, prune, failure).
                ROUTINE_TRIGGERS = {"health_check", "memory_maintenance", "context_warmup"}
                _notable_markers = ("re-sync", "pruned", "deleted", "removed",
                                    "repaired", "failed", "error")
                _is_routine = trigger in ROUTINE_TRIGGERS
                _notable = (not _is_routine) or any(
                    k in str(result).lower() for k in _notable_markers
                )
                if result and _notable:
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
                # User task — inject awareness + resource tracking before calling process_request.

                # Layer 2: tell this task what else is currently running so Clara
                # can reason about potential overlap without needing to ask.
                other_running = [
                    t for tid, t in self._task_graph._tasks.items()
                    if tid != task.id and t.state in ("running", "active")
                ]
                if other_running:
                    lines = [f"  - \"{t.goal[:80]}\"" for t in other_running]
                    ctx["active_tasks_context"] = (
                        "[ACTIVE TASKS — other work currently in progress]\n"
                        + "\n".join(lines)
                        + "\nIf your task may touch the same files or resources, be aware of the overlap."
                    )
                    slog.info(
                        f"[Orchestrator] [ACTIVE TASKS] injected into task {task.id[:8]}: "
                        + ", ".join(f"{t.id[:8]}" for t in other_running)
                    )

                # Layer 3: resource callback — registers filesystem touches into the
                # live ledger so ConflictDetector has ground-truth data at dispatch time.
                _orch = self
                _task_id = task.id

                def _resource_callback(tool_name: str, path: str, mode: str) -> None:
                    if not path:
                        return
                    ledger = _orch._task_resources.setdefault(_task_id, {"reads": set(), "writes": set()})
                    if mode == "write":
                        ledger["writes"].add(path)
                    else:
                        ledger["reads"].add(path)

                ctx["resource_callback"] = _resource_callback

                result = await self._agent.process_request(
                    query=ctx["text"],
                    image_data=ctx.get("image_data"),
                    file_data=ctx.get("file_data"),
                    on_step_update=ctx.get("on_step_update"),
                    source="user",
                    task_context=ctx,
                )

                # Brief 35 — task-level persistence (user tasks only). A DELIBERATE task that
                # soft-failed (status INCOMPLETE) gets ONE detached background retry: the live
                # future resolves NOW with Clara's honest answer + a retry notice, and a fresh
                # task re-attempts and delivers proactively when done. A retry's OWN terminal
                # outcome (is_retry) is delivered proactively here — there is no live future.
                _status = ctx.get("completion_status", "COMPLETE")
                _is_retry = bool(ctx.get("is_retry"))
                # Only REAL user traffic gets the detached-retry + proactive-delivery treatment.
                # Test traffic (memory_mode ephemeral/none — coherence drill, L1-L5 harness) must
                # NOT spawn a retry: it would run in 'full' and pollute episodic with the test's
                # fake scenario AND push a follow-up to Alkama's Telegram about a prompt that never
                # really happened (2026-06-08 leak). For test traffic, INCOMPLETE just resolves
                # honestly with no retry machinery. (retry_ctx also carries memory_mode now.)
                _memory_mode = ctx.get("memory_mode", "full")
                if _status == "INCOMPLETE" and not _is_retry and _memory_mode == "full":
                    await self._spawn_detached_retry(task, ctx, ctx.get("incomplete_reason", ""), result)
                    result = result.rstrip() + (
                        "\n\n*(I couldn't finish this on the first pass — I'm taking another "
                        "run at it now and will follow up.)*"
                    )
                elif _is_retry:
                    await self._deliver_retry_result(task, ctx, result, _status)

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
                # react_trace is None unless return_trace was set (Brief 32) — process_request
                # stashed it on the context after capturing the post-routing ReAct turns.
                payload={"task_id": task.id, "result": result,
                         "react_trace": ctx.get("react_trace")},
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
            self._task_resources.pop(task.id, None)   # clean up Layer 3 ledger
            from .resource_ledger import resource_ledger as _rl
            _rl.release_task(task.id)                  # read hashes ONLY. Write locks are released by the
                                                       # caller's own try/finally, per acquire_write's
                                                       # docstring — release_task does not touch them.
                                                       # (comment corrected 2026-08-24: it claimed both)

    # ── Brief 35: task-level persistence (detached retry + proactive delivery) ──────────
    async def _spawn_detached_retry(self, orig_task, orig_ctx: dict, reason: str, partial: str) -> None:
        """Spawn a DETACHED (no-future) background retry of a soft-failed user task.

        It runs through the normal pipeline (is_retry=True → process_request injects the failure
        context so Clara continues from progress), and delivers its result proactively when done.
        Capped at ONE retry: the spawned task carries is_retry=True, so if IT soft-fails again the
        worker will NOT spawn another (no retry chains)."""
        retry_ctx = {
            "text": orig_ctx.get("text", orig_task.goal),
            "image_data": orig_ctx.get("image_data"),
            "file_data": orig_ctx.get("file_data"),
            "message_id": orig_ctx.get("message_id", ""),
            "memory_mode": orig_ctx.get("memory_mode", "full"),  # retry inherits isolation (no full-mode pollution)
            "is_retry": True,
            "retry_of": orig_task.id,
            "failure_reason": reason,
            "partial_answer": partial,
        }
        retry_task = self._task_graph.add_task(
            goal=orig_task.goal,
            priority=orig_task.priority,
            reversibility=orig_task.reversibility,
            dependencies=[],
            context=retry_ctx,
            origin="user",
        )
        retry_task.context["task_id"] = retry_task.id   # no response_future — detached
        await self._broadcast_task("pending", retry_task)
        slog.info(f"[Orchestrator] Brief 35: detached retry {retry_task.id[:8]} of "
                  f"{orig_task.id[:8]} spawned — reason: {reason[:60]}")
        try:
            self._agent.log_system_episode(
                f"[TASK SOFT-RETRY] '{orig_task.goal[:60]}' soft-failed ({reason[:50]}); "
                f"detached retry {retry_task.id[:8]} spawned."
            )
        except Exception:
            pass

    async def _deliver_retry_result(self, task, ctx: dict, result: str, status: str) -> None:
        """Proactively deliver a detached retry's terminal outcome — the live future is long gone.

        Never lost: process_request already consolidated this outcome into episodic memory (the
        retry is the TERMINAL attempt, so will_retry was False → it memorized normally). Here we
        PUSH it to Alkama's channels, re-anchored to the original request: a fresh WS message
        (best-effort) + Telegram (reliable, purpose-built for proactive outbound)."""
        import uuid as _uuid
        goal = ctx.get("text", task.goal)
        lead = ("I went back and finished it:" if status == "COMPLETE"
                else "I retried but still couldn't complete it:")
        msg = f"Following up on your earlier request — \"{goal[:120]}\". {lead}\n\n{result}"

        # WS push — a fresh assistant bubble (new message_id so it's distinct from the resolved one)
        try:
            if getattr(self, "_send_message_fn", None):
                await self._send_message_fn({
                    "type": "final_answer",
                    "content": msg,
                    "message_id": f"retry-{_uuid.uuid4().hex[:8]}",
                })
        except Exception as e:
            slog.warning(f"   [Brief35] WS proactive push failed: {e}")

        # Telegram — reliable fallback / primary if the web UI is closed; no-ops if unconfigured.
        try:
            from .telegram_bot import notifier
            await notifier.send(msg)
        except Exception as e:
            slog.warning(f"   [Brief35] Telegram proactive push failed: {e}")

        slog.info(f"[Orchestrator] Brief 35: delivered retry outcome for '{task.goal[:50]}' (status {status}).")
        try:
            self._agent.log_system_episode(
                f"[TASK SOFT-RETRY] delivered retry outcome for '{task.goal[:50]}' — status {status}."
            )
        except Exception:
            pass
