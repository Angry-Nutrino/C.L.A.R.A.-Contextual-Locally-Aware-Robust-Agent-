import asyncio
from datetime import datetime, timezone
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .event_queue import EventQueue, make_event
from .task_graph import TaskGraph
from .session_logger import slog


# ---------------------------------------------------------------------------
# Priority table
# ---------------------------------------------------------------------------

_TRIGGER_PRIORITIES: dict = {
    "file_change":         0.4,
    "memory_growth":       0.35,
    "interaction_density": 0.3,
}

# Paths / patterns that generate high-frequency noise and must be ignored
IGNORED_PATTERNS: list = [
    "__pycache__",
    ".pyc",
    "tasks.db",
    "tasks.db-wal",
    "tasks.db-shm",
    ".log",
    "session_",
]


# ---------------------------------------------------------------------------
# Watchdog file-system event handler
# ---------------------------------------------------------------------------

class _FileEventHandler(FileSystemEventHandler):
    """Bridges watchdog OS-thread callbacks into the asyncio event loop."""

    def __init__(self, watcher: "EnvironmentWatcher"):
        super().__init__()
        self._watcher = watcher

    def _should_ignore(self, path: str) -> bool:
        return any(pat in path for pat in IGNORED_PATTERNS)

    def on_modified(self, event):
        if event.is_directory:
            return
        path = event.src_path
        if self._should_ignore(path):
            return
        asyncio.run_coroutine_threadsafe(
            self._watcher._on_file_changed(path, "modified"),
            self._watcher._event_loop,
        )

    def on_created(self, event):
        if event.is_directory:
            return
        path = event.src_path
        if self._should_ignore(path):
            return
        asyncio.run_coroutine_threadsafe(
            self._watcher._on_file_changed(path, "created"),
            self._watcher._event_loop,
        )


# ---------------------------------------------------------------------------
# EnvironmentWatcher
# ---------------------------------------------------------------------------

class EnvironmentWatcher:
    """
    Observes the external environment and emits system_trigger events when
    meaningful state changes occur. Complements BackgroundScheduler (clock-driven)
    with event-driven signals.

    Three observation targets:
      1. File system watch — watchdog monitors specified directories
      2. Memory growth watch — fires when episodic log crosses a growth threshold
      3. Interaction density — fires after N user interactions in a session
    """

    def __init__(
        self,
        task_graph: TaskGraph,
        event_queue: EventQueue,
        agent,
        event_loop: asyncio.AbstractEventLoop,
        watch_paths: list = None,
        memory_growth_threshold: int = 5,
        interaction_density_threshold: int = 10,
    ):
        self._task_graph  = task_graph
        self._event_queue = event_queue
        self._agent       = agent
        self._event_loop  = event_loop
        self._watch_paths = watch_paths if watch_paths is not None else ["core_logic/"]
        self._memory_growth_threshold      = memory_growth_threshold
        self._interaction_density_threshold = interaction_density_threshold

        self._running: bool = False
        self._observer = None
        self._last_episode_count: int = 0
        self._interaction_count: int  = 0
        self._emit_lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------ public

    async def start(self) -> None:
        if self._running:
            slog.warning("[EnvWatcher] Already running — start() ignored.")
            return
        self._running = True

        # Snapshot current episode count as baseline
        self._last_episode_count = len(
            self._agent.db.memory.get("episodic_log", [])
        )

        # Start watchdog Observer
        handler = _FileEventHandler(self)
        self._observer = Observer()
        for path in self._watch_paths:
            self._observer.schedule(handler, path, recursive=True)
        self._observer.start()

        slog.info(
            f"[EnvWatcher] Started. Watching: {self._watch_paths} | "
            f"Memory threshold: {self._memory_growth_threshold} | "
            f"Interaction threshold: {self._interaction_density_threshold}"
        )

    async def stop(self) -> None:
        self._running = False
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        slog.info("[EnvWatcher] Stopped.")

    def notify_interaction(self) -> None:
        """
        Called after each user interaction. Increments counter and fires
        interaction_density trigger when threshold is reached.
        Safe to call from sync context (WebSocket handler).
        """
        self._interaction_count += 1
        if self._interaction_count >= self._interaction_density_threshold:
            self._interaction_count = 0
            asyncio.run_coroutine_threadsafe(
                self._emit_trigger("interaction_density"),
                self._event_loop,
            )

    async def check_memory_growth(self) -> None:
        """
        Compare current episodic log size against last snapshot.
        Emits memory_growth trigger when growth >= threshold.
        """
        current_count = len(self._agent.db.memory.get("episodic_log", []))
        if current_count - self._last_episode_count >= self._memory_growth_threshold:
            self._last_episode_count = current_count
            await self._emit_trigger("memory_growth")

    # ------------------------------------------------------------------ private

    async def _on_file_changed(self, path: str, change_type: str) -> None:
        if not self._running:
            return

        # Normalise separators so Windows paths work with pattern checks
        normalised = path.replace("\\", "/")

        # Filter noise — same patterns as _FileEventHandler._should_ignore
        if any(pat in normalised for pat in IGNORED_PATTERNS):
            return

        # memory.json change → check growth instead of generic file_change
        if normalised.endswith("memory.json"):
            await self.check_memory_growth()
            return

        await self._emit_trigger(
            trigger_name="file_change",
            extra_context={"path": path, "change_type": change_type},
        )

    async def _emit_trigger(
        self,
        trigger_name: str,
        extra_context: dict = None,
    ) -> None:
        """Write-ahead: Task persisted to SQLite before the wake event is emitted."""
        async with self._emit_lock:
            try:
                context = {
                    "trigger": trigger_name,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                }
                if extra_context:
                    context.update(extra_context)

                task = self._task_graph.add_task(
                    goal=f"[ENVIRONMENT] {trigger_name}",
                    priority=_TRIGGER_PRIORITIES.get(trigger_name, 0.3),
                    reversibility="reversible",
                    dependencies=[],
                    context=context,
                    origin="system",
                )

                await self._event_queue.emit(make_event(
                    type="system_trigger",
                    payload={"trigger": trigger_name, "task_id": task.id},
                    priority=_TRIGGER_PRIORITIES.get(trigger_name, 0.3),
                    source="system",
                ))
                slog.info(
                    f"[EnvWatcher] Trigger emitted: {trigger_name} "
                    f"-> Task {task.id[:8]}"
                )
            except Exception as e:
                slog.error(
                    f"[EnvWatcher] Failed to emit trigger '{trigger_name}': {e}"
                )
