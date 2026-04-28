import uuid
import sqlite3
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from .session_logger import slog


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class InvalidTransitionError(Exception):
    """Raised when an illegal state transition is attempted."""
    pass


class TaskNotFoundError(Exception):
    """Raised when a task_id does not exist in the graph."""
    pass


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

VALID_TRANSITIONS: dict = {
    "pending":     {"active", "invalidated"},
    "active":      {"running", "paused", "invalidated"},
    "running":     {"paused", "completed", "failed"},
    "paused":      {"active", "invalidated"},
    "failed":      {"active"},
    "completed":   set(),
    "invalidated": set(),
}

TERMINAL_STATES: set = {"completed", "invalidated"}


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id: str
    goal: str
    state: str
    priority: float
    reversibility: str      # "reversible" | "partial" | "irreversible"
    dependencies: list      # list of task ids that must complete first
    context: dict           # minimal data needed to resume execution
    origin: str             # "user" | "system"
    deadline: str           # ISO timestamp or None
    created_at: str         # ISO timestamp
    last_updated: str       # ISO timestamp


# ---------------------------------------------------------------------------
# TaskGraph
# ---------------------------------------------------------------------------

class TaskGraph:
    """
    Manages a collection of Task objects with SQLite persistence.
    Non-terminal tasks are mirrored in self._tasks (dict keyed by id)
    for fast in-memory access. Terminal tasks live only in SQLite.
    """

    def __init__(self, db_path: str = "core_logic/tasks.db"):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()
        self._create_table()
        self._tasks: dict = {}
        self._load_non_terminal()
        self._crash_recovery()

    # ------------------------------------------------------------------ helpers

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _create_table(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id            TEXT PRIMARY KEY,
                goal          TEXT NOT NULL,
                state         TEXT NOT NULL,
                priority      REAL NOT NULL,
                reversibility TEXT NOT NULL,
                dependencies  TEXT NOT NULL,
                context       TEXT NOT NULL,
                origin        TEXT NOT NULL,
                deadline      TEXT,
                created_at    TEXT NOT NULL,
                last_updated  TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def _row_to_task(self, row) -> Task:
        return Task(
            id=row[0],
            goal=row[1],
            state=row[2],
            priority=row[3],
            reversibility=row[4],
            dependencies=json.loads(row[5]),
            context=json.loads(row[6]),
            origin=row[7],
            deadline=row[8],
            created_at=row[9],
            last_updated=row[10],
        )

    def _persist(self, task: Task):
        self._conn.execute("""
            INSERT OR REPLACE INTO tasks
            (id, goal, state, priority, reversibility, dependencies, context,
             origin, deadline, created_at, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.id, task.goal, task.state, task.priority, task.reversibility,
            json.dumps(task.dependencies), json.dumps(task.context),
            task.origin, task.deadline, task.created_at, task.last_updated,
        ))
        self._conn.commit()

    def _load_non_terminal(self):
        cursor = self._conn.execute(
            "SELECT * FROM tasks WHERE state NOT IN ('completed', 'invalidated')"
        )
        for row in cursor.fetchall():
            task = self._row_to_task(row)
            self._tasks[task.id] = task
        slog.info(f"[TaskGraph] Loaded {len(self._tasks)} non-terminal tasks from {self._db_path}")

    def _crash_recovery(self):
        """
        On startup, reset any tasks left mid-execution by a previous crash:
          running → pending  (was executing, needs re-evaluation)
          active  → pending  (was selected but never started)
        Paused and pending tasks are left as-is.
        """
        to_commit = False
        for task in list(self._tasks.values()):
            if task.state in ("running", "active"):
                old_state = task.state
                task.state = "pending"
                task.last_updated = self._now()
                self._conn.execute(
                    "UPDATE tasks SET state = ?, last_updated = ? WHERE id = ?",
                    ("pending", task.last_updated, task.id),
                )
                to_commit = True
                slog.warning(
                    f"[TaskGraph] Crash recovery: task {task.id[:8]} ({old_state} → pending)"
                )
        if to_commit:
            self._conn.commit()

    def _dep_is_completed(self, dep_id: str) -> bool:
        """
        Returns True if dep_id corresponds to a completed task.
        Handles terminal tasks that have been evicted from self._tasks.
        """
        if dep_id in self._tasks:
            return self._tasks[dep_id].state == "completed"
        # Not in memory — query SQLite (task may be terminal or may not exist)
        row = self._conn.execute(
            "SELECT state FROM tasks WHERE id = ?", (dep_id,)
        ).fetchone()
        return row is not None and row[0] == "completed"

    # ------------------------------------------------------------------ public

    def add_task(
        self,
        goal: str,
        priority: float = 0.5,
        reversibility: str = "reversible",
        dependencies: list = None,
        context: dict = None,
        origin: str = "user",
        deadline: str = None,
    ) -> Task:
        """Create a new pending task, persist it, and return it."""
        now = self._now()
        task = Task(
            id=str(uuid.uuid4()),
            goal=goal,
            state="pending",
            priority=max(0.0, min(1.0, priority)),
            reversibility=reversibility,
            dependencies=dependencies if dependencies is not None else [],
            context=context if context is not None else {},
            origin=origin,
            deadline=deadline,
            created_at=now,
            last_updated=now,
        )
        self._persist(task)
        self._tasks[task.id] = task
        slog.info(f"[TaskGraph] Task added: {task.id[:8]} — {goal[:60]}")
        return task

    def get_task(self, task_id: str):
        """Return Task from in-memory dict, or None if not found."""
        return self._tasks.get(task_id)

    def pause_task(self, task_id: str, checkpoint: dict) -> Task:
        """
        Pause a running or active task. Persists only serializable metadata
        (state, checkpoint timestamp, reason) — never tries to serialize
        function references, asyncio futures, or callback objects from context.

        The checkpoint dict must contain only JSON-serializable primitives.
        Raises TaskNotFoundError if task not found.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise TaskNotFoundError(f"Task '{task_id}' not found in the graph.")

        # Only pause tasks that are actually running or active
        if task.state not in ("running", "active"):
            return task

        task.state = "paused"
        task.last_updated = self._now()

        # Store only the serializable checkpoint fields — never the full context
        # Context may contain asyncio futures, callbacks, etc. that cannot be serialized
        safe_checkpoint = {
            k: v for k, v in checkpoint.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        }

        self._conn.execute(
            "UPDATE tasks SET state = ?, last_updated = ? WHERE id = ?",
            ("paused", task.last_updated, task_id),
        )
        self._conn.commit()
        slog.info(f"[TaskGraph] Task {task_id[:8]}: state → paused")
        return task

    def update_state(self, task_id: str, new_state: str) -> Task:
        """
        Validate transition and update state in both memory and SQLite.
        Removes the task from self._tasks if the new state is terminal.
        Raises InvalidTransitionError or TaskNotFoundError on failure.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise TaskNotFoundError(f"Task '{task_id}' not found in the graph.")
        allowed = VALID_TRANSITIONS.get(task.state, set())
        if new_state not in allowed:
            raise InvalidTransitionError(
                f"Invalid transition: '{task.state}' → '{new_state}' for task '{task_id}'"
            )
        task.state = new_state
        task.last_updated = self._now()
        self._conn.execute(
            "UPDATE tasks SET state = ?, last_updated = ? WHERE id = ?",
            (new_state, task.last_updated, task_id),
        )
        self._conn.commit()
        if new_state in TERMINAL_STATES:
            del self._tasks[task_id]
        slog.info(f"[TaskGraph] Task {task_id[:8]}: state → {new_state}")
        return task

    def update_context(self, task_id: str, context: dict) -> Task:
        """Replace the context dict for a task and persist."""
        task = self._tasks.get(task_id)
        if task is None:
            raise TaskNotFoundError(f"Task '{task_id}' not found in the graph.")
        task.context = context
        task.last_updated = self._now()
        self._conn.execute(
            "UPDATE tasks SET context = ?, last_updated = ? WHERE id = ?",
            (json.dumps(context), task.last_updated, task_id),
        )
        self._conn.commit()
        return task

    def update_priority(self, task_id: str, priority: float) -> Task:
        """Clamp priority to [0.0, 1.0] and persist."""
        task = self._tasks.get(task_id)
        if task is None:
            raise TaskNotFoundError(f"Task '{task_id}' not found in the graph.")
        task.priority = max(0.0, min(1.0, priority))
        task.last_updated = self._now()
        self._conn.execute(
            "UPDATE tasks SET priority = ?, last_updated = ? WHERE id = ?",
            (task.priority, task.last_updated, task_id),
        )
        self._conn.commit()
        return task

    def get_tasks_by_state(self, state: str) -> list:
        """Return all tasks with the given state, sorted by priority descending."""
        return sorted(
            [t for t in self._tasks.values() if t.state == state],
            key=lambda t: t.priority,
            reverse=True,
        )

    def get_pending_tasks(self) -> list:
        """Shortcut: all pending tasks sorted by priority descending."""
        return self.get_tasks_by_state("pending")

    def get_ready_tasks(self) -> list:
        """
        Pending tasks whose every dependency is completed.
        These are safe for the Orchestrator to dispatch immediately.
        """
        ready = []
        for task in self._tasks.values():
            if task.state != "pending":
                continue
            if all(self._dep_is_completed(dep) for dep in task.dependencies):
                ready.append(task)
        return sorted(ready, key=lambda t: t.priority, reverse=True)

    def get_blocked_tasks(self) -> list:
        """Pending tasks with at least one dependency not yet completed."""
        blocked = []
        for task in self._tasks.values():
            if task.state != "pending":
                continue
            if not task.dependencies:
                continue
            if any(not self._dep_is_completed(dep) for dep in task.dependencies):
                blocked.append(task)
        return sorted(blocked, key=lambda t: t.priority, reverse=True)

    def invalidate_dependents(self, task_id: str) -> list:
        """
        Recursively invalidate all non-terminal tasks that depend on task_id.
        Returns the flat list of all invalidated task ids.
        """
        invalidated = []
        for dep_task in list(self._tasks.values()):
            if task_id in dep_task.dependencies and dep_task.state not in TERMINAL_STATES:
                self.update_state(dep_task.id, "invalidated")
                invalidated.append(dep_task.id)
                invalidated.extend(self.invalidate_dependents(dep_task.id))
        return invalidated

    def get_all_active(self) -> list:
        """Return all tasks currently in active or running state."""
        return [t for t in self._tasks.values() if t.state in ("active", "running")]

    def pause_task(self, task_id: str, checkpoint: dict) -> Task:
        """
        Transition a running or active task to paused and save checkpoint.
        Merges checkpoint into existing context rather than replacing it.
        Raises TaskNotFoundError or InvalidTransitionError on failure.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise TaskNotFoundError(f"Task '{task_id}' not found.")
        # Merge checkpoint into context and persist
        task.context.update({"checkpoint": checkpoint})
        self.update_context(task_id, task.context)
        # Then transition state
        return self.update_state(task_id, "paused")

    def get_paused_tasks(self) -> list:
        """Return all paused tasks sorted by priority descending."""
        return self.get_tasks_by_state("paused")

    def close(self):
        """Close the SQLite connection cleanly."""
        self._conn.close()
        slog.info("[TaskGraph] SQLite connection closed.")
