import uuid
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from .session_logger import slog


# ---------------------------------------------------------------------------
# Valid values
# ---------------------------------------------------------------------------

VALID_EVENT_TYPES: set = {
    "user_input",
    "task_completed",
    "task_failed",
    "system_trigger",
    "error",
}

VALID_SOURCES: set = {"user", "worker", "system"}


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------

@dataclass
class Event:
    id: str           # uuid4
    type: str         # one of VALID_EVENT_TYPES
    payload: dict     # contents depend on type
    priority: float   # 0.0–1.0, higher = processed first
    timestamp: str    # ISO timestamp of creation
    source: str       # "user" | "worker" | "system"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_event(
    type: str,
    payload: dict,
    priority: float = 0.5,
    source: str = "system",
) -> Event:
    """
    Create a validated Event with auto-generated id and timestamp.
    Raises ValueError if type or source is not a known valid value.
    Priority is silently clamped to [0.0, 1.0].
    """
    if type not in VALID_EVENT_TYPES:
        raise ValueError(
            f"Unknown event type '{type}'. Valid types: {sorted(VALID_EVENT_TYPES)}"
        )
    if source not in VALID_SOURCES:
        raise ValueError(
            f"Unknown source '{source}'. Valid sources: {sorted(VALID_SOURCES)}"
        )
    return Event(
        id=str(uuid.uuid4()),
        type=type,
        payload=payload,
        priority=max(0.0, min(1.0, priority)),
        timestamp=datetime.now(timezone.utc).isoformat(),
        source=source,
    )


# ---------------------------------------------------------------------------
# EventQueue
# ---------------------------------------------------------------------------

class EventQueue:
    """
    Async priority queue for all signals reaching the Orchestrator.

    Internally wraps asyncio.PriorityQueue. Events are stored as
    (1.0 - priority, counter, event) tuples so that higher-priority
    events sort first in the min-heap. The counter breaks ties between
    equal-priority events by insertion order.
    """

    def __init__(self):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._counter: int = 0  # tie-breaker for equal-priority events

    async def emit(self, event: Event) -> None:
        """Put an event into the priority queue."""
        heap_key = (1.0 - event.priority, self._counter, event)
        self._counter += 1
        await self._queue.put(heap_key)
        slog.info(
            f"[EventQueue] emit | type={event.type} source={event.source} "
            f"priority={event.priority:.2f} id={event.id[:8]}"
        )

    async def drain(self) -> list:
        """
        Non-blocking drain of all currently available events.
        Returns them in priority-descending order.
        Returns [] if the queue is empty.
        """
        events = []
        while True:
            try:
                _, _, event = self._queue.get_nowait()
                events.append(event)
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        return events

    async def drain_blocking(self, timeout: float = 1.0) -> list:
        """
        Wait up to `timeout` seconds for at least one event, then drain
        all remaining events non-blocking.
        Returns events in priority-descending order.
        Returns [] if nothing arrives before the timeout.
        """
        try:
            first = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return []

        _, _, event = first
        self._queue.task_done()
        events = [event]

        # Drain any remaining events that are already in the queue
        while True:
            try:
                _, _, ev = self._queue.get_nowait()
                events.append(ev)
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

        return events

    def size(self) -> int:
        """Return the current number of events in the queue."""
        return self._queue.qsize()

    def is_empty(self) -> bool:
        """Return True if the queue has no events."""
        return self._queue.empty()
