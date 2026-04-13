import json
import os
from datetime import datetime, timezone


class Tracer:
    """
    Writes structured JSONL trace records to traces/<session_timestamp>.jsonl
    One record per line. Thread-safe for asyncio single-threaded use.
    No-op if disabled — all emit() calls are safe to call unconditionally.
    """

    def __init__(self, enabled: bool = True, traces_dir: str = "traces"):
        self._enabled = enabled
        self._file = None
        if not enabled:
            return
        os.makedirs(traces_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(traces_dir, f"trace_{ts}.jsonl")
        self._file = open(path, "a", encoding="utf-8", buffering=1)
        # buffering=1 = line-buffered: each record flushed immediately

    def emit(self, event: str, **fields) -> None:
        """Write one trace record. No-op if disabled or file not open."""
        if not self._enabled or self._file is None:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        try:
            self._file.write(json.dumps(record) + "\n")
        except Exception:
            pass  # never let tracer errors affect system behavior

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
