import asyncio
import hashlib
from .session_logger import slog


class ResourceLedger:
    """
    Per-task read-hash tracking and per-path write locks for concurrent task safety.

    Two mechanisms work together:
    1. Read-modify-write protection: records a hash of file content at read time,
       then validates it at write time. If another task wrote the file in between,
       returns a conflict error so Clara re-reads before overwriting.
    2. Pure-write exclusivity: asyncio.Lock per path, held only for the duration
       of the write call, so two pure writes to the same file cannot interleave.

    Both mechanisms are opt-in via task_id — background/system tasks pass no
    task_id and bypass all checks transparently.
    """

    def __init__(self):
        self._read_hashes: dict = {}       # (task_id, path) → hash_str
        self._write_locks: dict = {}       # path → asyncio.Lock
        self._meta_lock: asyncio.Lock | None = None  # protects _write_locks dict

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _hash_file(path: str) -> str | None:
        """Read file from disk and return MD5 hash. None if file doesn't exist."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return hashlib.md5(f.read().encode("utf-8", errors="replace")).hexdigest()
        except Exception:
            return None

    async def _get_write_lock(self, path: str) -> asyncio.Lock:
        if self._meta_lock is None:
            self._meta_lock = asyncio.Lock()
        async with self._meta_lock:
            if path not in self._write_locks:
                self._write_locks[path] = asyncio.Lock()
            return self._write_locks[path]

    # ── public API ────────────────────────────────────────────────────────────

    def record_read(self, task_id: str, path: str, content: str) -> None:
        """Record hash of file content at read time. Called after successful read_file."""
        h = self._hash(content)
        self._read_hashes[(task_id, path)] = h
        slog.debug(f"   [Ledger] task {task_id[:8]} read '{path}' @ {h[:8]}")

    def check_write(self, task_id: str, path: str) -> tuple:
        """
        Before a write: check whether the file changed since this task last read it.
        Returns (ok: bool, reason: str).
          ok=True, reason=""   → safe to proceed (acquire write lock and write)
          ok=False, reason=... → file modified by another task; Clara must re-read first
        """
        key = (task_id, path)
        if key not in self._read_hashes:
            return True, ""  # pure write — no prior read by this task, skip hash check

        stored = self._read_hashes[key]
        current = self._hash_file(path)

        if current is None:
            return True, ""  # file doesn't exist yet — new file, no conflict possible

        if current != stored:
            slog.warning(
                f"   [Ledger] CONFLICT: task {task_id[:8]} write to '{path}' blocked — "
                f"file changed since read (stored={stored[:8]}, current={current[:8]})"
            )
            return False, (
                f"Write blocked: '{path}' was modified by another task since you last "
                f"read it. Re-read the file first to get the current content, then write."
            )

        return True, ""

    async def acquire_write(self, path: str, task_id: str = "") -> asyncio.Lock:
        """
        Acquire exclusive write lock for path. Caller MUST release in a try/finally.
        Suspends the coroutine cooperatively if another task holds the lock.
        """
        lock = await self._get_write_lock(path)
        await lock.acquire()
        slog.debug(f"   [Ledger] task {task_id[:8]} acquired write lock '{path}'")
        return lock

    def release_task(self, task_id: str) -> None:
        """Remove all read hashes for a completed or failed task."""
        keys = [k for k in self._read_hashes if k[0] == task_id]
        for k in keys:
            del self._read_hashes[k]
        if keys:
            slog.debug(f"   [Ledger] task {task_id[:8]} released {len(keys)} read hash(es)")


# Module-level singleton — shared across all concurrent tasks
resource_ledger = ResourceLedger()
