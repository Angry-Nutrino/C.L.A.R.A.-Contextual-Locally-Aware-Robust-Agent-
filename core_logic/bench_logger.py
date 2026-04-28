"""
bench_logger.py — Request-level performance logger for CLARA.

Logs only user-facing requests (source == "user") to benchmarks/bench_<date>.log
One line per request, tab-separated for easy reading and parsing.

Format:
    HH:MM:SS  MODE     TOOL              TOTAL_MS  INTERP_MS  EXEC_MS   PROMPT  COMPLETION  TOTAL  CACHED  query_preview
    18:31:14  FAST     date_time         1243      512        731       124     45          169    0       what time is it?
    18:33:12  CHAT     -                 8103      487        7616      587     2341        2928   0       haha a friend of mine...
    18:08:11  DELIB    vision_tool       26420     623        25797     1234    5678        6912   1234    oh did you like her?

Columns:
    TIME       — wall clock time of request start
    MODE       — FAST / CHAT / DELIB
    TOOL       — tool name for FAST, "-" for CHAT/DELIB
    TOTAL_MS   — full request duration (interpret + execute + format)
    INTERP_MS  — time spent in Interpreter (Grok non-reasoning call)
    EXEC_MS    — time spent in execution (tool + format, or chat stream, or ReAct loop)
    PROMPT     — total prompt tokens
    COMPLETION — total completion tokens
    TOTAL      — total tokens (prompt + completion)
    CACHED     — cached tokens
    QUERY      — first 50 chars of the user query
"""

import os
import time
from datetime import datetime


# ── Module-level file handle ──────────────────────────────────────────────────
_bench_file = None


def init_bench_log(bench_dir: str = "benchmarks") -> None:
    """Call once at startup from api.py after session logger is initialized."""
    global _bench_file
    os.makedirs(bench_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(bench_dir, f"bench_{date_str}.log")
    _bench_file = open(path, "a", encoding="utf-8", buffering=1)

    # Write header if file is new (empty)
    if os.path.getsize(path) == 0:
        _bench_file.write(
            f"{'TIME':<10}{'MODE':<8}{'TOOL':<20}{'TOTAL_MS':<12}"
            f"{'INTERP_MS':<12}{'EXEC_MS':<12}{'PROMPT':<10}{'COMPLETION':<12}"
            f"{'TOTAL':<10}{'CACHED':<10}QUERY\n"
        )
        _bench_file.write("-" * 130 + "\n")


def log_request(
    mode: str,
    tool: str | None,
    total_ms: int,
    interp_ms: int,
    exec_ms: int,
    query: str,
    token_usage=None,
) -> None:
    """Write one benchmark record. No-op if bench log not initialized."""
    if _bench_file is None:
        return
    time_str = datetime.now().strftime("%H:%M:%S")
    tool_str  = tool if tool else "-"
    mode_str  = {"FAST": "FAST", "CHAT": "CHAT", "DELIBERATE": "DELIB"}.get(mode, mode)
    query_str = query[:50].replace("\n", " ").replace("\t", " ")

    tokens_str = ""
    if token_usage:
        tokens_str = (
            f"{token_usage.prompt_tokens:<10}"
            f"{token_usage.completion_tokens:<12}"
            f"{token_usage.total_tokens:<10}"
            f"{token_usage.cached_tokens:<10}"
        )

    _bench_file.write(
        f"{time_str:<10}{mode_str:<8}{tool_str:<20}{total_ms:<12}"
        f"{interp_ms:<12}{exec_ms:<12}{tokens_str}{query_str}\n"
    )


def close_bench_log() -> None:
    global _bench_file
    if _bench_file:
        _bench_file.close()
        _bench_file = None


# ── Context manager for timing a block ───────────────────────────────────────
class Timer:
    """Simple wall-clock timer. Usage: t = Timer(); ...; ms = t.elapsed_ms()"""
    def __init__(self):
        self._start = time.perf_counter()

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._start) * 1000)
