"""
session_logger.py — Per-session file logger for CLARA.

Call init_session_log() once at api.py startup.
Import `slog` anywhere else to write to the same session file.
"""
import logging
import os
from datetime import datetime

# Module-level logger — shared across all imports
slog = logging.getLogger("clara_session")


def init_session_log():
    """
    Creates a new timestamped log file under logs/ and attaches
    a FileHandler + StreamHandler to the shared logger.
    Called once at api.py startup.
    """
    if slog.handlers:
        return  # already initialized (e.g. uvicorn reload)

    slog.setLevel(logging.DEBUG)

    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(logs_dir, f"session_{timestamp}.log")

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    # File handler — everything goes here
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Stream handler — mirrors to console
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(fmt)

    slog.addHandler(fh)
    slog.addHandler(sh)

    slog.info(f"=== SESSION START === log: {os.path.basename(log_path)}")
    return log_path
