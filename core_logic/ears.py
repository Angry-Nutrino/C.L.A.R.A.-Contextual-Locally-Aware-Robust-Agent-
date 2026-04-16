"""
ears.py — STT thin wrapper.
Delegates entirely to VoiceCoordinator (core_logic/voice.py).
VoiceCoordinator is created and loaded by api.py at startup.
This file contains NO model loading, NO blocking calls.

Legacy _cli_listen() preserved at bottom for CLI testing only —
it requires manual instantiation and is never called by the system.
"""
from .session_logger import slog


def listen_push_to_talk() -> str | None:
    """
    Called after push-to-talk key released (voice_stop WebSocket event).
    Delegates to VoiceCoordinator.stop_recording() via api.py.
    Returns transcribed text or None.

    NOTE: In the WebSocket flow, stop_recording_async() is called directly
    on the VoiceCoordinator instance from api.py — this function is a
    reference shim for any non-WebSocket callers.
    """
    try:
        from .voice import get_voice
        v = get_voice()
        if v is None:
            slog.warning("[ears] VoiceCoordinator not loaded.")
            return None
        return v.stop_recording()
    except Exception as e:
        slog.error(f"[ears] listen_push_to_talk failed: {e}")
        return None


# ── Legacy CLI test (never called by the system) ─────────────────────────
# To test STT manually from terminal:
#   from core_logic.ears import _cli_listen
#   text = _cli_listen()
def _cli_listen() -> str | None:
    """Manual CLI test only. Requires VoiceCoordinator to be loaded first."""
    from .voice import get_voice
    v = get_voice()
    if v is None:
        print("[ears] VoiceCoordinator not loaded. Cannot listen.")
        return None
    input("Press Enter to start recording...")
    v.start_recording()
    input("Press Enter to stop recording...")
    return v.stop_recording()
