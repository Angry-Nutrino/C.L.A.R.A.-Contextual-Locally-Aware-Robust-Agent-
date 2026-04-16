"""
kokoro_mouth.py — TTS thin wrapper.
Delegates entirely to VoiceCoordinator (core_logic/voice.py).
VoiceCoordinator is created and loaded by api.py at startup.
This file contains NO model loading, NO blocking calls, NO time.sleep().
"""
from .session_logger import slog


def speak(text: str, block: bool = True) -> None:
    """
    Speak text via Kokoro TTS through VoiceCoordinator.
    block=True: waits for playback to complete.
    block=False: fires and returns immediately.
    No-op if VoiceCoordinator is not loaded.
    """
    try:
        from .voice import get_voice
        v = get_voice()
        if v is None:
            slog.warning(f"[mouth] VoiceCoordinator not loaded. Cannot speak: {text[:40]}")
            return
        v.speak(text, block=block)
    except Exception as e:
        slog.error(f"[mouth] speak() failed: {e}")


def interrupt() -> None:
    """Stop current TTS playback immediately."""
    try:
        from .voice import get_voice
        v = get_voice()
        if v:
            v.interrupt_speech()
    except Exception as e:
        slog.error(f"[mouth] interrupt() failed: {e}")


def is_speaking() -> bool:
    """Returns True if TTS is currently playing."""
    try:
        from .voice import get_voice
        v = get_voice()
        return v.is_speaking() if v else False
    except Exception:
        return False
