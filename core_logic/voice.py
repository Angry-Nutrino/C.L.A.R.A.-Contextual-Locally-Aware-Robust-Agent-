"""
voice.py — Voice I/O coordinator for CLARA.
Manages push-to-talk STT, TTS playback, and acknowledgment logic.
Phase 1: push-to-talk + blocking TTS with interrupt flag.
"""

import asyncio
import threading
import os
import tempfile
import time

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

from .session_logger import slog

# ── Config ────────────────────────────────────────────────────────────────
WHISPER_MODEL   = "medium.en"
WHISPER_DEVICE  = "cuda"
WHISPER_COMPUTE = "int8"
SAMPLE_RATE     = 16000
KOKORO_VOICE    = "af_bella"
KOKORO_SPEED    = 1.05
KOKORO_LANG     = "en-us"

ACKNOWLEDGMENTS = {
    "fast_tool":          "On it.",
    "deliberate":         "Give me a moment.",
    "deliberate_complex": "This will take a moment.",
}


class VoiceCoordinator:
    """
    Owns STT model, TTS model, push-to-talk state, and playback control.
    Thread-safe for audio playback. Async-safe for STT/pipeline integration.
    """

    def __init__(self, mic_name_substring: str = None):
        self._whisper    = None
        self._kokoro     = None
        self._recording  = False
        self._speaking   = False
        self._stop_flag  = threading.Event()
        self._audio_buf  = []
        self._enabled    = False
        self._on_speaking_change = None
        self._mic_name   = mic_name_substring
        self._stream     = None  # sounddevice InputStream, kept open

    def load(self):
        """Load Whisper and Kokoro models. Called once at startup."""
        slog.info("[Voice] Loading Faster-Whisper...")
        self._whisper = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE,
        )
        slog.info("[Voice] Whisper loaded.")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        onnx_path   = os.path.join(current_dir, "models", "kokoro-v0_19.onnx")
        voices_path = os.path.join(current_dir, "models", "voices.bin")
        slog.info("[Voice] Loading Kokoro TTS...")
        self._kokoro = Kokoro(onnx_path, voices_path)
        slog.info("[Voice] Kokoro loaded.")

        device_index = self._find_mic(self._mic_name)
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device_index,
            callback=self.audio_callback,
        )
        self._stream.start()
        slog.info(f"[Voice] Microphone stream open (device: {device_index}).")
        self._enabled = True

    def _find_mic(self, name_substring: str = None):
        """Return device index matching name_substring, or None for default."""
        if not name_substring:
            return None
        try:
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if name_substring.lower() in d["name"].lower() and d["max_input_channels"] > 0:
                    slog.info(f"[Voice] Mic found: '{d['name']}' at index {i}")
                    return i
        except Exception:
            pass
        slog.warning(f"[Voice] Mic '{name_substring}' not found. Using default.")
        return None

    def unload(self):
        """Free models and close audio stream."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._whisper = None
        self._kokoro  = None
        self._enabled = False
        slog.info("[Voice] Models and stream unloaded.")

    def set_speaking_callback(self, callback):
        """Register callback(is_speaking: bool) for UI waveform animation."""
        self._on_speaking_change = callback

    # ── Push-to-talk STT ──────────────────────────────────────────────────

    def start_recording(self):
        """Call when push-to-talk key is pressed."""
        if not self._enabled:
            return
        self._audio_buf = []
        self._recording = True
        slog.info("[Voice] Recording started.")

    def stop_recording(self) -> str | None:
        """
        Call when push-to-talk key is released.
        Returns transcribed text or None on silence/error.
        Blocks until transcription is complete (~200-400ms).
        """
        if not self._enabled or not self._recording:
            return None
        self._recording = False

        if not self._audio_buf:
            return None

        try:
            audio = np.concatenate(self._audio_buf, axis=0)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            import soundfile as sf
            sf.write(tmp_path, audio, SAMPLE_RATE)
            segments, _ = self._whisper.transcribe(
                tmp_path, beam_size=5,
                initial_prompt="CLARA, Alkama, orchestrator, Grok"
            )
            text = " ".join(s.text for s in segments).strip()
            os.unlink(tmp_path)
            slog.info(f"[Voice] STT: '{text}'")
            return text if text else None
        except Exception as e:
            slog.error(f"[Voice] STT error: {e}")
            return None

    def audio_callback(self, indata, frames, time_info, status):
        """sounddevice InputStream callback — buffers audio while recording."""
        if self._recording:
            self._audio_buf.append(indata.copy())

    async def stop_recording_async(self) -> str | None:
        """Async version — runs Whisper in thread to avoid blocking event loop."""
        return await asyncio.to_thread(self.stop_recording)

    # ── TTS Playback ──────────────────────────────────────────────────────

    def speak(self, text: str, block: bool = True) -> None:
        """
        Speak text via Kokoro TTS.
        If already speaking, skips (no clashing audio).
        block=True: waits for playback to finish.
        block=False: returns immediately (playback in background thread).
        """
        if not self._enabled or not self._kokoro or not text.strip():
            return
        if self._speaking:
            slog.info("[Voice] Already speaking — skipping overlapping TTS.")
            return
        self._stop_flag.clear()
        if block:
            self._play_speech(text)
        else:
            threading.Thread(
                target=self._play_speech, args=(text,), daemon=True
            ).start()

    def _play_speech(self, text: str) -> None:
        try:
            self._speaking = True
            if self._on_speaking_change:
                self._on_speaking_change(True)

            samples, sample_rate = self._kokoro.create(
                text, voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang=KOKORO_LANG
            )
            duration = len(samples) / sample_rate

            sd.play(samples, sample_rate)
            elapsed  = 0.0
            interval = 0.05
            while elapsed < duration + 0.1:
                if self._stop_flag.is_set():
                    sd.stop()
                    break
                time.sleep(interval)
                elapsed += interval
        except Exception as e:
            slog.error(f"[Voice] TTS error: {e}")
        finally:
            self._speaking = False
            if self._on_speaking_change:
                self._on_speaking_change(False)

    def interrupt_speech(self) -> None:
        """Stop current TTS playback immediately."""
        if self._speaking:
            self._stop_flag.set()
            sd.stop()
            slog.info("[Voice] Speech interrupted.")

    def is_speaking(self) -> bool:
        return self._speaking

    def is_enabled(self) -> bool:
        return self._enabled

    # ── Acknowledgment ────────────────────────────────────────────────────

    def get_acknowledgment(self, interpreted: dict, mode: str) -> str | None:
        """Return acknowledgment text or None (speak nothing)."""
        if mode == "FAST" and interpreted.get("tool"):
            return ACKNOWLEDGMENTS["fast_tool"]
        if mode == "FAST":
            return None
        confidence = interpreted.get("confidence", 0.5)
        if confidence >= 0.75:
            return ACKNOWLEDGMENTS["deliberate"]
        return ACKNOWLEDGMENTS["deliberate_complex"]


# ── Module-level singleton ────────────────────────────────────────────────
_voice: VoiceCoordinator | None = None


def get_voice() -> VoiceCoordinator | None:
    return _voice


def set_voice(v: VoiceCoordinator) -> None:
    global _voice
    _voice = v
