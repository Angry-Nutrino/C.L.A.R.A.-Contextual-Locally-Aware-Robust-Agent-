"""
voice.py — Voice I/O coordinator for CLARA.
Manages push-to-talk STT, TTS playback, and acknowledgment logic.
Phase 1: push-to-talk + blocking TTS with interrupt flag.
"""

import asyncio
import threading
import queue
import time
import re
import os
import tempfile

import sounddevice as sd
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

from .session_logger import slog

# ── Config ────────────────────────────────────────────────────────────────
WHISPER_MODEL   = "medium.en"
WHISPER_DEVICE  = "cuda"
WHISPER_COMPUTE = "int8"
SAMPLE_RATE     = 16000
KOKORO_VOICE    = "af_sky"
KOKORO_SPEED    = 1.1
KOKORO_LANG     = "en-us"
KOKORO_SR       = 24000   # Kokoro v0.19 fixed output sample rate
PLAY_CHUNK_FRAMES = 4800  # 0.2s chunks at 24kHz — stop_flag checked between each

ACKNOWLEDGMENTS = {
    "fast_tool":          "On it.",
    "deliberate":         "Give me a moment.",
    "deliberate_complex": "This will take a moment.",
}


class VoiceCoordinator:
    """
    Owns STT model, TTS model, push-to-talk state, and playback control.

    Audio architecture:
      - InputStream  (self._in_stream):  always-open mic capture, feeds audio_callback
      - OutputStream (self._out_stream): always-open speaker, written via stream.write()

    Both streams are opened once at load() and closed at unload().
    They are completely independent — no global sd.play()/sd.stop() calls that
    could disrupt the mic stream via Windows WASAPI device resets.
    """

    MAX_RECORD_SECONDS = 60

    def __init__(self, mic_name_substring: str = None):
        self._whisper    = None
        self._kokoro     = None
        self._recording  = False
        self._speak_lock = threading.Lock()
        self._speaking   = False
        self._stop_flag  = threading.Event()
        self._audio_buf  = []
        self._enabled    = False
        self._on_speaking_change = None
        self._mic_name   = mic_name_substring
        self._in_stream  = None   # InputStream — mic capture
        self._out_stream = None   # OutputStream — TTS playback

    def load(self):
        """Load Whisper and Kokoro models, open mic + speaker streams."""
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

        # kokoro-onnx GPU detection is broken — it checks find_spec("onnxruntime-gpu")
        # which always returns None (hyphens are invalid in Python module names).
        # We replace sess directly with a CUDA-enabled session.
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                self._kokoro.sess = ort.InferenceSession(
                    onnx_path,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                slog.info("[Voice] Kokoro sess upgraded to CUDA (~10x faster synthesis).")
            else:
                slog.info("[Voice] CUDAExecutionProvider unavailable — Kokoro on CPU.")
        except Exception as e:
            slog.warning(f"[Voice] Kokoro CUDA upgrade skipped: {e}")

        # Warmup: first ONNX inference is always slow due to JIT compilation.
        # Run it now so the first real speak() call is fast.
        try:
            slog.info("[Voice] Warming up Kokoro...")
            self._kokoro.create("Hello.", voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang=KOKORO_LANG)
            slog.info("[Voice] Kokoro warmup complete.")
        except Exception as e:
            slog.warning(f"[Voice] Kokoro warmup failed: {e}")

        device_index = self._find_mic(self._mic_name)
        self._in_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device_index,
            callback=self.audio_callback,
        )
        self._in_stream.start()
        slog.info(f"[Voice] Microphone stream open (device: {device_index}).")

        # Persistent output stream — never closed between speak() calls,
        # so Windows WASAPI never resets the audio device unexpectedly.
        self._out_stream = sd.OutputStream(
            samplerate=KOKORO_SR,
            channels=1,
            dtype="float32",
        )
        self._out_stream.start()
        slog.info("[Voice] Speaker stream open.")

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
        """Free models and close audio streams."""
        self._enabled = False
        for stream in (self._in_stream, self._out_stream):
            if stream:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
        self._in_stream  = None
        self._out_stream = None
        self._whisper = None
        self._kokoro  = None
        slog.info("[Voice] Models and streams unloaded.")

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
            slog.warning(f"[Voice] stop_recording skipped — enabled={self._enabled} recording={self._recording}")
            return None
        self._recording = False

        buf_len = len(self._audio_buf)
        if not self._audio_buf:
            slog.warning("[Voice] stop_recording: audio buffer empty — nothing captured.")
            return None

        slog.info(f"[Voice] stop_recording: transcribing {buf_len} chunks...")
        try:
            audio = np.concatenate(self._audio_buf, axis=0)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
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
            if len(self._audio_buf) * frames / SAMPLE_RATE > self.MAX_RECORD_SECONDS:
                self._recording = False
                slog.warning("[Voice] Max recording duration reached — auto-stopped.")

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
            slog.warning(f"[Voice] speak() skipped — enabled={self._enabled} kokoro={self._kokoro is not None} text_empty={not text.strip()}")
            return
        with self._speak_lock:
            if self._speaking:
                slog.info("[Voice] Already speaking — skipping overlapping TTS.")
                return
            self._speaking = True
        self._stop_flag.clear()
        if block:
            self._play_speech(text)
        else:
            threading.Thread(
                target=self._play_speech, args=(text,), daemon=True
            ).start()

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Convert numbers, currency, symbols to speakable form."""
        try:
            from num2words import num2words

            def _currency(m):
                symbol = m.group(1)
                integer_str = m.group(2).replace(',', '')
                decimal_str = m.group(3)
                name = {"$": "dollars", "£": "pounds", "€": "euros"}.get(symbol, "dollars")
                cent_name = {"$": "cents", "£": "pence", "€": "cents"}.get(symbol, "cents")
                result = num2words(int(integer_str)) + f" {name}"
                if decimal_str:
                    cents = int(decimal_str.ljust(2, '0')[:2])
                    if cents:
                        result += f" and {num2words(cents)} {cent_name}"
                return result

            def _big_number(m):
                return num2words(int(m.group().replace(',', '')))

            def _percent(m):
                val = float(m.group(1))
                return num2words(val).replace(' point zero', '') + ' percent'

            text = re.sub(r'([$£€])(\d{1,3}(?:,\d{3})*)(?:\.(\d{1,2}))?', _currency, text)
            text = re.sub(r'\b\d{1,3}(?:,\d{3})+\b', _big_number, text)
            text = re.sub(r'(\d+(?:\.\d+)?)\s*%', _percent, text)
        except ImportError:
            pass

        text = re.sub(r'\be\.g\.\B', 'for example', text)
        text = re.sub(r'\bi\.e\.\B', 'that is', text)
        text = re.sub(r'\betc\.\B', 'etcetera', text)
        text = re.sub(r'\bvs\.', 'versus', text)
        return text

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove markdown formatting so Kokoro doesn't speak symbols."""
        text = re.sub(r'```\w*\n[\s\S]*?```', ' [code block] ', text)
        text = re.sub(r'`[^`]+`', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\|[^\n]+\|', '', text)
        text = re.sub(r'\n{2,}', '. ', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    @staticmethod
    def _split_sentences(text: str) -> list:
        """
        Split into sentences for pipelined synthesis.
        The first segment is additionally sub-split at the first clause boundary
        (comma/semicolon/em-dash) if it's long — synthesis on a shorter first
        chunk means first audio starts sooner, regardless of CPU vs GPU.
        """
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\"\(])|(?<=:)\s+(?=[A-Z])', text)
        sentences = [p.strip() for p in parts if p.strip()]

        # Sub-split first sentence at earliest clause break after 30 chars
        if sentences and len(sentences[0]) > 60:
            first = sentences[0]
            m = re.search(r'[,;—]\s*', first[30:])
            if m:
                cut = 30 + m.start()
                part1 = first[:cut].rstrip(',;—').strip()
                part2 = first[30 + m.end():].strip()
                if part1 and part2:
                    sentences = [part1, part2] + sentences[1:]

        return sentences

    def _play_speech(self, text: str) -> None:
        """
        Pipelined sentence synthesis + playback via persistent OutputStream.

        Producer thread synthesizes sentences into a queue.
        Playback loop writes audio in PLAY_CHUNK_FRAMES chunks, checking
        _stop_flag between each chunk — so interruption is responsive (~200ms).

        No sd.play() / sd.wait() / sd.stop() — those are global calls that
        can disrupt the InputStream on Windows WASAPI. stream.write() and
        stream.abort() operate only on _out_stream, leaving the mic untouched.
        """
        clean = self._normalize_text(self._strip_markdown(text))
        if not clean:
            slog.warning("[Voice] _play_speech: nothing to speak after normalization.")
            self._speaking = False
            if self._on_speaking_change:
                self._on_speaking_change(False)
            return

        sentences = self._split_sentences(clean)
        slog.info(f"[Voice] Speaking {len(sentences)} sentences from {len(clean)} chars.")
        audio_q = queue.Queue(maxsize=3)

        def _synthesizer():
            for sentence in sentences:
                if self._stop_flag.is_set():
                    break
                sentence = sentence.strip()
                if not sentence:
                    continue
                try:
                    samples, sr = self._kokoro.create(
                        sentence, voice=KOKORO_VOICE,
                        speed=KOKORO_SPEED, lang=KOKORO_LANG,
                    )
                    audio_q.put((samples, sr))
                except Exception as e:
                    slog.error(f"[Voice] Synthesis error: '{sentence[:40]}' — {e}")
            audio_q.put(None)

        threading.Thread(target=_synthesizer, daemon=True).start()

        interrupted = False
        try:
            if self._on_speaking_change:
                self._on_speaking_change(True)

            while True:
                if self._stop_flag.is_set():
                    interrupted = True
                    break
                try:
                    item = audio_q.get(timeout=10)
                except queue.Empty:
                    break
                if item is None:
                    break

                samples, sr = item
                audio = samples.astype(np.float32)

                # Write in small chunks so _stop_flag is checked every ~0.2s.
                # stream.write() blocks until the OS audio buffer accepts the chunk,
                # which naturally paces multi-sentence playback.
                pos = 0
                while pos < len(audio):
                    if self._stop_flag.is_set():
                        interrupted = True
                        break
                    chunk = audio[pos: pos + PLAY_CHUNK_FRAMES]
                    try:
                        self._out_stream.write(chunk)
                    except Exception as e:
                        slog.error(f"[Voice] write error: {e}")
                        interrupted = True
                        break
                    pos += PLAY_CHUNK_FRAMES

                if interrupted:
                    break

            # Let the last buffered chunk drain before signalling done.
            # Only wait if we weren't interrupted.
            if not interrupted and not self._stop_flag.is_set():
                drain_s = PLAY_CHUNK_FRAMES / KOKORO_SR  # ~0.2s
                time.sleep(drain_s)

        except Exception as e:
            slog.error(f"[Voice] Playback error: {e}")
            interrupted = True
        finally:
            if interrupted:
                # Abort discards buffered audio immediately without touching _in_stream.
                try:
                    self._out_stream.abort()
                    self._out_stream.start()  # restart so stream is ready for next speak()
                except Exception:
                    pass
            self._speaking = False
            if self._on_speaking_change:
                self._on_speaking_change(False)

    def interrupt_speech(self) -> None:
        """Stop current TTS playback immediately."""
        if self._speaking:
            self._stop_flag.set()
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
        if mode == "FAST" or mode == "CHAT":
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
