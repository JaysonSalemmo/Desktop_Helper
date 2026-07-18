"""
Voice input: mic capture (sounddevice) + local transcription (faster-whisper).

Frontend-agnostic — the menu bar app (and later the TUI) uses this as:
    recorder.start()  →  user speaks  →  audio = recorder.stop()
    text = transcriber.transcribe(audio)
    ...text feeds the dispatcher exactly like a typed message.

Everything runs locally: no audio ever leaves the machine. The whisper model
(base.en, int8, CPU) downloads once (~75MB) on first use and loads in ~1s
after that — call `transcriber.warm_up()` in the background at startup so the
first real transcription isn't slow.

First mic use triggers the macOS microphone permission prompt, attributed to
the hosting terminal (or, later, the bundled .app).
"""
import subprocess
import threading

import numpy as np

SAMPLE_RATE = 16_000  # what whisper expects
WHISPER_MODEL = "base.en"

_say_process: subprocess.Popen | None = None


def is_speaking() -> bool:
    """True while a `say` utterance is still playing — the wake-word listener
    gates on this so the assistant can't wake itself."""
    return _say_process is not None and _say_process.poll() is None


def speak(text: str) -> None:
    """Speak text aloud via macOS `say` (offline, built-in voices).

    Non-blocking; a new utterance interrupts the previous one so queued
    replies never talk over each other."""
    global _say_process
    if _say_process is not None and _say_process.poll() is None:
        _say_process.terminate()
    _say_process = subprocess.Popen(["say", text])


class VoiceRecorder:
    """Accumulates mono 16kHz audio between start() and stop()."""

    def __init__(self):
        self._stream = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd

        self._chunks = []

        def callback(indata, frames, time_info, status):
            with self._lock:
                self._chunks.append(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)
        self._stream.stop()
        self._stream.close()
        self._stream = None
        with self._lock:
            chunks, self._chunks = self._chunks, []
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)


class Transcriber:
    """Lazy faster-whisper wrapper. Thread-safe single-load."""

    def __init__(self, model_name: str = WHISPER_MODEL):
        self.model_name = model_name
        self._model = None
        self._load_lock = threading.Lock()

    def warm_up(self) -> None:
        """Load (and on first run, download) the whisper model."""
        with self._load_lock:
            if self._model is None:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(self.model_name, device="cpu",
                                           compute_type="int8")

    def transcribe(self, audio: np.ndarray) -> str:
        """audio: mono float32 at 16kHz (or a path). Returns joined text."""
        if isinstance(audio, np.ndarray) and audio.size < SAMPLE_RATE // 4:
            return ""  # under a quarter second — nothing to hear
        self.warm_up()
        segments, _info = self._model.transcribe(audio, language="en", beam_size=5)
        return " ".join(seg.text.strip() for seg in segments).strip()
