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
import math
import os
import subprocess
import tempfile
import threading
import time
import wave

import numpy as np

SAMPLE_RATE = 16_000  # what whisper expects
WHISPER_MODEL = "base.en"
_TTS_RATE = 22_050    # `say` synthesis rate for the level-metered path

_say_process: subprocess.Popen | None = None
_speaking = threading.Event()      # set while WE are playing an utterance
_playback_stop = threading.Event()  # asks that playback to stop


def is_speaking() -> bool:
    """True while an utterance is still playing — the wake-word listener gates
    on this so the assistant can't wake itself. Covers both paths: our
    level-metered playback and the plain `say` fallback."""
    if _speaking.is_set():
        return True
    return _say_process is not None and _say_process.poll() is None


def speak(text: str, on_level=None) -> None:
    """Speak text aloud via macOS `say` (offline, built-in voices).

    Non-blocking; a new utterance interrupts the previous one so queued
    replies never talk over each other.

    With `on_level(0..1)`, the utterance is synthesized to a file and played
    back by us so the REAL amplitude can drive the UI — the orb then moves with
    the volume of what's being said, not merely "on". macOS gives no hook into
    `say`'s output level, and capturing system audio would need a loopback
    device, so owning playback is the only way to get a true signal. Any failure
    falls back to plain `say`: speaking must never break for the sake of a
    visual."""
    global _say_process
    stop_speaking()
    if on_level is None:
        _say_process = subprocess.Popen(["say", text])
        return
    threading.Thread(target=_speak_with_levels, args=(text, on_level),
                     daemon=True).start()


def stop_speaking() -> None:
    """Cut off whatever is being said (a new reply supersedes the old one)."""
    global _say_process
    _playback_stop.set()
    if _say_process is not None and _say_process.poll() is None:
        _say_process.terminate()


def _speak_with_levels(text: str, on_level) -> None:
    global _say_process
    tmp = None
    try:
        import sounddevice as sd

        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        # LEI16 WAV: readable by the stdlib `wave` module (aifc was removed in
        # Python 3.13, so AIFF — say's default — is no longer parseable here)
        subprocess.run(
            ["say", "-o", tmp, "--file-format=WAVE",
             f"--data-format=LEI16@{_TTS_RATE}", text],
            check=True, capture_output=True, timeout=60)
        with wave.open(tmp, "rb") as wav:
            channels = wav.getnchannels()
            rate = wav.getframerate()
            raw = wav.readframes(wav.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        if audio.size == 0:
            return

        _playback_stop.clear()
        _speaking.set()
        cursor = 0

        def callback(outdata, frames, time_info, status):
            nonlocal cursor
            if _playback_stop.is_set():
                raise sd.CallbackStop
            chunk = audio[cursor:cursor + frames]
            cursor += frames
            if chunk.size < frames:
                outdata[:chunk.size, 0] = chunk
                outdata[chunk.size:, 0] = 0.0
                _emit_level(on_level, chunk)
                raise sd.CallbackStop
            outdata[:, 0] = chunk
            _emit_level(on_level, chunk)

        with sd.OutputStream(samplerate=rate, channels=1, dtype="float32",
                             callback=callback, blocksize=1024) as stream:
            while stream.active and not _playback_stop.is_set():
                time.sleep(0.05)
    except Exception:
        # synthesis/playback unavailable — say it the plain way
        try:
            _say_process = subprocess.Popen(["say", text])
        except Exception:
            pass
    finally:
        _speaking.clear()
        try:
            on_level(0.0)  # leave the UI at rest, never mid-swell
        except Exception:
            pass
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


SPECTRUM_BANDS = 56  # must match the orb's bar count


def spectrum(chunk, bands: int = SPECTRUM_BANDS):
    """Per-band magnitudes 0..1 for a circular visualiser, or None.

    Log-spaced bands, because speech energy is bunched into the low end and
    linear bins would leave most of the ring dead. Normalised to the frame's
    own peak so quiet speech still shows shape — loudness is carried separately
    by the RMS level."""
    if chunk.size < 64:
        return None
    windowed = chunk * np.hanning(chunk.size)
    mag = np.abs(np.fft.rfft(windowed))
    if mag.size <= bands:
        return None
    edges = np.geomspace(2, mag.size - 1, bands + 1).astype(int)
    out = np.empty(bands, dtype=np.float32)
    for i in range(bands):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        out[i] = mag[lo:hi].mean()
    peak = float(out.max())
    if peak <= 1e-9:
        return None
    return (out / peak).tolist()


def _emit_level(on_level, chunk) -> None:
    if chunk.size == 0:
        return
    # gain is deliberately below the saturation point: pinned at 1.0 the orb
    # holds one size and reads as merely "on" instead of tracking the words
    rms = float(np.sqrt(np.mean(np.square(chunk))))
    try:
        on_level(min(1.0, math.sqrt(rms) * 1.8), spectrum(chunk))
    except Exception:
        pass  # a UI hiccup must never interrupt speech


class VoiceRecorder:
    """Accumulates mono 16kHz audio between start() and stop()."""

    def __init__(self, on_level=None):
        """on_level(0..1, bands) is called from the AUDIO thread with each
        chunk's loudness and its per-band spectrum — the signals that make the
        UI react to the user's actual voice instead of a canned loop. Anything
        it touches must be main-thread-hopped by the caller."""
        self._stream = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._on_level = on_level

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd

        self._chunks = []

        def callback(indata, frames, time_info, status):
            mono = indata[:, 0].copy()
            with self._lock:
                self._chunks.append(mono)
            if self._on_level is not None:
                # RMS → a perceptual-ish 0..1. Speech RMS sits well below 1.0,
                # so the sqrt lifts normal talking into the visible range.
                rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
                try:
                    self._on_level(min(1.0, math.sqrt(rms) * 2.1), spectrum(mono))
                except Exception:
                    pass  # a UI hiccup must never break the recording

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
