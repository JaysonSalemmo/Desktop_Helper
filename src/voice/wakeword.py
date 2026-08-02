"""
Wake word — "hey Jarvis" heard by an always-on local listener.

Completes the hands-free loop: wake phrase → chirp → capture until you stop
talking → whisper → dispatcher → spoken reply. Only the trigger is new;
everything downstream is the existing voice pipeline.

Fully local (openWakeWord ONNX, ~1MB model, a few % CPU). Verified offline:
synthesized "hey jarvis" scores 0.999, silence 0.000 — threshold 0.6.

The listener drains the mic even while paused (buffer health) but doesn't
score; a `gate` callable lets the app suppress detection while the assistant
itself is speaking (no self-triggering) or otherwise busy.
"""
import subprocess
import threading

import numpy as np

RATE = 16_000
FRAME = 1280  # 80ms — openWakeWord's native frame size

CHIRP = "/System/Library/Sounds/Pop.aiff"


def chirp() -> None:
    """Audible wake acknowledgment (non-blocking)."""
    subprocess.Popen(["afplay", CHIRP])


def capture_until_silence(read_frame=None, max_seconds: float = 10.0,
                          silence_seconds: float = 1.2) -> np.ndarray:
    """Record speech, stopping after trailing silence. Returns float32 @16k
    for whisper. `read_frame` (→ int16[FRAME]) is injectable for tests;
    default reads the microphone."""
    stream = None
    if read_frame is None:
        import sounddevice as sd
        stream = sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                                blocksize=FRAME)
        stream.start()

        def read_frame():
            data, _ = stream.read(FRAME)
            return data[:, 0]

    frames = []
    seen_speech = False
    trailing_silent = 0
    silence_frames = int(silence_seconds * RATE / FRAME)
    max_frames = int(max_seconds * RATE / FRAME)
    try:
        for _ in range(max_frames):
            frame = read_frame()
            frames.append(frame)
            rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
            if rms > 350:
                seen_speech = True
                trailing_silent = 0
            else:
                trailing_silent += 1
            if seen_speech and trailing_silent >= silence_frames:
                break
    finally:
        if stream is not None:
            stream.stop()
            stream.close()

    if not seen_speech:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames).astype(np.float32) / 32768.0


class WakeWordListener(threading.Thread):
    """Daemon thread scoring mic frames; calls `on_wake()` (from this thread —
    hop to main yourself) when the wake phrase is heard."""

    def __init__(self, on_wake, gate=None, model_name: str = "hey_jarvis_v0.1",
                 threshold: float = 0.6):
        super().__init__(daemon=True, name="wake-word")
        self.on_wake = on_wake
        self.gate = gate  # returns True → suppress detection right now
        self.model_name = model_name
        self.threshold = threshold
        self._paused = threading.Event()
        self._stopped = threading.Event()
        self.ok = None  # None starting, True live, False failed (mic/model)

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def stop(self) -> None:
        self._stopped.set()

    def run(self) -> None:
        try:
            import sounddevice as sd
            from openwakeword.model import Model
            model = Model(wakeword_models=[self.model_name],
                          inference_framework="onnx")
            # openwakeword keys predictions by the model's own name, which for a
            # custom .onnx is its basename — not the path we loaded it from. Ask
            # the loaded model rather than assuming the key is model_name.
            key = next(iter(model.models))
            stream = sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                                    blocksize=FRAME)
            stream.start()
        except Exception:
            self.ok = False
            return
        self.ok = True

        needs_reset = False
        try:
            while not self._stopped.is_set():
                frame, _ = stream.read(FRAME)  # always drain the mic
                if self._paused.is_set() or (self.gate is not None and self.gate()):
                    needs_reset = True  # buffer went stale while suppressed
                    continue
                if needs_reset:
                    model.reset()
                    needs_reset = False
                score = model.predict(frame[:, 0])[key]
                if score >= self.threshold:
                    model.reset()
                    self.on_wake()
                    needs_reset = True  # skip our own chirp tail on resume
        finally:
            stream.stop()
            stream.close()
