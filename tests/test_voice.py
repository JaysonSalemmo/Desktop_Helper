import numpy as np

from src.voice import voice


def test_short_audio_skips_model():
    t = voice.Transcriber()
    assert t.transcribe(np.zeros(100, dtype=np.float32)) == ""
    assert t._model is None  # guard returned before any whisper load


class FakeProcess:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return None if not self.terminated else 0

    def terminate(self):
        self.terminated = True


def test_speak_interrupts_previous_utterance(monkeypatch):
    spawned = []

    def fake_popen(cmd):
        proc = FakeProcess()
        spawned.append((cmd, proc))
        return proc

    monkeypatch.setattr(voice.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(voice, "_say_process", None)

    voice.speak("first reply")
    voice.speak("second reply")

    assert [cmd for cmd, _ in spawned] == [["say", "first reply"], ["say", "second reply"]]
    assert spawned[0][1].terminated       # first utterance was cut off
    assert not spawned[1][1].terminated   # second is speaking
