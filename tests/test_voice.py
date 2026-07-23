import numpy as np
import pytest

from src.voice import voice
from src.voice.wakeword import FRAME, capture_until_silence


def test_parse_combo_to_carbon_modifiers():
    from src.menubar.hotkey import combo_symbols, parse_combo

    # parse_combo → (keycode, Carbon modifier mask); Carbon masks:
    # control 0x1000, shift 0x0200, option 0x0800, command 0x0100
    assert parse_combo("<ctrl>+<space>") == (49, 0x1000)
    assert parse_combo("<ctrl>+<alt>+<space>") == (49, 0x1000 | 0x0800)
    assert parse_combo("<ctrl>+<shift>+<space>") == (49, 0x1000 | 0x0200)
    assert parse_combo("<f10>") == (109, 0)  # function keys may be bare
    with pytest.raises(ValueError):
        parse_combo("<space>")  # ordinary keys need a modifier

    # display glyphs for menu labels (display only — RegisterEventHotKey does
    # the real work, no functional keyEquivalent)
    assert combo_symbols("<ctrl>+<shift>+<space>") == "⌃⇧Space"
    assert combo_symbols("<ctrl>+<option>+<space>") == "⌃⌥Space"


def _frame_feeder(frames):
    it = iter(frames)
    return lambda: next(it, np.zeros(FRAME, dtype=np.int16))


def test_capture_stops_after_trailing_silence():
    speech = [np.full(FRAME, 3000, dtype=np.int16)] * 10   # 0.8s of "speech"
    silence = [np.zeros(FRAME, dtype=np.int16)] * 60
    audio = capture_until_silence(_frame_feeder(speech + silence))
    # captured the speech plus ~1.2s of trailing silence, not the full 10s cap
    assert 0 < len(audio) <= (10 + 16 + 2) * FRAME
    assert audio.dtype == np.float32
    assert abs(audio[:FRAME].max() - 3000 / 32768) < 1e-4


def test_capture_gives_up_on_pure_silence():
    audio = capture_until_silence(_frame_feeder([]), max_seconds=1.0)
    assert len(audio) == 0


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
