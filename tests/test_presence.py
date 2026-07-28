"""The header's living indicator — state transitions and the battery guarantee."""
import pytest

pytest.importorskip("Quartz", reason="macOS/PyObjC only")

from src.menubar.presence import Presence  # noqa: E402


def _keys(layer):
    return sorted(layer.animationKeys() or [])


@pytest.fixture
def presence():
    return Presence(30, 16)


def test_starts_dormant_and_breathing(presence):
    assert presence._state == "dormant"
    # dormant must still read as alive: travelling wave + slow opacity swell
    assert "travel" in _keys(presence._front)
    assert "breathe" in _keys(presence._front)
    assert len(presence.view.layer().sublayers()) == 2
    assert presence.view.layer().masksToBounds()  # clips the over-wide path


def test_active_states_drop_the_breathing(presence):
    for state in ("listening", "thinking", "speaking"):
        presence.set_state(state)
        assert presence._state == state
        assert "travel" in _keys(presence._front)
        assert "breathe" not in _keys(presence._front), f"{state} should not breathe"


def test_level_scales_the_wave(presence):
    presence.set_state("listening")
    quiet = presence._front.affineTransform().d
    presence.set_level(1.0)
    loud = presence._front.affineTransform().d
    assert loud > quiet, "louder input must raise the wave"
    presence.set_level(0.0)
    assert presence._front.affineTransform().d == pytest.approx(quiet)


def test_leaving_listening_resets_the_level(presence):
    presence.set_state("listening")
    presence.set_level(0.9)
    presence.set_state("dormant")
    # a stale mic level must not keep the dormant wave inflated
    assert presence._level == 0.0


def test_suspend_removes_every_animation(presence):
    # the battery guarantee: a hidden window costs nothing
    presence.suspend()
    assert _keys(presence._front) == []
    assert _keys(presence._back) == []
    # and stays quiet even if states/levels are pushed while hidden
    presence.set_state("thinking")
    presence.set_level(1.0)
    assert _keys(presence._front) == []
    presence.resume()
    assert "travel" in _keys(presence._front)


def test_bad_input_is_ignored(presence):
    presence.set_state("nonsense")
    assert presence._state == "dormant"
    presence.set_level("loud")   # must not raise
    presence.set_level(None)
    presence.set_level(5.0)
    assert 0.0 <= presence._level <= 1.0
