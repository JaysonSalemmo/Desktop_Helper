"""The orb front-end: state, docking, and the ephemeral-conversation model."""
import math

import pytest

pytest.importorskip("Quartz", reason="macOS/PyObjC only")

from src.menubar.orb import DOCK_SCALE, ORB_BOX, OrbView, OrbWindow  # noqa: E402


def _keys(layer):
    return sorted(layer.animationKeys() or [])


@pytest.fixture
def orb():
    return OrbView(ORB_BOX)


@pytest.fixture
def window():
    return OrbWindow(on_followup=lambda text: None)


def test_dormant_visibly_moves(orb):
    # dormant must not read as "just a grey circle" — the core breathes and the
    # bar ring turns, only slowly. Idle is resting, not switched off.
    assert "breathe" in _keys(orb._core)
    assert "spin" in _keys(orb._bars)


def test_dormant_is_calmer_than_active_states():
    from src.menubar.orb import _STATES

    dormant_period, dormant_opacity = _STATES["dormant"][1], _STATES["dormant"][2]
    for state in ("listening", "thinking", "speaking"):
        period, opacity = _STATES[state][1], _STATES[state][2]
        assert dormant_period > period, f"dormant must ripple slower than {state}"
        assert dormant_opacity < opacity, f"dormant must be fainter than {state}"


def test_active_states_spin_and_stop_breathing(orb):
    for state in ("listening", "thinking", "speaking"):
        orb.set_state(state)
        assert "spin" in _keys(orb._bars), state
        assert "breathe" not in _keys(orb._core), state


def _ring_reach(orb):
    import Quartz

    return Quartz.CGPathGetBoundingBox(orb._bars.path()).size.width / 2


def test_voice_extends_the_ring_not_the_core(orb):
    # Kai: "the circle expands to a specific size, stops — the waves should be
    # the ones reacting to my volume"
    orb.set_state("listening")
    core = orb._core.affineTransform().a
    quiet_ring = _ring_reach(orb)

    for _ in range(8):
        orb.set_level(1.0)
    assert _ring_reach(orb) > quiet_ring, "the ring answers the voice"
    assert orb._core.affineTransform().a == core, "the disc holds its size"

    for _ in range(80):
        orb.set_level(0.0)
    assert _ring_reach(orb) == pytest.approx(quiet_ring, abs=1.0)


def test_core_size_is_fixed_per_state(orb):
    sizes = {}
    for state in ("dormant", "listening", "speaking"):
        orb.set_state(state)
        sizes[state] = orb._core.affineTransform().a
        for _ in range(6):  # loud audio must not move it
            orb.set_level(1.0)
        assert orb._core.affineTransform().a == sizes[state]
    assert len(set(sizes.values())) > 1, "states still differ in size"


def test_envelope_attacks_fast_and_releases_slow(orb):
    # "I like the responsiveness but I want it smoother": smooth the SIGNAL,
    # not by adding animation lag, so onsets still land immediately
    orb.set_state("speaking")
    orb.set_level(1.0)
    after_onset = orb._level
    assert after_onset > 0.4, "a loud onset must register on the first sample"

    orb.set_level(0.0)
    assert orb._level > after_onset * 0.7, "one quiet sample must not collapse it"

    orb.set_state("listening")  # resets
    for _ in range(6):
        orb.set_level(0.8)
    assert orb._level > 0.6, "sustained speech should reach near the true level"


def test_leaving_listening_clears_a_stale_level(orb):
    orb.set_state("listening")
    orb.set_level(0.9)
    orb.set_state("dormant")
    assert orb._level == 0.0


def test_suspend_silences_everything(orb):
    orb.suspend()
    assert _keys(orb._core) == []
    assert _keys(orb._bars) == []
    orb.set_state("thinking")  # pushed while suspended → still silent
    assert _keys(orb._core) == []
    orb.resume()
    assert "spin" in _keys(orb._bars)


def test_bad_input_ignored(orb):
    orb.set_state("nonsense")
    assert orb._state == "dormant"
    orb.set_level("loud")
    orb.set_level(None)
    orb.set_level(9.0)
    assert 0.0 <= orb._level <= 1.0


def test_docking_scales_orb_and_its_hit_target(window):
    # the box never resizes (so the layer can scale smoothly) — the CLICK area
    # has to shrink with it, or the docked orb swallows transcript clicks
    assert window._orb._scale == 1.0
    assert window._hit._radius == ORB_BOX / 2
    window.expand()
    assert window._orb._scale == pytest.approx(DOCK_SCALE)
    assert window._hit._radius == pytest.approx(ORB_BOX / 2 * DOCK_SCALE)
    window.collapse()
    assert window._orb._scale == 1.0
    assert window._hit._radius == ORB_BOX / 2


def test_parking_does_not_claim_to_be_open(window):
    # a parked orb leaves "Show Chat" live — expanding is what that item does
    seen = []
    window._on_visibility = seen.append
    window.park()
    assert seen == []
    window.expand()
    assert seen == [True]
    window.collapse()
    assert seen == [True, False]


def test_message_slot_is_static_and_scrolling_pages_history(window):
    # Kai's model: the text lives in ONE fixed place; scrolling swaps which
    # message fills it, it never scrolls the text itself
    for i in (1, 2, 3):
        window.show(f"question {i}", f"answer {i}")
    frame = tuple(window._transcript.frame().origin)

    assert "answer 3" in str(window._transcript.string()), "newest fills the slot"

    window._step(1)  # scroll up → older
    assert "answer 2" in str(window._transcript.string())
    assert "1 back" in str(window._marker.stringValue())

    window._step(1)
    assert "answer 1" in str(window._transcript.string())
    window._step(1)  # already oldest — must clamp, not wrap
    assert "answer 1" in str(window._transcript.string())

    window._step(-1)  # scroll down → newer
    assert "answer 2" in str(window._transcript.string())

    assert tuple(window._transcript.frame().origin) == frame, "the slot must not move"


def test_new_answer_returns_the_slot_to_newest(window):
    window.show("old", "old answer")
    window.show("older still", "second answer")
    window._step(1)
    assert "old answer" in str(window._transcript.string())
    window.show("fresh", "fresh answer")
    assert "fresh answer" in str(window._transcript.string())
    assert str(window._marker.stringValue()) == "", "marker clears at the newest"


def test_scrolling_an_empty_history_is_harmless(window):
    window._step(1)
    window._step(-1)
    assert window._index == 0


def test_thinking_placeholder_is_replaced_by_the_answer(window):
    window.show_thinking("what's on my calendar?")
    assert "…" in str(window._transcript.string())
    window.show("what's on my calendar?", "Pottery at 4:35pm")
    text = str(window._transcript.string())
    assert "Pottery at 4:35pm" in text
    assert "…" not in text
    assert len(window._exchanges) == 1


def test_speak_action_lights_the_orb(window):
    window.set_action_state("Speak", True)
    assert window._orb._state == "listening"
    window.set_action_state("Speak", False)
    assert window._orb._state == "dormant"


def test_clear_empties_the_conversation(window):
    window.show("q", "a")
    window.clear()
    assert window._exchanges == []
    assert str(window._transcript.string()).strip() == ""


def test_stoplights_belong_to_the_surface_not_the_bare_orb(window):
    # Kai: "I still want the stoplight buttons" — but a parked orb should have
    # no chrome, so they appear only while the conversation surface is open
    def hidden():
        return [bool(window.panel.standardWindowButton_(i).isHidden())
                for i in (0, 1, 2)]

    assert hidden() == [True, True, True], "collapsed orb must show no chrome"
    window.expand()
    assert hidden() == [False, False, False], "open surface needs close/min/zoom"
    window.collapse()
    assert hidden() == [True, True, True]


def test_window_is_miniaturizable(window):
    # minimize only exists on a titled window — that's why it isn't borderless
    from AppKit import NSWindowStyleMaskMiniaturizable, NSWindowStyleMaskTitled

    mask = window.panel.styleMask()
    assert mask & NSWindowStyleMaskTitled
    assert mask & NSWindowStyleMaskMiniaturizable


def test_red_button_folds_to_the_orb_instead_of_destroying_it(window):
    window.expand()
    # the delegate vetoes the close and collapses — the presence stays put
    assert window._delegate.windowShouldClose_(window.panel) is False
    assert window._open is False


def test_level_response_is_snappy_and_wide():
    # "more responsive to volume and the words I'm saying": the 0.25s default
    # implicit animation smeared syllables together, and a small swing read as
    # merely "on". Both are pinned here so tuning can't quietly regress.
    from src.menubar.orb import LEVEL_SMOOTHING, OrbView

    assert LEVEL_SMOOTHING <= 0.1, "slower than this and single words blur"

    orb = OrbView(ORB_BOX)
    orb.set_state("speaking")
    # sustained levels, as real speech delivers — a single sample is only ever
    # part of the way there by design (that IS the envelope)
    for _ in range(8):
        orb.set_level(0.05)
    quiet = _ring_reach(orb)
    for _ in range(8):
        orb.set_level(1.0)
    loud = _ring_reach(orb)
    assert loud / quiet > 1.2, "the swing must be visible across a sentence"


def test_message_is_bottom_anchored_and_never_overflows(window):
    from src.menubar.orb import INPUT_H, MARGIN, OPEN_H, TEXT_BOTTOM, TEXT_MAX_H

    window.show("q", "a short answer")
    short = window._transcript.frame()
    assert short.origin.y == TEXT_BOTTOM, "the bottom edge is the static anchor"
    assert short.origin.y < OPEN_H / 2, "the message sits low, under the orb"
    assert short.size.height < TEXT_MAX_H

    # the live bug: the morning briefing drew straight over the ask box because
    # NSTextView doesn't clip to its frame
    window.show("long", "\n\n".join(f"line {i} of a very long reply" for i in range(20)))
    tall = window._transcript.frame()
    assert tall.origin.y == TEXT_BOTTOM, "still anchored — it grows upward"
    assert tall.size.height <= TEXT_MAX_H, "height is capped"
    assert tall.origin.y >= MARGIN + INPUT_H, "never reaches the ask box"
    assert window._transcript.layer().masksToBounds(), "overflow must be clipped"


def test_ask_box_text_is_centred(window):
    from AppKit import NSTextAlignmentCenter

    # NSTextAlignmentCenter is 1; 2 is RIGHT — passing 2 is what made every
    # message and the ask box render right-aligned in Kai's screenshot
    assert NSTextAlignmentCenter == 1
    assert window._input.alignment() == NSTextAlignmentCenter
    assert window._marker.alignment() == NSTextAlignmentCenter
    assert window._input._center_placeholder is True
    # the input fills the pill, so nothing is offset by an eyeballed inset
    assert window._input.frame().size.width == window._chrome.frame().size.width - 2 * 18


def test_collapse_expand_cycle_does_not_drift(window):
    # live bug: pressing the orb to collapse and reopening left it off-centre.
    # _orb_center() branches on self._open, and both transitions flipped that
    # flag BEFORE measuring — so each one measured the old window with the new
    # formula and shifted the frame a little further every cycle.
    window.expand()
    first = tuple(window.panel.frame().origin)
    for _ in range(3):
        window.collapse()
        window.expand()
    assert tuple(window.panel.frame().origin) == first


def test_replies_fade_in(window):
    # "fade in the response" — a new answer animates, it doesn't blink in
    window.show("hello", "Hey there — what do you need?")
    assert "swap" in (window._transcript.layer().animationKeys() or [])


def test_ring_is_a_radial_bar_visualiser(orb):
    # Kai's reference image: spokes all the way around the circle, each rising
    # with a frequency band — not rings, not arcs
    import Quartz

    from src.menubar.orb import BARS

    box = Quartz.CGPathGetBoundingBox(orb._bars.path())
    assert 0.8 < box.size.width / box.size.height < 1.25, "the ring is round"
    assert box.size.width > orb._r0 * 2, "bars extend beyond the core"

    orb.set_state("speaking")
    spectrum = [abs(math.sin(i / 4.0)) for i in range(BARS)]
    for _ in range(6):
        orb.set_level(0.8, spectrum)
    assert len(set(round(b, 2) for b in orb._bands)) > 10, (
        "bars must move independently with the spectrum, not as one ring")


def test_bars_settle_back_without_audio(orb):
    from src.menubar.orb import BARS

    orb.set_state("speaking")
    orb.set_level(1.0, [1.0] * BARS)
    loud = sum(orb._bands) / BARS
    for _ in range(40):
        orb.set_level(0.0, None)
    assert sum(orb._bands) / BARS < loud, "the ring relaxes when the voice stops"


def test_layout_recentres_at_any_window_size(window):
    # the live bug: minimise → reopen left the orb off-centre because layout was
    # baked from constants instead of the window's actual size
    from AppKit import NSMakeRect

    from src.menubar.orb import ORB_BOX

    window.expand()
    for width, height in ((460, 460), (600, 520), (380, 430)):
        window.panel.setFrame_display_(NSMakeRect(100, 100, width, height), False)
        window.relayout()
        assert window._hit.frame().origin.x == pytest.approx(width / 2 - ORB_BOX / 2)


def test_loading_state_sweeps_then_settles(orb):
    # "a bar moving around the circle until the model is fully loaded"
    orb.set_state("loading")
    assert not orb._progress.isHidden()
    assert "sweep" in _keys(orb._progress)

    orb.set_state("dormant")
    # it dissolves rather than blinking out — the lap animation stops at once
    # and the arc fades, hiding itself when the fade completes
    assert _keys(orb._progress) == [], "the lap stops immediately"
    assert orb._progress.opacity() == 0.0, "…and it fades away"


def test_expand_places_the_orb_for_the_TARGET_size(window):
    # the bug that survived two fixes: expand() laid out from the window's LIVE
    # bounds, which mid-animation are still the collapsed size — so the orb was
    # pinned to the small geometry and ended up off-centre once it grew
    from src.menubar.orb import OPEN_W, ORB_BOX

    window.expand()
    assert window._hit.frame().origin.x == pytest.approx(OPEN_W / 2 - ORB_BOX / 2)
    for _ in range(3):
        window.collapse()
        window.expand()
        assert window._hit.frame().origin.x == pytest.approx(OPEN_W / 2 - ORB_BOX / 2)


def test_scaling_grows_from_the_centre_not_a_corner(orb):
    # THE off-centre bug, finally: a view-BACKED layer has anchorPoint (0,0),
    # so scaling it pushed the orb up and to the right by (scale-1)*box/2 —
    # 25px at the old 1.33 dock scale, exactly the drift Kai kept seeing.
    # Everything therefore lives in a container sublayer, which defaults to a
    # centred anchor.
    from AppKit import NSMakeRect, NSView

    backing = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 150, 150))
    backing.setWantsLayer_(True)
    assert tuple(backing.layer().anchorPoint()) == (0.0, 0.0), (
        "if AppKit ever centres view-backed layers, this workaround can go")

    assert tuple(orb._group.anchorPoint()) == (0.5, 0.5)
    for layer in (orb._bars, orb._core, orb._progress):
        assert layer.superlayer() is orb._group, "scaled together, from the centre"


def test_orb_is_centred_on_the_same_axis_as_the_text(window):
    # Kai: "center the circle with the center of the text"
    window.expand()
    orb = window._hit.frame()
    box = window._input_box.frame()
    text = window._transcript.frame()
    orb_cx = orb.origin.x + orb.size.width / 2
    assert orb_cx == pytest.approx(box.origin.x + box.size.width / 2)
    assert orb_cx == pytest.approx(text.origin.x + text.size.width / 2)


def test_loading_arc_sits_between_the_core_and_the_bars(orb):
    # "fit it in the little gap between the waves and the circle itself"
    assert orb._r0 < orb._r0 * 1.15 < orb._bar_in


def test_bars_stay_within_a_narrow_band(orb):
    from src.menubar.orb import BAR_FLOOR

    # "less varying in size" — the ring should read as a living rim, not spikes
    assert BAR_FLOOR > 0.4


def test_idle_waves_are_small(orb):
    from src.menubar.orb import IDLE_REACH

    # "I want the idle design waves to be smaller" — with no audio the spokes
    # reach only a fraction of their full travel
    assert IDLE_REACH < 0.5
    idle = _ring_reach(orb)
    orb.set_state("speaking")
    for _ in range(8):
        orb.set_level(1.0)
    assert _ring_reach(orb) > idle * 1.25
