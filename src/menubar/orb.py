"""
OrbWindow — the assistant as a creature that lives on your desktop.

The front-end is not a chat app you open; it's a presence that sits at the edge
of the screen, breathing. Talk or type to it and it BLOOMS open into a
conversation surface; when you're done it collapses back to the orb. Kai's
direction (2026-07-27): "incredibly interactive, wavy and smooth in motion …
noticeably activated to dormant when it's not."

    dormant    a slow-breathing orb, dim, parked wherever you left it
    listening  blue, ripples driven by your ACTUAL mic amplitude
    thinking   accent, fast ripples — the model is generating
    speaking   green, steady swell

CONVERSATION MODEL (Kai's pick): ephemeral + history on demand. Only the current
exchange shows; the ⌄ button expands full scrollback. The transcript is a real
selectable NSTextView, so ⌘A/⌘C still work (see install_edit_menu).

MOTION RULES (learned the hard way — see presence.py and the DEVLOG)
- Every continuous animation is declarative Core Animation, run by the
  WindowServer on the GPU. Python is NEVER in the frame loop: inference holds
  the GIL, so a timer-driven redraw would stutter exactly during `thinking`.
- Python pushes only low-rate DATA (`set_presence_level`, ~10-20Hz from the mic).
  Ripple/core layers are sublayers, so Core Animation implicitly animates each
  change over ~0.25s — coarse samples render fluidly for free.
- A collapsed or hidden orb still breathes (one GPU animation, no Python), but
  `suspend()` strips everything when the window is ordered out.
- PyObjC maps method underscores to selector colons, so the visual controller is
  a plain Python class over vanilla NSViews — `set_state_` would publish as
  `set:state:` and fail arity checks.

Public API mirrors ReplyPanel exactly, so the app can swap between them
(`features.orb_ui`).
"""
import math

import objc
from AppKit import (
    NSAnimationContext,
    NSBackingStoreBuffered,
    NSButton,
    NSColor,
    NSFont,
    NSFontAttributeName,
    NSLayoutManager,
    NSForegroundColorAttributeName,
    NSParagraphStyleAttributeName,
    NSMakeRect,
    NSMutableParagraphStyle,
    NSPanel,
    NSScreen,
    NSScrollView,
    NSTextAlignmentCenter,
    NSTextField,
    NSTextView,
    NSView,
    NSVisualEffectView,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskFullSizeContentView,
    NSWindowStyleMaskMiniaturizable,
    NSWindowStyleMaskNonactivatingPanel,
    NSWindowStyleMaskTitled,
)
from Foundation import NSMutableAttributedString, NSObject, NSPoint
from Quartz import (
    CAAnimationGroup,
    CALayer,
    CATransaction,
    CATransition,
    CABasicAnimation,
    CACurrentMediaTime,
    CAMediaTimingFunction,
    CAShapeLayer,
    CATransform3DMakeScale,
    CGAffineTransformMakeScale,
    CGPathAddLineToPoint,
    CGPathCreateMutable,
    CGPathCreateWithEllipseInRect,
    CGPathMoveToPoint,
    CGRectMake,
    kCAMediaTimingFunctionEaseInEaseOut,
    kCAMediaTimingFunctionEaseOut,
    kCAMediaTimingFunctionLinear,
)

from src.menubar.panel import _ChatInput, _Submitter, install_edit_menu

# -- geometry ----------------------------------------------------------------
ORB_BOX = 150                   # parked window: orb + room for ripples
OPEN_W, OPEN_H = 460, 460       # bloomed conversation surface
ORB_SMALL = 176                 # docked a little larger than parked
MARGIN = 18
TRAFFIC_CLEARANCE = 76   # standard buttons sit at x=7/27/47
INPUT_H = 32
TEXT_BOTTOM = 76         # the message is anchored HERE and grows upward
TEXT_MAX_H = 160         # …no further: beyond this it clips instead of
                         # spilling over the ask box (the briefing did exactly
                         # that — long text ignored the frame entirely)
TEXT_TOP = TEXT_BOTTOM + TEXT_MAX_H
SCROLL_STEP = 8.0        # wheel delta that advances one message
LEVEL_SMOOTHING = 0.10   # seconds — short enough that single words still show
SWAP_S = 0.40           # message swap: fade + a small rise into place
# Asymmetric envelope, the way an audio meter behaves: rise quickly so a word's
# onset lands immediately, fall gently so the orb glides down instead of
# flickering between syllables. Raising the animation duration alone would have
# smoothed it by adding lag; this smooths the SIGNAL and keeps the attack.
LEVEL_ATTACK = 0.42      # fraction of the gap closed when getting louder
LEVEL_RELEASE = 0.12     # …and when getting quieter
BLOOM_S = 0.42                  # bloom / collapse duration
EDGE_GAP = 24                   # default parking distance from the screen edge

BARS = 56                # spokes in the ring (Bensound-style visualiser)
BAR_FLOOR = 0.52         # bars stay within a narrow band — the ring reads as
                         # one living rim rather than a spiky graph
BAND_ATTACK = 0.62       # bands move faster than the overall level: the ring
BAND_RELEASE = 0.34      # should shimmer actively, not heave
IDLE_REACH = 0.34        # how far the spokes reach with NO audio — idle waves
                         # stay small, and speech is what extends them
STATE_FADE_S = 0.45      # core easing between its fixed per-state sizes
PROGRESS_SPAN = 0.22     # how much of the loading ring the sweeping bar covers
PROGRESS_PERIOD = 1.15   # seconds per lap
SWEEP_FADE_S = 0.45      # loading arc dissolves into the live orb
DOCK_SCALE = ORB_SMALL / ORB_BOX  # how far the orb shrinks when docked

# per-state: (core scale, ring spin period s, bar opacity, color fn)
_STATES = {
    # the model takes ~50s to load; the sweeping arc says "working" honestly —
    # we have no real progress to report, so nothing pretends to measure it
    "loading":   (0.92, 40.0, 0.10, NSColor.secondaryLabelColor),
    "dormant":   (1.00, 34.0, 0.30, NSColor.controlAccentColor),
    "listening": (1.18, 10.0, 0.90, NSColor.systemBlueColor),
    "thinking":  (1.10,  6.0, 0.85, NSColor.controlAccentColor),
    "speaking":  (1.14,  9.0, 0.90, NSColor.systemGreenColor),
}





class OrbView:
    """The creature: a solid core ringed by a radial bar visualiser.

    The bars are the Bensound-style circular spectrum Kai asked for — spokes
    all the way around, each rising and falling with a frequency band, so
    speech makes the ring ripple rather than merely pulse.

    Main thread only. `.view` is the NSView to place in a superview.
    """

    def __init__(self, box: float):
        self.view = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, box, box))
        self.view.setWantsLayer_(True)
        # A view-BACKED layer has anchorPoint (0,0) — the bottom-left corner —
        # so scaling it grew the orb up and to the right instead of outward from
        # its middle, which is exactly why it kept landing off-centre (25px at
        # DOCK_SCALE 1.33). Everything lives in this container sublayer instead:
        # layers we create ourselves default to a centred anchorPoint.
        self._group = CALayer.layer()
        self._group.setFrame_(NSMakeRect(0, 0, box, box))
        self.view.layer().addSublayer_(self._group)
        self._box = box
        self._state = "dormant"
        self._level = 0.0
        self._bands = None          # smoothed per-bar heights, 0..1
        self._suspended = False
        self._scale = 1.0

        c = box / 2.0
        self._r0 = box * 0.22                   # core radius
        self._bar_in = self._r0 * 1.34          # leaves a gap for the loading arc
        self._bar_len = box * 0.27              # …and reach this much further
        unit = CGRectMake(c - self._r0, c - self._r0, self._r0 * 2, self._r0 * 2)

        # a deterministic, organic-looking resting profile: when there's no
        # audio the ring still has shape, and the slow spin animates it without
        # Python ever touching a frame
        self._idle = [
            0.42 + 0.13 * math.sin(i * 0.7) + 0.08 * math.sin(i * 0.31 + 1.3)
            for i in range(BARS)
        ]

        # ONE layer holds every bar: rebuilding a 56-segment path costs a single
        # property set, and Core Animation interpolates between two paths with
        # matching structure — so the ring morphs smoothly between audio frames
        # instead of snapping.
        self._bars = CAShapeLayer.layer()
        self._bars.setFrame_(self.view.bounds())
        self._bars.setFillColor_(None)
        self._bars.setLineWidth_(max(1.6, box * 0.018))
        self._bars.setLineCap_("round")
        self._bars.setPath_(self._bar_path(self._idle))
        self._group.addSublayer_(self._bars)

        # the loading sweep: a bar travelling around the core while the model
        # loads. Hidden in every other state.
        r_prog = self._r0 * 1.15   # in the gap between core and bars
        self._progress = CAShapeLayer.layer()
        self._progress.setFrame_(self.view.bounds())
        self._progress.setPath_(CGPathCreateWithEllipseInRect(
            CGRectMake(c - r_prog, c - r_prog, r_prog * 2, r_prog * 2), None))
        self._progress.setFillColor_(None)
        self._progress.setLineWidth_(max(2.0, box * 0.020))
        self._progress.setLineCap_("round")
        self._progress.setStrokeStart_(0.0)
        self._progress.setStrokeEnd_(PROGRESS_SPAN)
        self._progress.setHidden_(True)
        self._group.addSublayer_(self._progress)

        self._core = CAShapeLayer.layer()
        self._core.setFrame_(self.view.bounds())
        self._core.setPath_(CGPathCreateWithEllipseInRect(unit, None))
        self._group.addSublayer_(self._core)

        self._apply_state()

    # -- geometry ----------------------------------------------------------

    def _bar_path(self, heights):
        """One stroked spoke per band, radiating from just outside the core.

        Two things set a spoke's length: the band's share of the spectrum
        (shape) and the overall loudness (reach). The FFT is normalised per
        frame, so shape alone says nothing about volume — without the reach
        term the ring would change pattern while staying the same size.
        """
        path = CGPathCreateMutable()
        c = self._box / 2.0
        reach = IDLE_REACH + (1.0 - IDLE_REACH) * self._level
        for i, h in enumerate(heights):
            angle = 2.0 * math.pi * i / BARS
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            inner = self._bar_in
            extent = (BAR_FLOOR + (1 - BAR_FLOOR) * h) * reach
            outer = inner + self._bar_len * extent
            CGPathMoveToPoint(path, None, c + inner * cos_a, c + inner * sin_a)
            CGPathAddLineToPoint(path, None, c + outer * cos_a, c + outer * sin_a)
        return path

    # -- public ------------------------------------------------------------

    def set_state(self, state: str) -> None:
        if state not in _STATES or state == self._state:
            return
        self._state = state
        if state != "listening":
            self._level = 0.0
            self._bands = None
        self._apply_state()

    def set_level(self, level: float, bands=None) -> None:
        """Scalar loudness plus (optionally) per-band heights from the FFT.

        Both are envelope-followed — fast attack so a word's onset lands at
        once, slow release so the ring glides instead of chattering."""
        try:
            target = max(0.0, min(1.0, float(level)))
        except (TypeError, ValueError):
            return
        rate = LEVEL_ATTACK if target > self._level else LEVEL_RELEASE
        self._level += (target - self._level) * rate
        if self._level < 0.001:
            self._level = 0.0
        self._smooth_bands(bands)
        if not self._suspended:
            self._apply_level()

    def _smooth_bands(self, bands) -> None:
        if not bands:
            # no spectrum (silence, or a caller that only has a level): decay
            # the ring back toward its resting profile rather than freezing
            if self._bands is not None:
                self._bands = [b + (idle - b) * BAND_RELEASE
                               for b, idle in zip(self._bands, self._idle)]
            return
        values = list(bands)[:BARS]
        if len(values) < BARS:  # pad rather than reshape the path structure
            values += self._idle[len(values):]
        if self._bands is None:
            self._bands = list(self._idle)
        self._bands = [
            b + (v - b) * (BAND_ATTACK if v > b else BAND_RELEASE)
            for b, v in zip(self._bands, values)
        ]

    def suspend(self) -> None:
        self._suspended = True
        for layer in (self._bars, self._core, self._progress):
            layer.removeAllAnimations()

    def resume(self) -> None:
        if not self._suspended:
            return
        self._suspended = False
        self._apply_state()

    def set_scale(self, scale: float, duration: float = 0.0) -> None:
        """Shrink/grow the WHOLE orb as one unit — core, bars and glow together.
        Geometry is fixed at full size and scaled by the view's layer, so
        docking is one smooth transform rather than a re-layout that snaps."""
        layer = self._group
        if duration > 0:
            anim = CABasicAnimation.animationWithKeyPath_("transform.scale")
            anim.setFromValue_(self._scale)
            anim.setToValue_(scale)
            anim.setDuration_(duration)
            anim.setTimingFunction_(
                CAMediaTimingFunction.functionWithName_(kCAMediaTimingFunctionEaseInEaseOut))
            layer.addAnimation_forKey_(anim, "dock")
        layer.setTransform_(CATransform3DMakeScale(scale, scale, 1.0))
        self._scale = scale

    # -- internals ---------------------------------------------------------

    def _apply_state(self) -> None:
        if self._suspended:
            return
        _scale, spin, opacity, color_fn = _STATES[self._state]
        cg = color_fn().CGColor()
        self._core.setFillColor_(cg)
        self._core.setShadowColor_(cg)
        self._core.setShadowOffset_((0, 0))  # opacity/radius track the level
        self._bars.setStrokeColor_(cg)
        self._bars.setOpacity_(opacity)

        self._progress.setStrokeColor_(NSColor.controlAccentColor().CGColor())
        self._sweep(self._state == "loading")
        self._spin(spin)
        self._apply_core()
        self._apply_level()
        self._breathe(self._state == "dormant")

    def _sweep(self, enabled: bool) -> None:
        """The loading tell: a bar lapping the core until the model is ready."""
        self._progress.removeAnimationForKey_("sweep")
        if not enabled:
            if not self._progress.isHidden():
                # fade out rather than vanish, so loading hands over to the
                # live orb instead of blinking
                CATransaction.begin()
                CATransaction.setAnimationDuration_(SWEEP_FADE_S)
                CATransaction.setCompletionBlock_(
                    lambda: self._progress.setHidden_(True))
                self._progress.setOpacity_(0.0)
                CATransaction.commit()
            return
        self._progress.setOpacity_(1.0)
        self._progress.setHidden_(False)
        anim = CABasicAnimation.animationWithKeyPath_("transform.rotation.z")
        anim.setFromValue_(0.0)
        anim.setToValue_(2.0 * math.pi)
        anim.setDuration_(PROGRESS_PERIOD)
        anim.setRepeatCount_(float("inf"))
        anim.setTimingFunction_(
            CAMediaTimingFunction.functionWithName_(kCAMediaTimingFunctionLinear))
        self._progress.addAnimation_forKey_(anim, "sweep")

    def _spin(self, period: float) -> None:
        """The ring turns slowly and forever — motion that costs no Python."""
        self._bars.removeAnimationForKey_("spin")
        anim = CABasicAnimation.animationWithKeyPath_("transform.rotation.z")
        anim.setFromValue_(0.0)
        anim.setToValue_(2.0 * math.pi)
        anim.setDuration_(period)
        anim.setRepeatCount_(float("inf"))
        anim.setTimingFunction_(
            CAMediaTimingFunction.functionWithName_(kCAMediaTimingFunctionLinear))
        self._bars.addAnimation_forKey_(anim, "spin")

    def _apply_core(self) -> None:
        """The core settles at ONE size per state and stays there.

        It used to swell with every audio frame, which made the whole circle
        breathe while you talked. Kai wants the disc to reach its size and stop;
        the ring is what should answer the voice."""
        base, _spin, _opacity, _color = _STATES[self._state]
        CATransaction.begin()
        CATransaction.setAnimationDuration_(STATE_FADE_S)
        self._core.setAffineTransform_(CGAffineTransformMakeScale(base, base))
        self._core.setShadowOpacity_(0.35 if self._state == "dormant" else 0.7)
        self._core.setShadowRadius_(self._box * 0.16)
        CATransaction.commit()

    def _apply_level(self) -> None:
        """Only the ring moves with the audio now."""
        CATransaction.begin()
        CATransaction.setAnimationDuration_(LEVEL_SMOOTHING)
        # LINEAR: the default eased curve makes every ~45ms audio block ease in
        # and out, and chaining those reads as a pumping stutter. Linear
        # segments join seamlessly into one continuous motion.
        CATransaction.setAnimationTimingFunction_(
            CAMediaTimingFunction.functionWithName_(kCAMediaTimingFunctionLinear))
        self._bars.setPath_(self._bar_path(self._bands or self._idle))
        CATransaction.commit()

    def _breathe(self, enabled: bool) -> None:
        """The dormant tell — it has to MOVE, not just sit there dimmed."""
        self._core.removeAnimationForKey_("breathe")
        if not enabled:
            return
        base = _STATES[self._state][0]
        swell = CABasicAnimation.animationWithKeyPath_("transform.scale")
        swell.setFromValue_(base * 0.88)
        swell.setToValue_(base * 1.12)
        glow = CABasicAnimation.animationWithKeyPath_("opacity")
        glow.setFromValue_(0.35)
        glow.setToValue_(0.8)
        group = CAAnimationGroup.animation()
        group.setAnimations_([swell, glow])
        group.setDuration_(3.6)
        group.setAutoreverses_(True)
        group.setRepeatCount_(float("inf"))
        group.setTimingFunction_(
            CAMediaTimingFunction.functionWithName_(kCAMediaTimingFunctionEaseInEaseOut))
        self._core.addAnimation_forKey_(group, "breathe")


class _OrbDelegate(NSObject):
    """Red button = fold back to the orb. The orb is the app's presence; closing
    the conversation must not take it off the desktop."""

    def initWithCollapse_relayout_(self, collapse, relayout):
        self = objc.super(_OrbDelegate, self).init()
        if self is None:
            return None
        self._collapse = collapse
        self._relayout = relayout
        return self

    def windowShouldClose_(self, sender):
        self._collapse()
        return False

    def windowDidResize_(self, notification):
        self._relayout()

    def windowDidDeminiaturize_(self, notification):
        # restoring from the Dock can hand back a different content size than
        # we laid out for; re-place everything rather than trusting the frame
        self._relayout()


class _OrbPanel(NSPanel):
    """Borderless panels can't become key by default — without this the input
    would never accept a keystroke. Esc collapses."""

    def canBecomeKeyWindow(self):
        return True

    def cancelOperation_(self, sender):
        cb = getattr(self, "_on_cancel", None)
        if cb is not None:
            cb()


class _FadeText(NSTextView):
    """The message lives in ONE fixed place — scrolling doesn't move the text,
    it swaps which message occupies the slot (crossfaded). Scroll up for older,
    down toward the newest. Kai's ask: 'static text … when scrolling it fades
    and shows the older messages'."""

    def initWithFrame_onStep_(self, frame, on_step):
        self = objc.super(_FadeText, self).initWithFrame_(frame)
        if self is None:
            return None
        self._on_step = on_step
        self._accum = 0.0
        return self

    def scrollWheel_(self, event):
        delta = event.scrollingDeltaY()
        if event.hasPreciseScrollingDeltas():
            delta /= 3.0  # trackpads report far finer deltas than a wheel
        if event.isDirectionInvertedFromDevice():
            delta = -delta
        self._accum += delta
        while self._accum >= SCROLL_STEP:
            self._accum -= SCROLL_STEP
            self._on_step(1)    # scrolling up reaches back in time
        while self._accum <= -SCROLL_STEP:
            self._accum += SCROLL_STEP
            self._on_step(-1)


class _OrbHitView(NSView):
    """Click to bloom, drag to reparks the orb. Implemented manually rather than
    movableByWindowBackground so a click and a drag stay distinguishable."""

    def initWithFrame_onClick_(self, frame, on_click):
        self = objc.super(_OrbHitView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._on_click = on_click
        self._down = None
        self._moved = 0.0
        self._radius = frame.size.width / 2.0
        return self

    def setHitRadius_(self, radius):
        self._radius = float(radius)

    def hitTest_(self, point):
        """Only the DRAWN orb is clickable. The view keeps its full-size frame
        so the layer can scale smoothly, so without this the docked orb would
        still swallow clicks across a 92px box — including the transcript."""
        center = self.convertPoint_toView_(
            NSPoint(self.bounds().size.width / 2, self.bounds().size.height / 2),
            self.superview())
        dx = point.x - center.x
        dy = point.y - center.y
        if (dx * dx + dy * dy) ** 0.5 > self._radius:
            return None
        return objc.super(_OrbHitView, self).hitTest_(point)

    def mouseDown_(self, event):
        self._down = event.locationInWindow()
        self._moved = 0.0

    def mouseDragged_(self, event):
        if self._down is None:
            return
        win = self.window()
        where = event.locationInWindow()
        dx = where.x - self._down.x
        dy = where.y - self._down.y
        self._moved += abs(dx) + abs(dy)
        frame = win.frame()
        win.setFrameOrigin_(NSPoint(frame.origin.x + dx, frame.origin.y + dy))

    def mouseUp_(self, event):
        was_click = self._down is not None and self._moved < 4.0
        self._down = None
        if was_click and self._on_click is not None:
            self._on_click()


class OrbWindow:
    """Drop-in replacement for ReplyPanel. Main thread only."""

    def __init__(self, on_followup, actions: dict | None = None,
                 on_visibility=None):
        self._on_followup = on_followup
        self._on_visibility = on_visibility
        self._actions = dict(actions or {})
        self._exchanges: list[tuple[str, str]] = []
        self._pending_question = None
        self._index = 0          # 0 = newest; scrolling up walks backwards
        self._open = False

        # TITLED (not borderless): Kai wants the real stoplight buttons, and
        # miniaturize only exists on a titled window. The titlebar is
        # transparent with FullSizeContentView, so the surface still looks
        # frameless — and the buttons are HIDDEN while collapsed, where a naked
        # orb should have no chrome at all.
        mask = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskNonactivatingPanel
                | NSWindowStyleMaskFullSizeContentView)
        self.panel = _OrbPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, ORB_BOX, ORB_BOX), mask, NSBackingStoreBuffered, False)
        self.panel.setTitlebarAppearsTransparent_(True)
        self.panel.setTitleVisibility_(1)  # hidden — the header carries the name
        self.panel.setOpaque_(False)
        self.panel.setBackgroundColor_(NSColor.clearColor())
        self.panel.setLevel_(3)  # NSFloatingWindowLevel
        self.panel.setHidesOnDeactivate_(False)
        self.panel.setReleasedWhenClosed_(False)
        self.panel.setBecomesKeyOnlyIfNeeded_(False)
        self.panel.setMovableByWindowBackground_(False)
        self.panel._on_cancel = self.collapse
        self.panel.setHasShadow_(False)  # the orb's glow is its shadow
        self._delegate = _OrbDelegate.alloc().initWithCollapse_relayout_(
            self.collapse, self.relayout)
        self.panel.setDelegate_(self._delegate)
        self._chrome_buttons_visible(False)

        content = self.panel.contentView()
        content.setWantsLayer_(True)

        # -- the conversation chrome (hidden while collapsed) ---------------
        self._chrome = NSVisualEffectView.alloc().initWithFrame_(
            NSMakeRect(0, 0, OPEN_W, OPEN_H))
        self._chrome.setWantsLayer_(True)
        self._chrome.layer().setCornerRadius_(20)
        self._chrome.layer().setMasksToBounds_(True)
        self._chrome.setBlendingMode_(0)      # behind-window: real translucency
        self._chrome.setMaterial_(15)         # NSVisualEffectMaterialHUDWindow
        self._chrome.setState_(1)             # active
        self._chrome.setAlphaValue_(0.0)
        content.addSubview_(self._chrome)

        self._submitter = _Submitter.alloc().initWithCallback_(self._submit)
        self._submitter._actions = {}

        # the message occupies a FIXED slot — no scrolling text, no bubbles.
        # Scrolling pages through history in place (see _FadeText / _step).
        text_x = MARGIN + 16
        self._transcript = _FadeText.alloc().initWithFrame_onStep_(
            NSMakeRect(text_x, TEXT_BOTTOM, OPEN_W - 2 * text_x,
                       TEXT_TOP - TEXT_BOTTOM),
            self._step)
        self._transcript.setEditable_(False)
        self._transcript.setSelectable_(True)
        self._transcript.setDrawsBackground_(False)
        self._transcript.setWantsLayer_(True)  # crossfades happen on its layer
        self._transcript.layer().setMasksToBounds_(True)  # never spill past the slot
        self._transcript.setTextContainerInset_((0, 0))
        self._transcript.setVerticallyResizable_(False)
        self._transcript.setHorizontallyResizable_(False)
        self._transcript.textContainer().setWidthTracksTextView_(True)
        self._chrome.addSubview_(self._transcript)

        # only visible once you've scrolled back — tells you where you are
        self._marker = NSTextField.labelWithString_("")
        self._marker.setFont_(NSFont.systemFontOfSize_(10.5))
        self._marker.setTextColor_(NSColor.tertiaryLabelColor())
        self._marker.setAlignment_(NSTextAlignmentCenter)
        self._marker.setFrame_(NSMakeRect(text_x, TEXT_BOTTOM - 22,
                                          OPEN_W - 2 * text_x, 14))
        self._chrome.addSubview_(self._marker)

        # the input fills the pill and centres its text both ways — the old
        # frame was inset by an eyeballed 6/8px and read as off-centre
        self._input = _ChatInput.alloc().initWithFrame_submit_placeholder_onChange_(
            NSMakeRect(MARGIN, MARGIN, OPEN_W - 2 * MARGIN, INPUT_H),
            self._submit, "Ask anything…", None)
        self._input.setFont_(NSFont.systemFontOfSize_(13))
        self._input.setDrawsBackground_(False)
        self._input.setAlignment_(NSTextAlignmentCenter)  # like everything else here
        # measured, not eyeballed: a hardcoded line height left the caret and
        # text sitting slightly high in the pill
        _line_h = NSLayoutManager.alloc().init().defaultLineHeightForFont_(
            self._input.font())
        self._input.setTextContainerInset_((14, max(0.0, (INPUT_H - _line_h) / 2)))
        self._input._center_placeholder = True
        box = NSView.alloc().initWithFrame_(
            NSMakeRect(MARGIN, MARGIN, OPEN_W - 2 * MARGIN, INPUT_H))
        box.setWantsLayer_(True)
        box.layer().setCornerRadius_(INPUT_H / 2)
        box.layer().setBorderWidth_(1)
        box.layer().setBorderColor_(NSColor.separatorColor().CGColor())
        self._chrome.addSubview_(box)
        self._input_box = box
        self._chrome.addSubview_(self._input)

        # -- the orb itself, on top of everything ---------------------------
        self._orb = OrbView(ORB_BOX)
        self._hit = _OrbHitView.alloc().initWithFrame_onClick_(
            NSMakeRect(0, 0, ORB_BOX, ORB_BOX), self._orb_clicked)
        self._hit.addSubview_(self._orb.view)
        content.addSubview_(self._hit)

        self._park_default()
        self._render()

    def relayout(self) -> None:
        """Re-place everything for the window's CURRENT size (resize, restore)."""
        bounds = self.panel.contentView().bounds()
        self._place(bounds.size.width, bounds.size.height)

    def _place(self, width: float, height: float) -> None:
        """Lay the surface out for an EXPLICIT size.

        Callers that are starting a resize animation pass the TARGET size
        rather than the live bounds: mid-animation the window still reports its
        old size, so laying out from `contentView().bounds()` at that moment
        pinned the orb to the collapsed geometry and left it off-centre once the
        window finished growing (Kai's before/after, twice). Resize and restore
        notifications still call relayout() with the live size, which corrects
        any path that doesn't go through an animation."""
        if width < 2 or height < 2:
            return
        if not self._open:
            # parked: dead centre of the little window
            self._hit.setFrame_(NSMakeRect(width / 2 - ORB_BOX / 2,
                                           height / 2 - ORB_BOX / 2,
                                           ORB_BOX, ORB_BOX))
            return

        self._chrome.setFrame_(NSMakeRect(0, 0, width, height))
        self._hit.setFrame_(NSMakeRect(width / 2 - ORB_BOX / 2,
                                       height - MARGIN - ORB_SMALL / 2 - ORB_BOX / 2,
                                       ORB_BOX, ORB_BOX))
        text_x = MARGIN + 16
        text_w = max(40.0, width - 2 * text_x)
        self._transcript.setFrame_(NSMakeRect(
            text_x, TEXT_BOTTOM, text_w, self._transcript.frame().size.height))
        self._marker.setFrame_(NSMakeRect(text_x, TEXT_BOTTOM - 22, text_w, 14))
        input_w = max(40.0, width - 2 * MARGIN)
        self._input_box.setFrame_(NSMakeRect(MARGIN, MARGIN, input_w, INPUT_H))
        self._input.setFrame_(NSMakeRect(MARGIN, MARGIN, input_w, INPUT_H))
        self._layout_text()

    def _chrome_buttons_visible(self, visible: bool) -> None:
        """Stoplights belong to the conversation surface, not the bare orb."""
        for index in (0, 1, 2):  # close, miniaturize, zoom
            button = self.panel.standardWindowButton_(index)
            if button is not None:
                button.setHidden_(not visible)

    def _chrome_button(self, glyph: str, ident: str, x: float):
        btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(x, OPEN_H - MARGIN - ORB_SMALL + 4, 26, 24))
        btn.setTitle_(glyph)
        btn.setIdentifier_(ident)
        btn.setBezelStyle_(1)
        btn.setBordered_(False)
        btn.setFont_(NSFont.systemFontOfSize_(13))
        btn.setTarget_(self._submitter)
        btn.setAction_("header:")
        self._chrome.addSubview_(btn)
        return btn

    # -- placement ---------------------------------------------------------

    def _park_default(self) -> None:
        screen = NSScreen.mainScreen().visibleFrame()
        x = screen.origin.x + screen.size.width - ORB_BOX - EDGE_GAP
        y = screen.origin.y + EDGE_GAP
        self.panel.setFrame_display_(NSMakeRect(x, y, ORB_BOX, ORB_BOX), False)

    def _clamp(self, x: float, y: float, w: float, h: float):
        screen = NSScreen.mainScreen().visibleFrame()
        x = max(screen.origin.x + 8,
                min(x, screen.origin.x + screen.size.width - w - 8))
        y = max(screen.origin.y + 8,
                min(y, screen.origin.y + screen.size.height - h - 8))
        return NSMakeRect(x, y, w, h)

    def _orb_center(self):
        f = self.panel.frame()
        if self._open:
            return (f.origin.x + OPEN_W / 2,
                    f.origin.y + OPEN_H - MARGIN - ORB_SMALL / 2)
        return (f.origin.x + ORB_BOX / 2, f.origin.y + ORB_BOX / 2)

    # -- bloom / collapse --------------------------------------------------

    def _orb_clicked(self) -> None:
        self.collapse() if self._open else self.expand()

    def expand(self) -> None:
        """Bloom open. The orb keeps its place on screen as far as the screen
        edges allow, and the surface unfurls around it."""
        self._present_window()
        if self._open:
            return
        cx, cy = self._orb_center()  # measure BEFORE flipping the state flag
        self._open = True
        self._chrome_buttons_visible(True)
        if self._on_visibility is not None:
            self._on_visibility(True)
        target = self._clamp(cx - OPEN_W / 2,
                             cy - OPEN_H + MARGIN + ORB_SMALL / 2, OPEN_W, OPEN_H)
        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(BLOOM_S)
        NSAnimationContext.currentContext().setCompletionHandler_(self.relayout)
        self.panel.animator().setFrame_display_(target, True)
        self._chrome.animator().setAlphaValue_(1.0)
        NSAnimationContext.endGrouping()
        self._place(OPEN_W, OPEN_H)  # the size we're animating TO, not from
        self._orb.set_scale(DOCK_SCALE, BLOOM_S)
        self._hit.setHitRadius_(ORB_BOX / 2 * DOCK_SCALE)
        self._render()
        self.panel.makeFirstResponder_(self._input)

    def collapse(self) -> None:
        """Fold back to the orb, leaving it where the surface's orb sat."""
        if not self._open:
            return
        cx, cy = self._orb_center()  # measure BEFORE flipping the state flag
        self._open = False
        self._chrome_buttons_visible(False)
        if self._on_visibility is not None:
            self._on_visibility(False)
        target = self._clamp(cx - ORB_BOX / 2, cy - ORB_BOX / 2, ORB_BOX, ORB_BOX)

        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(BLOOM_S)
        NSAnimationContext.currentContext().setCompletionHandler_(self.relayout)
        self._chrome.animator().setAlphaValue_(0.0)
        self.panel.animator().setFrame_display_(target, True)
        NSAnimationContext.endGrouping()
        self._place(ORB_BOX, ORB_BOX)
        self._orb.set_scale(1.0, BLOOM_S)
        self._hit.setHitRadius_(ORB_BOX / 2)

    # -- transcript --------------------------------------------------------

    def _layout_text(self) -> None:
        """Anchor the message at the bottom and let it grow UP, capped.

        NSTextView doesn't clip to its frame, so a long reply (the morning
        briefing, a file list) drew straight over the ask box. Height is now
        measured and pinned to TEXT_MAX_H, with the layer masking anything past
        it — the bottom edge stays put either way, which is what makes the slot
        feel static."""
        view = self._transcript
        layout = view.layoutManager()
        container = view.textContainer()
        layout.ensureLayoutForTextContainer_(container)
        used = layout.usedRectForTextContainer_(container).size.height
        height = max(1.0, min(used, TEXT_MAX_H))
        frame = view.frame()
        view.setFrame_(NSMakeRect(frame.origin.x, TEXT_BOTTOM,
                                  frame.size.width, height))

    def _set_body(self, body) -> None:
        self._transcript.textStorage().setAttributedString_(body)
        self._layout_text()

    def _fade_swap(self, body) -> None:
        """Swap the message, then let the new one rise gently into place.

        The content is set FIRST and the animation decorates it — an earlier
        version faded out, swapped in a completion handler, then faded in, which
        made the displayed text depend on an animation callback actually firing.
        Here the slot is always correct even if the animation is dropped."""
        self._set_body(body)
        fade = CABasicAnimation.animationWithKeyPath_("opacity")
        fade.setFromValue_(0.0)
        fade.setToValue_(1.0)
        rise = CABasicAnimation.animationWithKeyPath_("transform.translation.y")
        rise.setFromValue_(-10.0)
        rise.setToValue_(0.0)
        group = CAAnimationGroup.animation()
        group.setAnimations_([fade, rise])
        group.setDuration_(SWAP_S)
        group.setTimingFunction_(
            CAMediaTimingFunction.functionWithName_(kCAMediaTimingFunctionEaseOut))
        self._transcript.layer().addAnimation_forKey_(group, "swap")

    def _step(self, direction: int) -> None:
        """Page one message back (+1) or forward (-1) through history. The text
        never moves — only what fills the slot, crossfaded."""
        history = self._visible_history()
        if not history:
            return
        target = max(0, min(self._index + direction, len(history) - 1))
        if target == self._index:
            return
        self._index = target
        self._render(fade=True)

    def _visible_history(self):
        """Newest first — index 0 is always the most recent thing said."""
        history = list(reversed(self._exchanges))
        if self._pending_question is not None:
            history.insert(0, (self._pending_question, "…"))
        return history

    def _render(self, fade: bool = False) -> None:
        """One exchange in a fixed slot. Scrolling swaps it; nothing scrolls."""
        history = self._visible_history()
        self._index = min(self._index, max(0, len(history) - 1))

        body = NSMutableAttributedString.alloc().init()
        if history:
            question, reply = history[self._index]
            centered = NSMutableParagraphStyle.alloc().init()
            centered.setAlignment_(NSTextAlignmentCenter)
            centered.setLineSpacing_(2.0)
            q_attrs = {NSFontAttributeName: NSFont.systemFontOfSize_(11.5),
                       NSForegroundColorAttributeName: NSColor.secondaryLabelColor(),
                       NSParagraphStyleAttributeName: centered}
            a_attrs = {NSFontAttributeName: NSFont.systemFontOfSize_(15),
                       NSForegroundColorAttributeName: NSColor.labelColor(),
                       NSParagraphStyleAttributeName: centered}
            body.appendAttributedString_(
                NSMutableAttributedString.alloc()
                .initWithString_attributes_(f"{question}\n\n", q_attrs))
            body.appendAttributedString_(
                NSMutableAttributedString.alloc()
                .initWithString_attributes_(reply, a_attrs))

        if fade:
            # A CATransition snapshots the layer and cross-dissolves the two,
            # which smears while the frame is also being resized for the new
            # text. Fading the view OUT, swapping, then fading IN is a clean
            # two-step: nothing is ever half-drawn mid-resize.
            self._fade_swap(body)
        else:
            self._set_body(body)

        depth = len(history) - 1
        if self._index and depth:
            self._marker.setStringValue_(
                f"{self._index} back · scroll down for latest")
        else:
            self._marker.setStringValue_("")

    # -- ReplyPanel-compatible API ----------------------------------------

    def _present_window(self) -> None:
        self.panel.orderFrontRegardless()
        self._orb.resume()

    def park(self) -> None:
        """Put the orb on the desktop, dormant and collapsed — how the app
        idles. Deliberately does NOT report itself as 'visible': the menu's
        Show Chat stays live, because expanding is what it does."""
        self._present_window()

    def present(self) -> None:
        self.expand()

    def show(self, question: str, reply: str) -> None:
        self._pending_question = None
        self._index = 0  # a new answer always takes the slot
        self._exchanges.append((question, reply))
        self._input.setString_("")
        self._input.setNeedsDisplay_(True)
        self.expand()
        self._render(fade=True)  # a reply should arrive, not blink into place

    def show_sections(self, question: str, sections: list[str]) -> None:
        self.show(question, "\n\n".join(sections))

    def show_thinking(self, question: str) -> None:
        self._pending_question = question
        self._index = 0
        self.expand()
        self._render()

    def set_action_state(self, name: str, active: bool,
                         active_title: str | None = None) -> None:
        """No header buttons here — the ORB is the state display. Speak mode
        simply lights it up (the menu bar keeps the actual controls)."""
        if name == "Speak":
            self.set_presence("listening" if active else "dormant")

    def set_presence(self, state: str) -> None:
        self._orb.set_state(state)

    def set_presence_level(self, level: float, bands=None) -> None:
        self._orb.set_level(level, bands)

    def clear(self) -> None:
        self._exchanges.clear()
        self._pending_question = None
        self._index = 0
        self._render()

    def close(self) -> None:
        """Fold away. The orb stays parked on the desktop (it's a presence, not
        a window) — only its conversation surface goes."""
        self.collapse()

    # -- input -------------------------------------------------------------

    def _submit(self) -> None:
        text = str(self._input.string()).strip()
        if not text:
            return
        self._input.setString_("")
        self._on_followup(text)


def install_orb_edit_menu() -> None:
    """⌘A/⌘C in the transcript — same reason as the panel (see panel.py)."""
    install_edit_menu()
