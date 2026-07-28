"""
Presence — the assistant as a living presence, not a static box.

A compact waveform in the panel header that visibly wakes when you talk or type
to it and settles back to dormant when it's idle. Four states, driven by the
state machine the app already has (READY / LISTENING / THINKING + speaking):

    dormant    dim, nearly flat, slow breathing — asleep but alive
    listening  accent-colored, height driven by your ACTUAL mic amplitude
    thinking   brighter, fast travel — the model is generating
    speaking   green, steady swell — it's talking back

HOW THE MOTION WORKS (and why it stays smooth while the model runs)

Every continuous animation here is declarative Core Animation: we hand the
system one description ("translate this layer by one wavelength, forever") and
the WindowServer animates it on the GPU. Python is not in the frame loop — which
matters because inference holds the GIL on this process, and a Python-driven
redraw would stutter exactly during `thinking`, the moment the motion is most
meaningful.

The travelling wave is the classic scroll trick: the path is built one
wavelength WIDER than the view and translated by exactly one wavelength, so the
loop is seamless, and the view's layer clips the overhang. One animation, no
path rebuilding, no timer.

Amplitude is the only thing Python pushes, via `set_level()` — and because these
are sublayers (not a view's backing layer), Core Animation IMPLICITLY animates
the change over ~0.25s. So coarse 10-20Hz samples from the mic render as fluid
motion for free, instead of visible steps.

Dormant still breathes (one opacity animation, GPU-side, no Python), but a
HIDDEN window suspends everything — `suspend()` strips every animation so a
closed panel costs nothing on battery. The panel calls it on show/close.

NOTE: this is a plain Python controller over a VANILLA NSView, deliberately not
an NSView subclass — PyObjC maps method underscores to selector colons, so
`set_state_` would be published as `set:state:` and fail arity checks. Nothing
here needs custom drawing or event handling, only layers, so a subclass buys
nothing and costs that footgun.
"""
import math

from AppKit import NSColor, NSMakeRect, NSView
from Quartz import (
    CABasicAnimation,
    CAMediaTimingFunction,
    CAShapeLayer,
    CGAffineTransformMakeScale,
    CGPathAddLineToPoint,
    CGPathCreateMutable,
    CGPathMoveToPoint,
    kCAMediaTimingFunctionEaseInEaseOut,
    kCAMediaTimingFunctionLinear,
)

WAVELENGTH = 15.0   # px per wave cycle
STEP = 1.0          # path resolution — 1px reads as smooth at this size
LINE_W = 1.6

# per-state: (base amplitude 0..1, seconds per wavelength, opacity, color fn)
_STATES = {
    "dormant":   (0.10, 6.0, 0.45, NSColor.tertiaryLabelColor),
    "listening": (0.40, 1.1, 1.00, NSColor.systemBlueColor),
    "thinking":  (0.55, 0.7, 1.00, NSColor.controlAccentColor),
    "speaking":  (0.60, 1.4, 1.00, NSColor.systemGreenColor),
}

# the back wave trails slower and dimmer than the front one — two layers at
# different speeds read as fluid depth rather than a flat oscilloscope
_BACK_SPEED = 1.7
_BACK_OPACITY = 0.35


class Presence:
    """Main thread only. `.view` is the NSView to drop into a superview."""

    def __init__(self, width: float, height: float):
        self.view = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, width, height))
        self.view.setWantsLayer_(True)
        self.view.layer().setMasksToBounds_(True)  # clips the over-wide path

        self._span = width + 2 * WAVELENGTH  # path wider than the view
        self._height = height
        self._state = "dormant"
        self._level = 0.0
        self._suspended = False

        # back layer added first so the front wave draws over it
        self._back = self._make_wave(phase=math.pi / 2)
        self._front = self._make_wave(phase=0.0)
        for layer in (self._back, self._front):
            self.view.layer().addSublayer_(layer)

        self._apply_state()

    # -- construction ------------------------------------------------------

    def _make_wave(self, phase: float):
        layer = CAShapeLayer.layer()
        # extends a wavelength past each edge so translation never exposes an end
        layer.setFrame_(NSMakeRect(-WAVELENGTH, 0, self._span, self._height))
        layer.setPath_(self._wave_path(phase))
        layer.setFillColor_(None)
        layer.setLineWidth_(LINE_W)
        layer.setLineCap_("round")
        return layer

    def _wave_path(self, phase: float):
        """A full-amplitude sine across the span. Height is scaled at runtime by
        a transform, so this path is built once and never rebuilt."""
        path = CGPathCreateMutable()
        mid = self._height / 2.0
        peak = (self._height - LINE_W) / 2.0
        x = 0.0
        first = True
        while x <= self._span:
            y = mid + peak * math.sin(2 * math.pi * x / WAVELENGTH + phase)
            if first:
                CGPathMoveToPoint(path, None, x, y)
                first = False
            else:
                CGPathAddLineToPoint(path, None, x, y)
            x += STEP
        return path

    # -- public API --------------------------------------------------------

    def set_state(self, state: str) -> None:
        """dormant | listening | thinking | speaking."""
        if state not in _STATES or state == self._state:
            return
        self._state = state
        if state != "listening":
            self._level = 0.0  # only listening is externally driven
        self._apply_state()

    def set_level(self, level: float) -> None:
        """0..1 intensity — mic amplitude while listening, or token activity.
        Implicitly animated, so coarse samples still look fluid."""
        try:
            self._level = max(0.0, min(1.0, float(level)))
        except (TypeError, ValueError):
            return
        if not self._suspended:
            self._apply_amplitude()

    def suspend(self) -> None:
        """Strip every animation — a hidden window must cost nothing."""
        self._suspended = True
        for layer in (self._back, self._front):
            layer.removeAllAnimations()

    def resume(self) -> None:
        if not self._suspended:
            return
        self._suspended = False
        self._apply_state()

    # -- state application -------------------------------------------------

    def _apply_state(self) -> None:
        if self._suspended:
            return
        _amp, period, opacity, color_fn = _STATES[self._state]
        cg = color_fn().CGColor()
        for layer in (self._back, self._front):
            layer.setStrokeColor_(cg)
        self._front.setOpacity_(opacity)
        self._back.setOpacity_(opacity * _BACK_OPACITY)

        self._travel(self._front, period)
        self._travel(self._back, period * _BACK_SPEED)
        self._apply_amplitude()
        self._breathe(self._state == "dormant", opacity)

    def _travel(self, layer, period: float) -> None:
        """Seamless scroll: translate exactly one wavelength, forever."""
        layer.removeAnimationForKey_("travel")
        anim = CABasicAnimation.animationWithKeyPath_("position.x")
        start = layer.position().x
        anim.setFromValue_(start)
        anim.setToValue_(start - WAVELENGTH)
        anim.setDuration_(period)
        anim.setRepeatCount_(float("inf"))
        anim.setTimingFunction_(
            CAMediaTimingFunction.functionWithName_(kCAMediaTimingFunctionLinear))
        layer.addAnimation_forKey_(anim, "travel")

    def _apply_amplitude(self) -> None:
        base = _STATES[self._state][0]
        # level lifts the wave toward full height; dormant stays nearly flat
        amp = base + (1.0 - base) * self._level
        for layer, scale in ((self._front, amp), (self._back, amp * 0.7)):
            # implicit animation smooths each step (see module docstring)
            layer.setAffineTransform_(CGAffineTransformMakeScale(1.0, scale))

    def _breathe(self, enabled: bool, opacity: float) -> None:
        """The dormant tell: a slow opacity swell so idle still reads alive."""
        self._front.removeAnimationForKey_("breathe")
        if not enabled:
            return
        anim = CABasicAnimation.animationWithKeyPath_("opacity")
        anim.setFromValue_(opacity * 0.55)
        anim.setToValue_(opacity)
        anim.setDuration_(3.2)
        anim.setAutoreverses_(True)
        anim.setRepeatCount_(float("inf"))
        anim.setTimingFunction_(
            CAMediaTimingFunction.functionWithName_(kCAMediaTimingFunctionEaseInEaseOut))
        self._front.addAnimation_forKey_(anim, "breathe")
