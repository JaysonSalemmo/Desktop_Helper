"""
ReplyPanel — the assistant's non-modal chat window.

Chat-bubble layout: your messages on the right (accent color), the
assistant's on the left (gray), conversation history scrolling within the
session. A follow-up field keeps the exchange going without reopening the
menu. Bubble text is selectable (filepaths, track names).

Why NSPanel with NonactivatingPanel: floats above other windows without
stealing app focus (macOS won't activate menu-bar apps anyway), yet clicking
the follow-up field still lets you type. Non-modal — nothing blocks the main
thread; the next hotkey/task works while the panel is up.

All methods must be called on the main thread (AppHelper.callAfter).
"""
import objc
from AppKit import (
    NSAnimationContext,
    NSBackingStoreBuffered,
    NSButton,
    NSColor,
    NSFloatingWindowLevel,
    NSFont,
    NSFontAttributeName,
    NSMakeRect,
    NSPanel,
    NSScrollView,
    NSStringDrawingUsesLineFragmentOrigin,
    NSTextField,
    NSView,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskFullSizeContentView,
    NSWindowStyleMaskMiniaturizable,
    NSWindowStyleMaskNonactivatingPanel,
    NSWindowStyleMaskTitled,
)
from Foundation import NSObject, NSString
from Quartz import CABasicAnimation, CAMediaTimingFunction

ANIM_S = 0.28  # bubble entrance / scroll glide duration

WIDTH, HEIGHT = 500, 380
PAD = 14
HEADER_H = 28
TRAFFIC_LIGHT_CLEARANCE = 96  # well clear of the standard traffic lights
INPUT_H = 26
BUBBLE_MAX_W = 340
BUBBLE_PAD_X = 12
BUBBLE_PAD_Y = 7
BUBBLE_GAP = 8
FONT_SIZE = 13


class _FlippedView(NSView):
    """Document view that grows top-down (AppKit default is bottom-up)."""

    def isFlipped(self):
        return True


class _Submitter(NSObject):
    """ObjC target for the input field / Ask button."""

    def initWithCallback_(self, callback):
        self = objc.super(_Submitter, self).init()
        if self is None:
            return None
        self._callback = callback
        self._actions = {}
        return self

    def submit_(self, sender):
        self._callback()

    def header_(self, sender):
        action = self._actions.get(str(sender.identifier()))
        if action is not None:
            action()


class ReplyPanel:
    def __init__(self, on_followup, actions: dict | None = None):
        """on_followup(text) is called on the main thread when submitted.
        actions: {"Speak": fn, "Briefing": fn, ...} → header-bar buttons."""
        self._on_followup = on_followup

        # no UtilityWindow mask: utility panels get miniature, oddly-placed
        # traffic lights — standard chrome gives full-size, centered ones
        mask = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskNonactivatingPanel
                | NSWindowStyleMaskFullSizeContentView)
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, WIDTH, HEIGHT), mask, NSBackingStoreBuffered, False)
        self.panel.setTitle_("Desktop Helper")
        self.panel.setTitlebarAppearsTransparent_(True)
        self.panel.setTitleVisibility_(1)  # hidden — the header bar is the title
        self.panel.setLevel_(NSFloatingWindowLevel)
        self.panel.setReleasedWhenClosed_(False)
        self.panel.setHidesOnDeactivate_(False)
        self.panel.center()

        content = self.panel.contentView()
        input_top = PAD + INPUT_H + 10

        self._submitter = _Submitter.alloc().initWithCallback_(self._submit)

        # -- header bar (VS Code-style: title left, actions right) -----------
        header = NSView.alloc().initWithFrame_(
            NSMakeRect(0, HEIGHT - HEADER_H, WIDTH, HEADER_H))
        header.setWantsLayer_(True)
        header.layer().setBackgroundColor_(
            NSColor.windowBackgroundColor().CGColor())
        title = NSTextField.labelWithString_("◆ Desktop Helper")
        title.setFont_(NSFont.boldSystemFontOfSize_(12))
        title.setTextColor_(NSColor.secondaryLabelColor())
        # offsets eyeballed with Kai against the traffic-light line
        title.setFrame_(NSMakeRect(TRAFFIC_LIGHT_CLEARANCE, (HEADER_H - 15) / 2 - 0.5,
                                   200, 15))
        header.addSubview_(title)
        actions = dict(actions or {})
        actions.setdefault("Clear", self.clear)
        self._submitter._actions = actions
        self._action_buttons = {}
        x = WIDTH - 10
        for name in reversed(list(actions)):
            w = 40 + 6 * len(name)  # roomy enough for state titles ("🎤 Stop")
            x -= w + 4
            btn = NSButton.alloc().initWithFrame_(
                NSMakeRect(x, (HEADER_H - 20) / 2 - 1.5, w, 20))
            btn.setTitle_(name)
            btn.setIdentifier_(name)  # stable dispatch key — titles change
            btn.setBezelStyle_(1)
            btn.setFont_(NSFont.systemFontOfSize_(10.5))
            btn.setTarget_(self._submitter)
            btn.setAction_("header:")
            header.addSubview_(btn)
            self._action_buttons[name] = btn
        separator = NSView.alloc().initWithFrame_(
            NSMakeRect(0, HEIGHT - HEADER_H - 1, WIDTH, 1))
        separator.setWantsLayer_(True)
        separator.layer().setBackgroundColor_(NSColor.separatorColor().CGColor())
        content.addSubview_(header)
        content.addSubview_(separator)

        self.scroll = NSScrollView.alloc().initWithFrame_(
            NSMakeRect(0, input_top, WIDTH, HEIGHT - input_top - HEADER_H - 2))
        self.scroll.setHasVerticalScroller_(True)
        self.scroll.setBorderType_(0)
        self.scroll.setDrawsBackground_(False)
        self.doc = _FlippedView.alloc().initWithFrame_(
            NSMakeRect(0, 0, WIDTH, 10))
        self.scroll.setDocumentView_(self.doc)
        content.addSubview_(self.scroll)

        ask_w = 52
        self.input = NSTextField.alloc().initWithFrame_(
            NSMakeRect(PAD, PAD, WIDTH - 2 * PAD - ask_w - 8, INPUT_H))
        self.input.setPlaceholderString_("Follow up…")
        self.input.setBezelStyle_(1)  # rounded
        self.input.setTarget_(self._submitter)
        self.input.setAction_("submit:")
        content.addSubview_(self.input)

        ask = NSButton.alloc().initWithFrame_(
            NSMakeRect(WIDTH - PAD - ask_w, PAD - 3, ask_w, INPUT_H + 6))
        ask.setTitle_("Ask")
        ask.setBezelStyle_(1)
        ask.setTarget_(self._submitter)
        ask.setAction_("submit:")
        content.addSubview_(ask)

        self._y = PAD
        self._pending = None           # the "…" assistant bubble awaiting a reply
        self._pending_question = None  # its question — dedupes the user bubble


    # -- bubbles -------------------------------------------------------------

    @staticmethod
    def _measure(text: str, font) -> tuple[float, float]:
        rect = NSString.stringWithString_(text).boundingRectWithSize_options_attributes_(
            (BUBBLE_MAX_W - 2 * BUBBLE_PAD_X, 100000),
            NSStringDrawingUsesLineFragmentOrigin,
            {NSFontAttributeName: font},
        )
        return rect.size.width, rect.size.height

    def _add_bubble(self, text: str, user: bool):
        font = NSFont.systemFontOfSize_(FONT_SIZE)
        tw, th = self._measure(text, font)
        bw = min(tw + 2 * BUBBLE_PAD_X + 4, BUBBLE_MAX_W)
        bh = th + 2 * BUBBLE_PAD_Y + 2
        x = WIDTH - PAD - bw - 14 if user else PAD  # 14: clear of the scroller

        bubble = NSView.alloc().initWithFrame_(NSMakeRect(x, self._y, bw, bh))
        bubble.setWantsLayer_(True)
        layer = bubble.layer()
        layer.setCornerRadius_(11.0)
        color = (NSColor.controlAccentColor() if user
                 else NSColor.unemphasizedSelectedContentBackgroundColor())
        layer.setBackgroundColor_(color.CGColor())

        label = NSTextField.wrappingLabelWithString_(text)
        label.setFont_(font)
        label.setSelectable_(True)
        if user:
            label.setTextColor_(NSColor.whiteColor())
        label.setFrame_(NSMakeRect(BUBBLE_PAD_X, BUBBLE_PAD_Y,
                                   bw - 2 * BUBBLE_PAD_X, th + 2))
        bubble.addSubview_(label)

        self.doc.addSubview_(bubble)
        self._animate_in(bubble)
        self._y += bh + BUBBLE_GAP
        self._relayout()
        return bubble

    @staticmethod
    def _animate_in(bubble) -> None:
        """iMessage-style entrance: fade in while rising into place."""
        layer = bubble.layer()
        ease = CAMediaTimingFunction.functionWithName_("easeOut")
        fade = CABasicAnimation.animationWithKeyPath_("opacity")
        fade.setFromValue_(0.0)
        fade.setToValue_(1.0)
        rise = CABasicAnimation.animationWithKeyPath_("transform.translation.y")
        rise.setFromValue_(14.0)  # flipped view: starts slightly below, rises up
        rise.setToValue_(0.0)
        for key, anim in (("in-fade", fade), ("in-rise", rise)):
            anim.setDuration_(ANIM_S)
            anim.setTimingFunction_(ease)
            layer.addAnimation_forKey_(anim, key)

    def _remove_bubble(self, bubble) -> None:
        self._y = bubble.frame().origin.y
        bubble.removeFromSuperview()

    def _relayout(self) -> None:
        height = max(self._y, self.scroll.contentSize().height)
        self.doc.setFrameSize_((WIDTH, height))
        # glide to the newest bubble instead of jumping
        clip = self.scroll.contentView()
        bottom = max(0.0, height - clip.bounds().size.height)
        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(ANIM_S)
        clip.animator().setBoundsOrigin_((0, bottom))
        NSAnimationContext.endGrouping()

    # -- api (main thread only) ----------------------------------------------

    def show(self, question: str, reply: str) -> None:
        if not self._resolve_pending(question):
            self._add_bubble(question, user=True)
        self._add_bubble(reply, user=False)
        self.input.setStringValue_("")
        self.panel.orderFrontRegardless()

    def show_sections(self, question: str, sections: list[str]) -> None:
        """One user bubble, then each section as its own assistant bubble
        (the briefing: greeting / weather / calendar / headlines)."""
        if not self._resolve_pending(question):
            self._add_bubble(question, user=True)
        for section in sections:
            self._add_bubble(section, user=False)
        self.panel.orderFrontRegardless()

    def show_thinking(self, question: str) -> None:
        if self._pending is not None:
            self._remove_bubble(self._pending)
        self._add_bubble(question, user=True)
        self._pending = self._add_bubble("…", user=False)
        self._pending_question = question
        self.panel.orderFrontRegardless()

    def _resolve_pending(self, question: str) -> bool:
        """Remove the '…' bubble if this reply answers it. True → the user
        bubble is already on screen (added by show_thinking)."""
        if self._pending is None:
            return False
        self._remove_bubble(self._pending)
        self._pending = None
        answered = question == self._pending_question
        self._pending_question = None
        return answered

    def set_action_state(self, name: str, active: bool,
                         active_title: str | None = None) -> None:
        """Reflect a mode on a header button (e.g. Speak → '🎤 Stop', red)."""
        btn = self._action_buttons.get(name)
        if btn is None:
            return
        btn.setTitle_((active_title or name) if active else name)
        try:
            btn.setBezelColor_(NSColor.systemRedColor() if active else None)
        except Exception:
            pass  # bezel tint is cosmetic; older macOS may lack it

    def clear(self) -> None:
        """Empty the conversation view."""
        for view in list(self.doc.subviews()):
            view.removeFromSuperview()
        self._y = PAD
        self._pending = None
        self._pending_question = None
        self._relayout()

    def close(self) -> None:
        self.panel.orderOut_(None)

    # -- internal ------------------------------------------------------------

    def _submit(self) -> None:
        text = str(self.input.stringValue()).strip()
        if not text:
            return
        self.input.setStringValue_("")
        self._on_followup(text)
