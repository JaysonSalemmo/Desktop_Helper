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
    NSBezelBorder,
    NSButton,
    NSColor,
    NSFloatingWindowLevel,
    NSFont,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSMakeRect,
    NSMenu,
    NSMenuItem,
    NSPanel,
    NSScrollView,
    NSStringDrawingUsesLineFragmentOrigin,
    NSTextField,
    NSTextView,
    NSView,
    NSViewHeightSizable,
    NSViewWidthSizable,
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
INPUT_MIN_H = 30   # input starts one line tall…
INPUT_MAX_H = 120  # …grows with content up to here, then scrolls (iMessage-style)
INPUT_INSET = 6    # text padding inside the rounded input
ASK_W = 52
BUBBLE_MAX_W = 340
BUBBLE_PAD_X = 12
BUBBLE_PAD_Y = 7
BUBBLE_GAP = 8
FONT_SIZE = 13


def install_edit_menu() -> None:
    """Register an Edit menu so ⌘X/⌘C/⌘V/⌘A work app-wide.

    macOS dispatches these key equivalents through the main menu's Edit items to
    the first responder (cut:/copy:/paste:/selectAll:). A menu-bar (LSUIElement)
    app has NO main menu by default, so without this every ⌘C/⌘A just beeps —
    even on selectable text. The menu bar isn't shown for an agent app, but its
    key equivalents still fire. Call once at startup; idempotent."""
    from AppKit import NSApplication
    app = NSApplication.sharedApplication()
    main = app.mainMenu()
    if main is None:
        main = NSMenu.alloc().init()
        app.setMainMenu_(main)
    if any(str(main.itemAtIndex_(i).title()) == "Edit"
           for i in range(main.numberOfItems())):
        return  # already installed
    edit_item = NSMenuItem.alloc().init()
    edit_menu = NSMenu.alloc().initWithTitle_("Edit")
    for title, action, key in (("Cut", "cut:", "x"), ("Copy", "copy:", "c"),
                               ("Paste", "paste:", "v"),
                               ("Select All", "selectAll:", "a")):
        edit_menu.addItem_(NSMenuItem.alloc()
                           .initWithTitle_action_keyEquivalent_(title, action, key))
    edit_item.setSubmenu_(edit_menu)
    main.addItem_(edit_item)


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


_SHIFT_MASK = 1 << 17  # NSEventModifierFlagShift


class _ChatInput(NSTextView):
    """Multi-line chat input: Enter sends, Shift+Enter inserts a newline (the
    standard chat pattern). Draws a placeholder while empty — NSTextView, unlike
    NSTextField, has none of its own."""

    def initWithFrame_submit_placeholder_onChange_(self, frame, submit,
                                                   placeholder, on_change):
        self = objc.super(_ChatInput, self).initWithFrame_(frame)
        if self is None:
            return None
        self._submit = submit
        self._placeholder = placeholder
        self._on_change = on_change
        return self

    def keyDown_(self, event):
        if event.keyCode() in (36, 76):  # Return / numpad Enter
            if not (event.modifierFlags() & _SHIFT_MASK):
                self._submit()
                return
        objc.super(_ChatInput, self).keyDown_(event)

    def didChangeText(self):
        # fired on every edit (typing, paste, delete) — drives the auto-grow
        objc.super(_ChatInput, self).didChangeText()
        if self._on_change is not None:
            self._on_change()

    def content_height(self) -> float:
        """Laid-out height of the current text, so the box can grow to fit."""
        lm = self.layoutManager()
        tc = self.textContainer()
        lm.ensureLayoutForTextContainer_(tc)
        return lm.usedRectForTextContainer_(tc).size.height

    def drawRect_(self, rect):
        objc.super(_ChatInput, self).drawRect_(rect)
        if self.string() == "":
            NSString.stringWithString_(self._placeholder).drawAtPoint_withAttributes_(
                (INPUT_INSET + 2, INPUT_INSET),
                {NSFontAttributeName: self.font(),
                 NSForegroundColorAttributeName: NSColor.placeholderTextColor()})


class _CloseWatcher(NSObject):
    """Window delegate that fires a callback when the panel is closed (its X
    button), so the app can re-enable the 'Show Chat' menu item."""

    def initWithCallback_(self, callback):
        self = objc.super(_CloseWatcher, self).init()
        if self is None:
            return None
        self._callback = callback
        return self

    def windowWillClose_(self, notification):
        self._callback()


class ReplyPanel:
    def __init__(self, on_followup, actions: dict | None = None,
                 on_visibility=None):
        """on_followup(text) is called on the main thread when submitted.
        actions: {"Speak": fn, "Briefing": fn, ...} → header-bar buttons.
        on_visibility(bool) fires when the window is shown (True) or closed
        (False) — lets the app grey out its 'Show Chat' item accordingly."""
        self._on_followup = on_followup
        self._on_visibility = on_visibility

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
        # become key on ANY click (default for non-activating panels is "only
        # if needed"), so clicking a reply to select it takes keyboard focus and
        # ⌘A / ⌘C reach the text — without this, copy silently fails
        self.panel.setBecomesKeyOnlyIfNeeded_(False)
        # notified when the user closes the window (X) so the app re-enables its
        # reopen item; kept as an attribute or ObjC would deallocate the delegate
        self._close_watcher = _CloseWatcher.alloc().initWithCallback_(
            lambda: self._notify_visible(False))
        self.panel.setDelegate_(self._close_watcher)
        self.panel.center()

        content = self.panel.contentView()
        self._input_h = INPUT_MIN_H  # current input height; grows with content
        input_top = PAD + self._input_h + 10

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

        input_w = WIDTH - 2 * PAD - ASK_W - 8
        # A plain NSView draws the rounded, bordered box reliably — an
        # NSScrollView's own layer border gets hidden by its clip view. The
        # (transparent, borderless) scroll view sits inside and auto-fills it.
        self.input_box = NSView.alloc().initWithFrame_(
            NSMakeRect(PAD, PAD, input_w, self._input_h))
        self.input_box.setWantsLayer_(True)
        _bl = self.input_box.layer()
        _bl.setCornerRadius_(INPUT_MIN_H / 2)  # capsule at one line, rounded when grown
        _bl.setMasksToBounds_(True)
        _bl.setBorderWidth_(1.0)
        _bl.setBorderColor_(NSColor.separatorColor().CGColor())
        _bl.setBackgroundColor_(NSColor.textBackgroundColor().CGColor())

        self.input_scroll = NSScrollView.alloc().initWithFrame_(
            NSMakeRect(0, 0, input_w, self._input_h))
        self.input_scroll.setHasVerticalScroller_(False)  # shown only past max
        self.input_scroll.setBorderType_(0)               # NSNoBorder
        self.input_scroll.setDrawsBackground_(False)
        self.input_scroll.contentView().setDrawsBackground_(False)
        self.input_scroll.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        self.input = _ChatInput.alloc().initWithFrame_submit_placeholder_onChange_(
            NSMakeRect(0, 0, input_w, self._input_h), self._submit,
            "Follow up…  (⇧⏎ for a new line)", self._resize_input)
        self.input.setFont_(NSFont.systemFontOfSize_(FONT_SIZE))
        self.input.setDrawsBackground_(False)  # let the box's fill show through
        self.input.setRichText_(False)
        self.input.setVerticallyResizable_(True)
        self.input.setHorizontallyResizable_(False)
        self.input.setTextContainerInset_((INPUT_INSET, INPUT_INSET))
        self.input.textContainer().setWidthTracksTextView_(True)
        self.input_scroll.setDocumentView_(self.input)
        self.input_box.addSubview_(self.input_scroll)
        content.addSubview_(self.input_box)

        # centered vertically on the one-line input (not stretched to its height)
        btn_h = 22
        ask = NSButton.alloc().initWithFrame_(NSMakeRect(
            WIDTH - PAD - ASK_W, PAD + (INPUT_MIN_H - btn_h) / 2, ASK_W, btn_h))
        ask.setTitle_("Ask")
        ask.setBezelStyle_(1)
        ask.setTarget_(self._submitter)
        ask.setAction_("submit:")
        content.addSubview_(ask)

        self._y = PAD
        self._pending = None           # the "…" assistant bubble awaiting a reply
        self._pending_question = None  # its question — dedupes the user bubble

    # -- auto-growing input --------------------------------------------------

    def _resize_input(self) -> None:
        """Grow (or shrink) the input to fit its text, iMessage-style: one line
        to start, taller as you type, capped at INPUT_MAX_H after which it
        scrolls internally."""
        wanted = self.input.content_height() + 2 * INPUT_INSET  # top+bottom padding
        new_h = max(INPUT_MIN_H, min(INPUT_MAX_H, wanted))
        if abs(new_h - self._input_h) < 0.5:
            return
        self._input_h = new_h
        # show the scroller only once we've hit the cap
        self.input_scroll.setHasVerticalScroller_(wanted > INPUT_MAX_H)
        self._reflow()

    def _reflow(self) -> None:
        """Reposition the input (anchored at the bottom) and the message area
        above it for the current input height."""
        input_w = WIDTH - 2 * PAD - ASK_W - 8
        # the scroll view auto-fills the box (autoresizing mask), so resize the box
        self.input_box.setFrame_(NSMakeRect(PAD, PAD, input_w, self._input_h))
        input_top = PAD + self._input_h + 10
        self.scroll.setFrame_(
            NSMakeRect(0, input_top, WIDTH, HEIGHT - input_top - HEADER_H - 2))

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

        # a selectable (read-only) NSTextView, NOT a label — NSTextView handles
        # ⌘A / ⌘C natively when it's first responder, so replies are copyable
        label = NSTextView.alloc().initWithFrame_(
            NSMakeRect(BUBBLE_PAD_X, BUBBLE_PAD_Y, bw - 2 * BUBBLE_PAD_X, th + 2))
        label.setString_(text)
        label.setFont_(font)
        label.setEditable_(False)
        label.setSelectable_(True)
        label.setDrawsBackground_(False)
        label.setTextContainerInset_((0, 0))
        label.textContainer().setLineFragmentPadding_(0)
        label.setTextColor_(NSColor.whiteColor() if user else NSColor.labelColor())
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

    def _notify_visible(self, visible: bool) -> None:
        if self._on_visibility is not None:
            self._on_visibility(visible)

    def _present_window(self) -> None:
        self.panel.orderFrontRegardless()
        self._notify_visible(True)

    def present(self) -> None:
        """Reopen the window (menu 'Show Chat') with its conversation intact."""
        self._present_window()

    def show(self, question: str, reply: str) -> None:
        if not self._resolve_pending(question):
            self._add_bubble(question, user=True)
        self._add_bubble(reply, user=False)
        self.input.setString_("")
        self.input.setNeedsDisplay_(True)  # repaint the placeholder
        self._resize_input()               # shrink back to one line
        self._present_window()

    def show_sections(self, question: str, sections: list[str]) -> None:
        """One user bubble, then each section as its own assistant bubble
        (the briefing: greeting / weather / calendar / headlines)."""
        if not self._resolve_pending(question):
            self._add_bubble(question, user=True)
        for section in sections:
            self._add_bubble(section, user=False)
        self._present_window()

    def show_thinking(self, question: str) -> None:
        if self._pending is not None:
            self._remove_bubble(self._pending)
        self._add_bubble(question, user=True)
        self._pending = self._add_bubble("…", user=False)
        self._pending_question = question
        self._present_window()

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
        self.panel.orderOut_(None)  # orderOut doesn't fire windowWillClose
        self._notify_visible(False)

    # -- internal ------------------------------------------------------------

    def _submit(self) -> None:
        text = str(self.input.string()).strip()
        if not text:
            return
        self.input.setString_("")
        self.input.setNeedsDisplay_(True)  # repaint the placeholder
        self._resize_input()               # shrink back to one line
        self._on_followup(text)
