"""
Shared AppKit widgets for the assistant's UI.

Extracted from the retired `panel.py` when the orb became the only front-end:
these three are general-purpose, not specific to any one surface, and the orb
imported them across a module boundary that no longer had a reason to exist.
"""
import objc
from AppKit import (
    NSColor,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSMenu,
    NSMenuItem,
    NSTextView,
)
from Foundation import NSObject, NSString

INPUT_INSET = 6    # text padding inside the rounded input
_SHIFT_MASK = 1 << 17  # NSEventModifierFlagShift

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

class Submitter(NSObject):
    """ObjC target for the input field / Ask button."""

    def initWithCallback_(self, callback):
        self = objc.super(Submitter, self).init()
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

class ChatInput(NSTextView):
    """Multi-line chat input: Enter sends, Shift+Enter inserts a newline (the
    standard chat pattern). Draws a placeholder while empty — NSTextView, unlike
    NSTextField, has none of its own."""

    def initWithFrame_submit_placeholder_onChange_(self, frame, submit,
                                                   placeholder, on_change):
        self = objc.super(ChatInput, self).initWithFrame_(frame)
        if self is None:
            return None
        self._submit = submit
        self._placeholder = placeholder
        self._on_change = on_change
        # no smart typography in a command box: macOS otherwise rewrites the
        # typed ' into ’ (U+2019), which broke possessive extraction live
        # ("Find kai's resume" searched for "kai’s resume" verbatim)
        self.setAutomaticQuoteSubstitutionEnabled_(False)
        self.setAutomaticDashSubstitutionEnabled_(False)
        self.setAutomaticTextReplacementEnabled_(False)
        return self

    def keyDown_(self, event):
        if event.keyCode() in (36, 76):  # Return / numpad Enter
            if not (event.modifierFlags() & _SHIFT_MASK):
                self._submit()
                return
        objc.super(ChatInput, self).keyDown_(event)

    def didChangeText(self):
        # fired on every edit (typing, paste, delete) — drives the auto-grow
        objc.super(ChatInput, self).didChangeText()
        if self._on_change is not None:
            self._on_change()

    def content_height(self) -> float:
        """Laid-out height of the current text, so the box can grow to fit."""
        lm = self.layoutManager()
        tc = self.textContainer()
        lm.ensureLayoutForTextContainer_(tc)
        return lm.usedRectForTextContainer_(tc).size.height

    def drawRect_(self, rect):
        objc.super(ChatInput, self).drawRect_(rect)
        if self.string() != "":
            return
        attrs = {NSFontAttributeName: self.font(),
                 NSForegroundColorAttributeName: NSColor.placeholderTextColor()}
        text = NSString.stringWithString_(self._placeholder)
        if getattr(self, "_center_placeholder", False):
            # match the typed text's alignment/inset exactly — a placeholder
            # drawn at a fixed corner is what made the ask box look off-centre
            inset = self.textContainerInset()
            size = text.sizeWithAttributes_(attrs)
            origin = ((self.bounds().size.width - size.width) / 2.0, inset.height)
        else:
            origin = (INPUT_INSET + 2, INPUT_INSET)
        text.drawAtPoint_withAttributes_(origin, attrs)
