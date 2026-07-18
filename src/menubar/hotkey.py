"""
Global hotkey via a raw Quartz event tap — replaces pynput.

Why not pynput: its macOS backend converts every tapped CGEvent into an
NSEvent on the listener thread; on macOS 15 the Caps Lock / input-source
path inside that conversion asserts it's on the main dispatch queue and
SIGTRAPs the whole process (observed live 2026-07-18). We only need
"is this keycode + these modifiers", which the raw CGEvent answers without
any AppKit involvement.

The tap is ACTIVE: matched hotkey events are consumed (otherwise the
keystroke also lands in whatever has focus — e.g. a space typed into the
panel's follow-up field); unmatched events pass through untouched. The
callback is a few integer compares, so keyboard latency is unaffected.
Needs the same Accessibility permission pynput did; tap creation fails
cleanly (ok=False) when it's missing.
"""
import threading

import Quartz

_MODIFIERS = {
    "ctrl": Quartz.kCGEventFlagMaskControl,
    "control": Quartz.kCGEventFlagMaskControl,
    "cmd": Quartz.kCGEventFlagMaskCommand,
    "alt": Quartz.kCGEventFlagMaskAlternate,
    "option": Quartz.kCGEventFlagMaskAlternate,
    "shift": Quartz.kCGEventFlagMaskShift,
}

# all modifier bits we consider when matching — a binding matches only if its
# required modifiers are pressed AND no other of these are (exact matching, so
# ctrl+space doesn't also fire on ctrl+shift+space)
_RELEVANT_FLAGS = (Quartz.kCGEventFlagMaskControl | Quartz.kCGEventFlagMaskCommand
                   | Quartz.kCGEventFlagMaskAlternate | Quartz.kCGEventFlagMaskShift)

# kVK_* virtual keycodes (letters, digits, space, function keys)
_KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8,
    "v": 9, "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "9": 25, "7": 26,
    "8": 28, "0": 29, "o": 31, "u": 32, "i": 34, "p": 35, "l": 37, "j": 38,
    "k": 40, "n": 45, "m": 46, "space": 49,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98,
    "f8": 100, "f9": 101, "f10": 109, "f11": 103, "f12": 111,
}

_FUNCTION_KEYCODES = {_KEYCODES[f"f{n}"] for n in range(1, 13)}


def parse_combo(combo: str) -> tuple[int, int]:
    """"<ctrl>+<space>" (pynput syntax, as in config) → (keycode, flag mask).

    Function keys may be bound bare ("<f10>"); everything else needs at least
    one modifier so ordinary typing can't trigger the assistant."""
    keycode = None
    flags = 0
    for token in combo.lower().split("+"):
        token = token.strip().strip("<>")
        if token in _MODIFIERS:
            flags |= _MODIFIERS[token]
        elif token in _KEYCODES:
            keycode = _KEYCODES[token]
        else:
            raise ValueError(f"unsupported hotkey token: {token!r}")
    if keycode is None:
        raise ValueError(f"hotkey needs a key: {combo!r}")
    if flags == 0 and keycode not in _FUNCTION_KEYCODES:
        raise ValueError(f"non-function keys need at least one modifier: {combo!r}")
    return keycode, flags


def flags_match(event_flags: int, required: int) -> bool:
    """Exact modifier match over the relevant bits."""
    return (event_flags & _RELEVANT_FLAGS) == required


# NSEvent modifier masks — for DISPLAYING combos as native menu shortcuts
_NS_MODMASKS = {"ctrl": 1 << 18, "control": 1 << 18, "cmd": 1 << 20,
                "alt": 1 << 19, "option": 1 << 19, "shift": 1 << 17}


def combo_display(combo: str) -> tuple[str, int] | None:
    """Combo → (keyEquivalent char, NSEvent modifier mask) so menu items can
    show the shortcut natively (dim, right-aligned). Display only — the event
    tap consumes matched combos before the app would ever see them."""
    char = None
    mask = 0
    for token in combo.lower().split("+"):
        token = token.strip().strip("<>")
        if token in _NS_MODMASKS:
            mask |= _NS_MODMASKS[token]
        elif token == "space":
            char = " "
        elif len(token) == 1:
            char = token
        elif token.startswith("f") and token[1:].isdigit():
            char = chr(0xF704 + int(token[1:]) - 1)  # NSF1FunctionKey + n
    return (char, mask) if char else None


class HotkeyListener(threading.Thread):
    """One event tap serving multiple keybinds. `bindings` is a list of
    (keycode, flags, callback); callbacks fire from this thread — hop to main
    yourself. Daemon thread; lives for the app's lifetime."""

    def __init__(self, bindings: list[tuple[int, int, object]]):
        super().__init__(daemon=True, name="hotkey-tap")
        self.bindings = bindings
        self.ok = None  # None = starting, True = tap live, False = no permission

    def run(self) -> None:
        def callback(proxy, event_type, event, refcon):
            if event_type == Quartz.kCGEventTapDisabledByTimeout:
                Quartz.CGEventTapEnable(tap, True)
                return event
            if event_type == Quartz.kCGEventKeyDown:
                keycode = Quartz.CGEventGetIntegerValueField(
                    event, Quartz.kCGKeyboardEventKeycode)
                flags = Quartz.CGEventGetFlags(event)
                for want_key, want_flags, on_press in self.bindings:
                    if keycode == want_key and flags_match(flags, want_flags):
                        try:
                            on_press()
                        except Exception:
                            pass  # a callback error must never take down the tap
                        return None  # consume — don't also type into the focused app
            return event

        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionDefault,
            Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown),
            callback,
            None,
        )
        if tap is None:  # Accessibility permission not granted
            self.ok = False
            return
        source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), source,
                                  Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(tap, True)
        self.ok = True
        Quartz.CFRunLoopRun()
