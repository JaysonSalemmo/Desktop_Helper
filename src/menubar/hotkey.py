"""
Global hotkey via a raw Quartz event tap — replaces pynput.

Why not pynput: its macOS backend converts every tapped CGEvent into an
NSEvent on the listener thread; on macOS 15 the Caps Lock / input-source
path inside that conversion asserts it's on the main dispatch queue and
SIGTRAPs the whole process (observed live 2026-07-18). We only need
"is this keycode + these modifiers", which the raw CGEvent answers without
any AppKit involvement.

The tap is LISTEN-ONLY (can't block or modify input) and needs the same
Accessibility permission pynput did. Tap creation fails cleanly (ok=False)
when the permission is missing.
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

# kVK_ANSI_* virtual keycodes (letters, digits, space)
_KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8,
    "v": 9, "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "9": 25, "7": 26,
    "8": 28, "0": 29, "o": 31, "u": 32, "i": 34, "p": 35, "l": 37, "j": 38,
    "k": 40, "n": 45, "m": 46, "space": 49,
}


def parse_combo(combo: str) -> tuple[int, int]:
    """"<ctrl>+<space>" (pynput syntax, as in config) → (keycode, flag mask)."""
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
    if keycode is None or flags == 0:
        raise ValueError(f"hotkey needs at least one modifier and a key: {combo!r}")
    return keycode, flags


class HotkeyListener(threading.Thread):
    """Fires `on_press()` (from this thread — hop to main yourself) when the
    combo is pressed. Daemon thread; lives for the app's lifetime."""

    def __init__(self, keycode: int, flags: int, on_press):
        super().__init__(daemon=True, name="hotkey-tap")
        self.keycode = keycode
        self.flags = flags
        self.on_press = on_press
        self.ok = None  # None = starting, True = tap live, False = no permission

    def run(self) -> None:
        def callback(proxy, event_type, event, refcon):
            if event_type == Quartz.kCGEventTapDisabledByTimeout:
                Quartz.CGEventTapEnable(tap, True)
                return event
            if event_type == Quartz.kCGEventKeyDown:
                keycode = Quartz.CGEventGetIntegerValueField(
                    event, Quartz.kCGKeyboardEventKeycode)
                if keycode == self.keycode and \
                        (Quartz.CGEventGetFlags(event) & self.flags) == self.flags:
                    try:
                        self.on_press()
                    except Exception:
                        pass  # a callback error must never take down the tap
            return event

        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
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
