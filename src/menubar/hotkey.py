"""
Global hotkey via Carbon's RegisterEventHotKey — needs NO Accessibility.

History: this was a raw Quartz CGEventTap. That works, but an active event tap
requires the Accessibility permission (TCC), which for an ad-hoc/self-signed
frozen .app was endless pain — the grant resets on every rebuild/re-sign, is
hard to verify, and silently no-ops when the signature or entry is stale. The
tap would fall through (the keystroke typed into the focused app, or an AppKit
menu key-equivalent firing only while our app was frontmost).

RegisterEventHotKey is the API built for exactly this: register a specific
system-wide hotkey, get a Carbon event when it's pressed, and the key is
consumed for you. It needs no permission at all — it's how classic menu-bar
hotkey apps (Alfred, etc.) always did it. Carbon is a system framework loaded
via ctypes, so it behaves identically in the frozen bundle.

The handler is dispatched on the main thread by the app's Carbon/Cocoa event
loop (rumps runs NSApplication), so registration happens on the main thread and
callbacks arrive there too.
"""
import ctypes
import ctypes.util
from ctypes import (CFUNCTYPE, POINTER, Structure, byref, c_int32, c_uint32,
                    c_ulong, c_void_p, sizeof)

_carbon = ctypes.CDLL(ctypes.util.find_library("Carbon"))

# Carbon modifier masks used by RegisterEventHotKey (NOT the CGEvent flags)
_CARBON_MODIFIERS = {
    "cmd": 0x0100, "command": 0x0100,
    "shift": 0x0200,
    "alt": 0x0800, "option": 0x0800,
    "ctrl": 0x1000, "control": 0x1000,
}

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
    """"<ctrl>+<space>" (pynput syntax, as in config) → (keycode, Carbon mods).

    Function keys may be bound bare ("<f10>"); everything else needs at least
    one modifier so ordinary typing can't trigger the assistant."""
    keycode = None
    mods = 0
    for token in combo.lower().split("+"):
        token = token.strip().strip("<>")
        if token in _CARBON_MODIFIERS:
            mods |= _CARBON_MODIFIERS[token]
        elif token in _KEYCODES:
            keycode = _KEYCODES[token]
        else:
            raise ValueError(f"unsupported hotkey token: {token!r}")
    if keycode is None:
        raise ValueError(f"hotkey needs a key: {combo!r}")
    if mods == 0 and keycode not in _FUNCTION_KEYCODES:
        raise ValueError(f"non-function keys need at least one modifier: {combo!r}")
    return keycode, mods


# glyphs for showing a combo as menu text (display only — RegisterEventHotKey
# does the actual work, so these are NOT functional NSMenuItem keyEquivalents)
_SYMBOLS = {"ctrl": "⌃", "control": "⌃", "alt": "⌥", "option": "⌥",
            "shift": "⇧", "cmd": "⌘", "command": "⌘", "space": "Space"}


def combo_symbols(combo: str) -> str:
    """"<ctrl>+<shift>+<space>" → "⌃⇧Space" for a menu-item label."""
    parts = []
    for token in combo.lower().split("+"):
        token = token.strip().strip("<>")
        parts.append(_SYMBOLS.get(token, token.upper() if len(token) == 1 else token))
    return "".join(parts)


# --- Carbon event plumbing (ctypes) ---
class _EventTypeSpec(Structure):
    _fields_ = [("eventClass", c_uint32), ("eventKind", c_uint32)]


class _EventHotKeyID(Structure):
    _fields_ = [("signature", c_uint32), ("id", c_uint32)]


_kEventClassKeyboard = 0x6b657962      # 'keyb'
_kEventHotKeyPressed = 6
_kEventParamDirectObject = 0x2d2d2d2d  # '----'
_typeEventHotKeyID = 0x686b6964        # 'hkid'
_HOTKEY_SIGNATURE = 0x44534852         # 'DSHR'
_noErr = 0

# OSStatus handler(EventHandlerCallRef, EventRef, void *userData)
_HANDLER_FUNC = CFUNCTYPE(c_int32, c_void_p, c_void_p, c_void_p)

_carbon.GetApplicationEventTarget.restype = c_void_p
_carbon.RegisterEventHotKey.argtypes = [c_uint32, c_uint32, _EventHotKeyID,
                                        c_void_p, c_uint32, POINTER(c_void_p)]
_carbon.RegisterEventHotKey.restype = c_int32
_carbon.UnregisterEventHotKey.argtypes = [c_void_p]
_carbon.UnregisterEventHotKey.restype = c_int32
_carbon.InstallEventHandler.argtypes = [c_void_p, c_void_p, c_uint32,
                                        POINTER(_EventTypeSpec), c_void_p,
                                        POINTER(c_void_p)]
_carbon.InstallEventHandler.restype = c_int32
_carbon.GetEventParameter.argtypes = [c_void_p, c_uint32, c_uint32, c_void_p,
                                      c_ulong, c_void_p, c_void_p]
_carbon.GetEventParameter.restype = c_int32


class HotkeyListener:
    """Registers global hotkeys via Carbon. `bindings` is a list of
    (keycode, carbon_modifiers, callback). MUST be started on the main thread;
    callbacks fire on the main thread (the app's Carbon event loop dispatches
    them). `ok` mirrors the old tap interface: True once at least one hotkey
    registered, False on failure."""

    def __init__(self, bindings: list[tuple[int, int, object]]):
        self.bindings = bindings
        self.ok = None  # None = not started, True = registered, False = failed
        self._by_id: dict[int, object] = {}
        self._refs: list[c_void_p] = []
        self._handler = None  # keep the CFUNCTYPE alive or the callback is GC'd

    def start(self) -> None:
        from src.applog import get_logger
        log = get_logger()
        try:
            target = _carbon.GetApplicationEventTarget()
            self._handler = _HANDLER_FUNC(self._on_event)  # stored → not GC'd
            spec = _EventTypeSpec(_kEventClassKeyboard, _kEventHotKeyPressed)
            handler_ref = c_void_p()
            status = _carbon.InstallEventHandler(target, self._handler, 1,
                                                 byref(spec), None,
                                                 byref(handler_ref))
            if status != _noErr:
                self.ok = False
                log.warning("hotkey handler install failed (status %d)", status)
                return
            for i, (keycode, mods, callback) in enumerate(self.bindings, start=1):
                hk_id = _EventHotKeyID(_HOTKEY_SIGNATURE, i)
                ref = c_void_p()
                status = _carbon.RegisterEventHotKey(keycode, mods, hk_id,
                                                     target, 0, byref(ref))
                if status != _noErr:
                    log.warning("RegisterEventHotKey failed for binding %d "
                                "(status %d)", i, status)
                    continue
                self._by_id[i] = callback
                self._refs.append(ref)
            self.ok = bool(self._by_id)
            log.info("hotkey registered: %d/%d bindings live (no Accessibility "
                     "needed)", len(self._by_id), len(self.bindings))
        except Exception:
            self.ok = False
            log.exception("hotkey registration crashed")

    def _on_event(self, call_ref, event, user_data) -> int:
        try:
            hk_id = _EventHotKeyID()
            _carbon.GetEventParameter(event, _kEventParamDirectObject,
                                      _typeEventHotKeyID, None,
                                      sizeof(hk_id), None, byref(hk_id))
            callback = self._by_id.get(hk_id.id)
            if callback is not None:
                callback()
        except Exception:
            pass  # a callback error must never propagate into the Carbon loop
        return _noErr

    def start_thread_safe(self) -> None:
        # kept for interface parity with the old threaded listener
        self.start()
