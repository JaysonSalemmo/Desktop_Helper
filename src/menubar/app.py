"""
Menu bar frontend — the assistant living in the macOS status bar.

Run: uv run python -m src.menubar

Architecture (macOS constraints drive all of it):
- rumps/AppKit owns the MAIN thread; all UI (windows, alerts, title changes)
  must happen there. Anything called from another thread goes through
  `AppHelper.callAfter`, which schedules onto the main event loop.
- The model loads on a daemon worker thread at startup (same as the TUI);
  inference likewise runs on a worker so the menu bar never freezes.
- The global hotkey (config "hotkey", pynput syntax e.g. "<ctrl>+<space>")
  runs on pynput's listener thread. It needs the macOS Input Monitoring /
  Accessibility permission; if the listener can't start, the menu item still
  works — the hotkey is a convenience, not a dependency.
- Voice (config features.voice): the hotkey becomes a talk toggle — press to
  start listening (🎤), press again to stop; the recording is transcribed
  locally by faster-whisper and fed to the dispatcher exactly like a typed
  message. "Ask…" still opens the typed prompt either way. First mic use
  triggers the macOS microphone permission prompt.

Title doubles as a status indicator: ⏳ loading → ◆ ready → 🎤 listening →
… thinking.
"""
import threading
import traceback

import rumps
from AppKit import NSApplication, NSFloatingWindowLevel, NSImage
from PyObjCTools import AppHelper

from src.applog import get_logger
from src.assistant.engine import load_engine
from src.config import settings
from src.paths import is_frozen, resource_path

ICON_PATH = resource_path("assets", "AppIcon.icns")

TITLE_LOADING = "⏳"
TITLE_READY = "◆"
TITLE_THINKING = "…"
TITLE_LISTENING = "🎤"


class DesktopHelperMenuBar(rumps.App):
    def __init__(self):
        super().__init__("Desktop Helper", title=TITLE_LOADING, quit_button="Quit")
        # our icon on dialogs instead of the Python rocket (the process is
        # python, so NSApp otherwise inherits python's icon)
        if ICON_PATH.exists():
            icon = NSImage.alloc().initWithContentsOfFile_(str(ICON_PATH))
            if icon:
                NSApplication.sharedApplication().setApplicationIconImage_(icon)
        self.config = settings.load()
        self.dispatcher = None
        self.busy = False
        self.hotkey_combo = self.config.get("hotkey", "<ctrl>+<space>")

        self.voice_enabled = self.config.get("features", {}).get("voice", False)
        # speak replies aloud for voice-initiated turns (say what you heard)
        self.voice_replies = self.config.get("features", {}).get("voice_replies", True)
        self.recorder = None
        self.transcriber = None
        if self.voice_enabled:
            from src.voice.voice import Transcriber, VoiceRecorder
            self.recorder = VoiceRecorder(on_level=self._presence_level)
            self.transcriber = Transcriber()

        self.status_item = rumps.MenuItem("Loading model…")  # no callback → non-clickable
        self.ask_item = rumps.MenuItem("Ask…", callback=self._ask_clicked)
        # reopens the chat window; greyed while it's already showing (a callback
        # of None renders disabled — see _panel_visibility)
        self.show_chat_item = rumps.MenuItem("Show Chat", callback=self._show_chat_clicked)
        # keep the two keybound actions together at the top — Ask, then Speak —
        # with Show Chat right below them (Kai's layout).
        menu = [self.ask_item]
        if self.voice_enabled:
            self.speak_item = rumps.MenuItem("Speak", callback=self._toggle_voice)
            menu.append(self.speak_item)
        menu.append(self.show_chat_item)
        if self.voice_enabled:
            self.voice_replies_item = rumps.MenuItem("Voice Replies",
                                                     callback=self._toggle_voice_replies)
            self.voice_replies_item.state = 1 if self.voice_replies else 0
            menu.append(self.voice_replies_item)
            self.wake_enabled = self.config.get("features", {}).get("wake_word", False)
            if self.wake_enabled:
                # menu toggle only shows when the feature is opted into via
                # config — parked until the custom "hey helper" model exists
                self.wake_item = rumps.MenuItem("Wake Word (“hey Jarvis”)",
                                                callback=self._toggle_wake)
                self.wake_item.state = 1
                menu.append(self.wake_item)
        self.briefing_item = rumps.MenuItem("Morning Briefing", callback=self._briefing_clicked)
        menu.append(self.briefing_item)
        self.menu = [*menu, None, self.status_item]
        self._apply_keybind_labels()

        self._panel = None  # OrbWindow, created lazily on the main thread
        self._wake = None   # WakeWordListener, created on first enable
        if self.voice_enabled and getattr(self, "wake_enabled", False):
            self._start_wake()
        self._hotkeys = None
        self._start_hotkey()  # before the load thread — _load_model reads _hotkeys
        # scheduled, so it runs once the main loop is up
        self._on_main(self._park_orb)
        threading.Thread(target=self._load_model, daemon=True).start()
        if self.config.get("features", {}).get("startup_briefing", False):
            # briefing needs no model — deliver as a notification while it loads
            threading.Thread(target=self._startup_briefing, daemon=True).start()

    def _apply_keybind_labels(self) -> None:
        """Show each action's global keybind in its menu item title. DISPLAY
        ONLY — a functional NSMenuItem keyEquivalent would double-fire with the
        real global hotkey (RegisterEventHotKey), and only worked while the app
        was frontmost anyway. Plain text sidesteps both."""
        from src.menubar.hotkey import combo_symbols
        keybinds = dict(self.config.get("keybinds") or {})
        if not keybinds:
            keybinds["speak" if self.voice_enabled else "ask"] = self.hotkey_combo
        items = {"ask": self.ask_item, "briefing": self.briefing_item}
        if self.voice_enabled:
            items["speak"] = self.speak_item
        for name, combo in keybinds.items():
            item = items.get(name)
            if item is None or not combo:
                continue
            try:
                base = item.title.split("  ")[0]  # idempotent if re-applied
                item.title = f"{base}  {combo_symbols(combo)}"
                if name == "speak":
                    # the Speak title is rewritten during recording ("Stop
                    # listening" → "Speak"); remember the labelled idle title so
                    # those resets restore the keybind instead of dropping it
                    self._speak_idle_title = item.title
            except Exception:
                pass  # cosmetic — rumps internals may shift

    # -- error visibility ----------------------------------------------------

    def _on_main(self, fn, *args) -> None:
        """callAfter with a safety net: exceptions in main-thread UI callbacks
        otherwise vanish silently (a turn 'stops working' with no trace).
        Logs the traceback and releases the busy flag so the app recovers."""
        AppHelper.callAfter(self._safely, fn, *args)

    def _safely(self, fn, *args) -> None:
        try:
            fn(*args)
        except Exception:
            print(f"UI error in {getattr(fn, '__name__', fn)}:\n{traceback.format_exc()}",
                  flush=True)
            self.busy = False
            self.title = TITLE_READY

    # -- startup ------------------------------------------------------------

    def _load_model(self) -> None:
        log = get_logger()
        log.info("loading model from %s", self.config.get("model", {}).get("checkpoint"))
        try:
            dispatcher, device = load_engine(self.config)
        except Exception as exc:
            log.exception("model failed to load")
            self._on_main(self._set_status, "⚠️", f"Model failed to load: {exc}")
            return
        log.info("model ready on %s", device.type)
        self.dispatcher = dispatcher
        hotkey_live = self._hotkeys is not None and self._hotkeys.ok
        from src.menubar.hotkey import combo_symbols
        speak_combo = (self.config.get("keybinds") or {}).get("speak") or self.hotkey_combo
        hotkey_hint = f"  ({combo_symbols(speak_combo)})" if hotkey_live else ""
        self._ready_status = f"Ready on {device.type}{hotkey_hint}"
        self._on_main(self._set_status, TITLE_READY, self._ready_status)
        # the orb is a presence, not a window — it takes its place on the
        # desktop as soon as there's a model behind it, and breathes there
        self._on_main(self._park_orb)
        if self.transcriber is not None:
            self.transcriber.warm_up()  # download/load whisper off the hot path

    def _set_status(self, title: str, status: str) -> None:
        self.title = title
        self.status_item.title = status

    # -- presence (the header's living indicator) ----------------------------

    def _presence(self, state: str) -> None:
        """Drive the panel's presence view. Never forces the panel into
        existence — a state change is not a reason to build the window.
        getattr: the recorder binds these before _panel is assigned."""
        panel = getattr(self, "_panel", None)
        if panel is not None:
            panel.set_presence(state)

    def _presence_level(self, level: float, bands=None) -> None:
        """Called from the AUDIO thread — hop to the main thread for AppKit."""
        panel = getattr(self, "_panel", None)
        if panel is not None:
            self._on_main(panel.set_presence_level, level, bands)

    def _start_hotkey(self) -> None:
        try:
            from src.menubar.hotkey import HotkeyListener, parse_combo
            actions = {
                "speak": self._toggle_voice if self.voice_enabled else self._ask_clicked,
                "ask": self._ask_clicked,
                "briefing": self._briefing_clicked,
            }
            keybinds = dict(self.config.get("keybinds") or {})
            if not keybinds:
                # legacy single "hotkey" → speak (or ask when voice is off)
                keybinds["speak"] = self.hotkey_combo
            bindings = []
            for name, combo in keybinds.items():
                action = actions.get(name)
                if not combo or action is None:
                    continue
                try:
                    keycode, flags = parse_combo(combo)
                except ValueError:
                    continue  # bad combo in config — skip it, don't die
                # fires on the tap thread → hop to the main thread
                bindings.append((keycode, flags,
                                 lambda a=action: self._on_main(a, None)))
            self._hotkeys = HotkeyListener(bindings) if bindings else None
            if self._hotkeys is not None:
                self._hotkeys.start()
        except Exception:
            # tap setup failure — the menu items still work
            self._hotkeys = None

    # -- ask flow (main thread) ---------------------------------------------

    def _ask_clicked(self, _sender) -> None:
        if self.busy:
            return
        if self.dispatcher is None:
            rumps.alert("Desktop Helper", "Still loading the model — one moment.")
            return

        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        window = rumps.Window(
            message="", title="Desktop Helper", default_text="",
            ok="Ask", cancel="Cancel", dimensions=(380, 96),  # taller: multi-line asks
        )
        try:
            # let long/pasted text wrap to multiple lines instead of scrolling
            # off one line (rumps' field is single-line by default)
            field = window._textfield
            field.setUsesSingleLineMode_(False)
            field.cell().setWraps_(True)
            field.cell().setScrollable_(False)
        except Exception:
            pass  # rumps internal — best effort, never break the ask flow
        try:
            self._float_front(window._alert.window())  # rumps internal — best effort
        except AttributeError:
            pass
        response = window.run()
        message = response.text.strip()
        if not response.clicked or not message:
            return

        self.busy = True
        self.title = TITLE_THINKING
        self._presence("thinking")
        threading.Thread(target=self._respond, args=(message,), daemon=True).start()

    def _toggle_voice_replies(self, sender) -> None:
        """Menu checkmark toggling spoken replies; persisted to config.json."""
        self.voice_replies = not self.voice_replies
        sender.state = 1 if self.voice_replies else 0
        self.config.setdefault("features", {})["voice_replies"] = self.voice_replies
        try:
            settings.save(self.config)
        except Exception:
            pass  # toggle still applies for this session

    # -- wake word -----------------------------------------------------------

    def _start_wake(self) -> None:
        if self._wake is None:
            from src.voice import voice as voice_mod
            from src.voice.wakeword import WakeWordListener
            self._wake = WakeWordListener(
                on_wake=lambda: self._on_main(self._wake_triggered),
                # never trigger on our own say voice, mid-turn, or while the
                # push-to-talk recorder owns the conversation
                gate=lambda: (self.busy or voice_mod.is_speaking()
                              or (self.recorder is not None and self.recorder.recording)),
            )
            self._wake.start()
        else:
            self._wake.resume()

    def _toggle_wake(self, sender) -> None:
        self.wake_enabled = not self.wake_enabled
        sender.state = 1 if self.wake_enabled else 0
        self.config.setdefault("features", {})["wake_word"] = self.wake_enabled
        try:
            settings.save(self.config)
        except Exception:
            pass
        if self.wake_enabled:
            self._start_wake()
        elif self._wake is not None:
            self._wake.pause()

    def _wake_triggered(self) -> None:
        """Main thread — the wake phrase was heard."""
        if self.busy or self.dispatcher is None or \
                (self.recorder is not None and self.recorder.recording):
            return
        from src.voice import wakeword
        wakeword.chirp()
        self.busy = True
        self.title = TITLE_LISTENING
        self._presence("listening")
        self.status_item.title = "Listening… (speak now)"
        self._get_panel().set_action_state("Speak", True, "🎤 …")
        if self._wake is not None:
            self._wake.pause()
        threading.Thread(target=self._wake_capture, daemon=True).start()

    def _wake_capture(self) -> None:
        """Worker: record until silence, then the normal voice pipeline."""
        from src.voice import wakeword
        try:
            audio = wakeword.capture_until_silence()
        except Exception as exc:
            self._on_main(self._show_reply, "Voice", f"Capture failed: {exc}")
            return
        finally:
            if self._wake is not None and self.wake_enabled:
                self._wake.resume()
        self._on_main(self._wake_capture_done)
        self._transcribe_and_respond(audio)

    def _wake_capture_done(self) -> None:
        self.title = TITLE_THINKING
        self._presence("thinking")
        self.status_item.title = "Transcribing…"
        self._get_panel().set_action_state("Speak", False)

    # -- briefing ------------------------------------------------------------

    def _briefing_clicked(self, _sender) -> None:
        threading.Thread(target=self._show_briefing, daemon=True).start()

    def _startup_briefing(self) -> None:
        # alert, not a notification: NSUserNotification from our launcher-style
        # process "succeeds" without visibly delivering (no registered bundle).
        # Revisit notifications if the app ever becomes a real py2app freeze.
        self._show_briefing()

    def _show_briefing(self) -> None:
        """Worker thread: compose (network + EventKit) then show as bubbles."""
        from src.briefing import briefing
        try:
            sections = briefing.compose_sections(self.config)
        except Exception as exc:
            sections = [f"Briefing failed: {exc}"]
        self._on_main(self._show_briefing_sections, sections)

    def _show_briefing_sections(self, sections: list[str]) -> None:
        self._get_panel().show_sections("Morning Briefing", sections)

    @staticmethod
    def _float_front(ns_window) -> None:
        """Menu-bar apps can't force activation on modern macOS (alerts open
        BEHIND the current app, needing a Dock hunt to dismiss) — a floating
        window level + orderFrontRegardless shows them on top anyway."""
        ns_window.setLevel_(NSFloatingWindowLevel)
        ns_window.orderFrontRegardless()

    def _get_panel(self):
        if self._panel is None:
            from src.menubar.widgets import install_edit_menu
            install_edit_menu()  # ⌘C/⌘A/⌘V in the panel — needs the app fully up
            actions = {}
            if self.voice_enabled:
                actions["Speak"] = lambda: self._toggle_voice(None)
            actions["Briefing"] = lambda: self._briefing_clicked(None)
            from src.menubar.orb import OrbWindow
            self._panel = OrbWindow(on_followup=self._followup, actions=actions,
                                    on_visibility=self._panel_visibility)
        return self._panel

    def _park_orb(self) -> None:
        """Put the orb on the desktop. Before the model is ready it shows the
        loading sweep, so the ~50s startup is visible instead of a dead orb."""
        surface = self._get_panel()
        park = getattr(surface, "park", None)
        if park is not None:
            park()
        self._presence("dormant" if self.dispatcher is not None else "loading")

    def _panel_visibility(self, visible: bool) -> None:
        """Grey out 'Show Chat' while the window is showing, restore it when
        hidden (a None callback renders the item disabled in rumps)."""
        self.show_chat_item.set_callback(None if visible else self._show_chat_clicked)

    def _show_chat_clicked(self, _sender=None) -> None:
        self._get_panel().present()

    def _show_alert(self, title: str, body: str) -> None:
        """Non-modal reply panel (replaced the modal NSAlert flow — nothing
        blocks; the follow-up field keeps the conversation going)."""
        self._get_panel().show(title, body)

    def _followup(self, text: str) -> None:
        """Main thread — panel input submitted."""
        if self.busy or self.dispatcher is None:
            return
        self.busy = True
        self.title = TITLE_THINKING
        self._get_panel().show_thinking(text)
        threading.Thread(target=self._respond, args=(text,), daemon=True).start()

    # -- voice flow ----------------------------------------------------------

    def _toggle_voice(self, _sender) -> None:
        """Main thread. First press starts listening, second press stops,
        transcribes (worker thread), and dispatches like a typed message."""
        if self.busy or self.dispatcher is None:
            return
        if not self.recorder.recording:
            try:
                self.recorder.start()  # first use triggers the mic permission prompt
            except Exception as exc:
                rumps.alert("Desktop Helper", f"Couldn't open the microphone: {exc}")
                return
            self.title = TITLE_LISTENING
            self._presence("listening")
            self.status_item.title = f"Listening… ({self.hotkey_combo} to stop)"
            if self.voice_enabled:
                self.speak_item.title = "Stop listening"
            self._get_panel().set_action_state("Speak", True, "🎤 Stop")
            return

        audio = self.recorder.stop()
        self.busy = True
        self.title = TITLE_THINKING
        self.status_item.title = "Transcribing…"
        self._presence("thinking")
        if self.voice_enabled:
            self.speak_item.title = getattr(self, "_speak_idle_title", "Speak")
        self._get_panel().set_action_state("Speak", False)
        threading.Thread(target=self._transcribe_and_respond, args=(audio,),
                         daemon=True).start()

    def _transcribe_and_respond(self, audio) -> None:
        """Worker thread: whisper → dispatcher → reply alert."""
        try:
            text = self.transcriber.transcribe(audio)
        except Exception as exc:
            self._on_main(self._show_reply, "Voice", f"Transcription failed: {exc}")
            return
        if not text:
            self._on_main(self._voice_heard_nothing)
            return
        self._on_main(self._set_status, TITLE_THINKING, f"“{text}”")
        self._respond(text, spoken=True)

    def _voice_heard_nothing(self) -> None:
        self.busy = False
        self.title = TITLE_READY
        self.status_item.title = "Didn't catch that — try again"

    # -- inference (worker thread) ------------------------------------------

    def _respond(self, message: str, spoken: bool = False) -> None:
        spoken_reply = None
        try:
            result = self.dispatcher.respond(message)
            body = result.response
            if spoken and self.voice_replies:
                spoken_reply = result.response  # you talked to it — it talks back
        except Exception as exc:
            body = f"Error: {exc}"
        # show first, THEN speak: _show_reply drops the orb to dormant, so
        # starting playback before it would have the reset stomp the speaking state
        self._on_main(self._show_reply, message, body)
        if spoken_reply is not None:
            self._speak_aloud(spoken_reply)

    def _speak_aloud(self, text: str) -> None:
        """Speak, with the orb moving to the VOLUME of the speech (not merely
        'on') — `speak` plays the utterance itself so it can meter amplitude."""
        from src.voice.voice import is_speaking, speak
        self._on_main(self._presence, "speaking")
        speak(text, on_level=self._presence_level)

        def settle() -> None:
            import time
            deadline = time.time() + 180
            while time.time() < deadline and not is_speaking():
                time.sleep(0.05)  # synthesis runs before playback begins
            while time.time() < deadline and is_speaking():
                time.sleep(0.1)
            self._on_main(self._presence, "dormant")

        threading.Thread(target=settle, daemon=True).start()

    def _show_reply(self, message: str, body: str) -> None:
        self.busy = False
        self.title = TITLE_READY
        self._presence("dormant")
        self.status_item.title = getattr(self, "_ready_status", "Ready")
        self._show_alert(message, body)


def main() -> None:
    log = get_logger()
    log.info("Desktop Helper starting (frozen=%s)", is_frozen())
    try:
        DesktopHelperMenuBar().run()
    except Exception:
        log.exception("fatal error in menu bar app")
        raise


if __name__ == "__main__":
    main()
