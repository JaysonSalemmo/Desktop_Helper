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

import rumps
from AppKit import NSAlert, NSApplication, NSFloatingWindowLevel, NSImage
from PyObjCTools import AppHelper

from src.assistant.engine import PROJECT_ROOT, load_engine
from src.config import settings

ICON_PATH = PROJECT_ROOT / "assets" / "AppIcon.icns"

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
            self.recorder = VoiceRecorder()
            self.transcriber = Transcriber()

        self.status_item = rumps.MenuItem("Loading model…")  # no callback → non-clickable
        self.ask_item = rumps.MenuItem("Ask…", callback=self._ask_clicked)
        menu = [self.ask_item]
        if self.voice_enabled:
            self.speak_item = rumps.MenuItem("Speak", callback=self._toggle_voice)
            menu.append(self.speak_item)
            self.voice_replies_item = rumps.MenuItem("Voice Replies",
                                                     callback=self._toggle_voice_replies)
            self.voice_replies_item.state = 1 if self.voice_replies else 0
            menu.append(self.voice_replies_item)
        self.briefing_item = rumps.MenuItem("Morning Briefing", callback=self._briefing_clicked)
        menu.append(self.briefing_item)
        self.menu = [*menu, None, self.status_item]

        self._hotkeys = None
        self._start_hotkey()  # before the load thread — _load_model reads _hotkeys
        threading.Thread(target=self._load_model, daemon=True).start()
        if self.config.get("features", {}).get("startup_briefing", False):
            # briefing needs no model — deliver as a notification while it loads
            threading.Thread(target=self._startup_briefing, daemon=True).start()

    # -- startup ------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            dispatcher, device = load_engine(self.config)
        except Exception as exc:
            AppHelper.callAfter(self._set_status, "⚠️", f"Model failed to load: {exc}")
            return
        self.dispatcher = dispatcher
        hotkey_hint = f"  ({self.hotkey_combo})" if self._hotkeys else ""
        self._ready_status = f"Ready on {device.type}{hotkey_hint}"
        AppHelper.callAfter(self._set_status, TITLE_READY, self._ready_status)
        if self.transcriber is not None:
            self.transcriber.warm_up()  # download/load whisper off the hot path

    def _set_status(self, title: str, status: str) -> None:
        self.title = title
        self.status_item.title = status

    def _start_hotkey(self) -> None:
        try:
            from pynput import keyboard
            # voice on: hotkey toggles listening; voice off: hotkey opens the prompt
            action = self._toggle_voice if self.voice_enabled else self._ask_clicked
            self._hotkeys = keyboard.GlobalHotKeys({
                # hotkey fires on pynput's thread → hop to the main thread
                self.hotkey_combo: lambda: AppHelper.callAfter(action, None)
            })
            self._hotkeys.daemon = True
            self._hotkeys.start()
        except Exception:
            # needs Input Monitoring permission; the menu item still works
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
            ok="Ask", cancel="Cancel", dimensions=(380, 48),
        )
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

    # -- briefing ------------------------------------------------------------

    def _briefing_clicked(self, _sender) -> None:
        threading.Thread(target=self._show_briefing, daemon=True).start()

    def _startup_briefing(self) -> None:
        # alert, not a notification: NSUserNotification from our launcher-style
        # process "succeeds" without visibly delivering (no registered bundle).
        # Revisit notifications if the app ever becomes a real py2app freeze.
        self._show_briefing()

    def _show_briefing(self) -> None:
        """Worker thread: compose (network + EventKit) then show the alert."""
        from src.briefing import briefing
        try:
            text = briefing.compose(self.config)
        except Exception as exc:
            text = f"Briefing failed: {exc}"
        AppHelper.callAfter(self._show_alert, "Morning Briefing", text)

    @staticmethod
    def _float_front(ns_window) -> None:
        """Menu-bar apps can't force activation on modern macOS (alerts open
        BEHIND the current app, needing a Dock hunt to dismiss) — a floating
        window level + orderFrontRegardless shows them on top anyway."""
        ns_window.setLevel_(NSFloatingWindowLevel)
        ns_window.orderFrontRegardless()

    def _show_alert(self, title: str, body: str) -> None:
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        alert = NSAlert.alloc().init()
        alert.setMessageText_(title)
        alert.setInformativeText_(body)
        alert.addButtonWithTitle_("OK")
        self._float_front(alert.window())
        alert.runModal()

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
            self.status_item.title = f"Listening… ({self.hotkey_combo} to stop)"
            if self.voice_enabled:
                self.speak_item.title = "Stop listening"
            return

        audio = self.recorder.stop()
        self.busy = True
        self.title = TITLE_THINKING
        self.status_item.title = "Transcribing…"
        if self.voice_enabled:
            self.speak_item.title = "Speak"
        threading.Thread(target=self._transcribe_and_respond, args=(audio,),
                         daemon=True).start()

    def _transcribe_and_respond(self, audio) -> None:
        """Worker thread: whisper → dispatcher → reply alert."""
        try:
            text = self.transcriber.transcribe(audio)
        except Exception as exc:
            AppHelper.callAfter(self._show_reply, "Voice", f"Transcription failed: {exc}")
            return
        if not text:
            AppHelper.callAfter(self._voice_heard_nothing)
            return
        AppHelper.callAfter(self._set_status, TITLE_THINKING, f"“{text}”")
        self._respond(text, spoken=True)

    def _voice_heard_nothing(self) -> None:
        self.busy = False
        self.title = TITLE_READY
        self.status_item.title = "Didn't catch that — try again"

    # -- inference (worker thread) ------------------------------------------

    def _respond(self, message: str, spoken: bool = False) -> None:
        try:
            result = self.dispatcher.respond(message)
            body = result.response
            if result.tool is not None:
                body += f"\n\n[{result.tool}] {result.tool_result}"
            if spoken and self.voice_replies:
                # you talked to it — it talks back (reply only, not the tool line)
                from src.voice.voice import speak
                speak(result.response)
        except Exception as exc:
            body = f"Error: {exc}"
        AppHelper.callAfter(self._show_reply, message, body)

    def _show_reply(self, message: str, body: str) -> None:
        self.busy = False
        self.title = TITLE_READY
        self.status_item.title = getattr(self, "_ready_status", "Ready")
        self._show_alert(message, body)


def main() -> None:
    DesktopHelperMenuBar().run()


if __name__ == "__main__":
    main()
