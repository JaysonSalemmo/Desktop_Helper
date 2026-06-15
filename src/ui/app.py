from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog
from textual.containers import Vertical
from src.config import settings


class DesktopHelperApp(App):
    CSS = """
    Vertical {
        height: 1fr;
    }
    RichLog {
        height: 1fr;
        border: solid $primary;
        padding: 1 2;
    }
    Input {
        dock: bottom;
        margin: 1 0;
    }
    """

    BINDINGS = [("ctrl+q", "quit", "Quit")]

    def __init__(self):
        super().__init__()
        self.config = settings.load()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield RichLog(id="chat_log", wrap=True, markup=True)
        yield Input(placeholder="Ask me anything...", id="chat_input")
        yield Footer()

    def on_mount(self) -> None:
        name = self.config["user"]["name"]
        self.title = "Desktop Helper"
        log = self.query_one("#chat_log", RichLog)
        log.write(f"[bold green]Hello {name}! I'm your desktop assistant.[/bold green]")
        log.write("[dim]Type a message and press Enter. Press Ctrl+Q to quit.[/dim]")
        self.query_one("#chat_input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        message = event.value.strip()
        if not message:
            return

        log = self.query_one("#chat_log", RichLog)
        input_widget = self.query_one("#chat_input", Input)

        log.write(f"[bold cyan]You:[/bold cyan] {message}")
        input_widget.clear()

        response = self._handle_message(message)
        log.write(f"[bold green]Assistant:[/bold green] {response}")

    def _handle_message(self, message: str) -> str:
        lower = message.lower()

        if "open" in lower:
            app_name = message.lower().replace("open", "").strip()
            return f"[dim](App launcher not yet wired up — looking for '{app_name}')[/dim]"

        if "weather" in lower:
            return "[dim](Weather module not yet wired up)[/dim]"

        if "news" in lower:
            return "[dim](News module not yet wired up)[/dim]"

        if "stock" in lower or "market" in lower:
            return "[dim](Stocks module not yet wired up)[/dim]"

        if "calendar" in lower or "events" in lower:
            return "[dim](Calendar module not yet wired up)[/dim]"

        if "reminder" in lower:
            return "[dim](Reminders module not yet wired up)[/dim]"

        if "note" in lower:
            return "[dim](Notes module not yet wired up)[/dim]"

        if "screenshot" in lower or "screen" in lower or "what's on" in lower:
            return "[dim](Screen capture not yet wired — Phase 4)[/dim]"

        if "spotify" in lower or "play" in lower or "pause" in lower or "skip" in lower or "music" in lower:
            return "[dim](Spotify module not yet wired up)[/dim]"

        return "[dim](Local model not yet loaded — see PLAN.md Phase 1)[/dim]"
