from pathlib import Path

from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Header, Footer, Input, RichLog

from model.device import get_device
from model.generate import load_model
from model.tokenizer import DesktopHelperTokenizer
from src.assistant.dispatcher import ToolDispatcher
from src.assistant.tools import HANDLERS
from src.config import settings

TOKENIZER_PATH = "model/tokenizer.json"


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
        self.dispatcher: ToolDispatcher | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield RichLog(id="chat_log", wrap=True, markup=True)
        yield Input(placeholder="Loading model...", id="chat_input", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        name = self.config["user"]["name"]
        self.title = "Desktop Helper"
        log = self.query_one("#chat_log", RichLog)
        log.write(f"[bold green]Hello {name}! I'm your desktop assistant.[/bold green]")
        log.write("[dim]Loading model — one moment...[/dim]")
        self._load_model()

    @work(thread=True)
    def _load_model(self) -> None:
        log = self.query_one("#chat_log", RichLog)
        checkpoint = Path(self.config["model"]["checkpoint"])
        if not checkpoint.exists():
            self.call_from_thread(
                log.write,
                f"[bold red]Model checkpoint not found:[/bold red] {checkpoint}\n"
                "[dim]Download it from Google Drive into model/checkpoints/ "
                "(weights are gitignored), then restart.[/dim]",
            )
            return
        try:
            device = get_device(require_cuda=False)
            tokenizer = DesktopHelperTokenizer.load(TOKENIZER_PATH)
            model = load_model(str(checkpoint), device)
            dispatcher = ToolDispatcher(model, tokenizer, HANDLERS, device)
        except Exception as exc:
            self.call_from_thread(
                log.write, f"[bold red]Failed to load model:[/bold red] {escape(str(exc))}"
            )
            return
        self.dispatcher = dispatcher
        self.call_from_thread(self._on_model_ready, device)

    def _on_model_ready(self, device) -> None:
        log = self.query_one("#chat_log", RichLog)
        log.write(f"[dim]Model ready ({device.type}). Type a message and press Enter. Ctrl+Q to quit.[/dim]")
        input_widget = self.query_one("#chat_input", Input)
        input_widget.disabled = False
        input_widget.placeholder = "Ask me anything..."
        input_widget.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        message = event.value.strip()
        if not message or self.dispatcher is None:
            return

        log = self.query_one("#chat_log", RichLog)
        input_widget = self.query_one("#chat_input", Input)

        log.write(f"[bold cyan]You:[/bold cyan] {escape(message)}")
        input_widget.clear()
        # one turn at a time — inference holds the model, so queueing would only confuse
        input_widget.disabled = True
        input_widget.placeholder = "Thinking..."
        self._respond(message)

    @work(thread=True)
    def _respond(self, message: str) -> None:
        log = self.query_one("#chat_log", RichLog)
        try:
            result = self.dispatcher.respond(message)
        except Exception as exc:
            self.call_from_thread(
                log.write, f"[bold red]Error:[/bold red] {escape(str(exc))}"
            )
            self.call_from_thread(self._reenable_input)
            return

        lines = f"[bold green]Assistant:[/bold green] {escape(result.response)}"
        if result.tool is not None:
            lines += f"\n[dim]tool: {result.tool} → {escape(result.tool_result or '')}[/dim]"
        self.call_from_thread(log.write, lines)
        self.call_from_thread(self._reenable_input)

    def _reenable_input(self) -> None:
        input_widget = self.query_one("#chat_input", Input)
        input_widget.disabled = False
        input_widget.placeholder = "Ask me anything..."
        input_widget.focus()
