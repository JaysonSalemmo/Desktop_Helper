"""
Tool handlers for the dispatcher.

Each handler takes the user's message and returns a result string that gets
injected as `[RESULT]...[/RESULT]`. The model has already chosen the tool; the
handler parses the message to pick the specific action within that tool.

Result strings deliberately match the shapes in the training data
(model/data/tool_calls.py) — the model wraps familiar formats far more
faithfully than novel ones.

`build_handlers(config)` is the entry point: it wires every tool to its real
backend and respects the feature flags in config.json (a disabled tool still
responds, but says it's disabled).
"""
from collections.abc import Callable

from src.launcher import launcher
from src.news import news
from src.notes import notes
from src.screen_capture import capture
from src.spotify import spotify
from src.stocks import stocks
from src.weather import weather

Handler = Callable[[str], str]


def spotify_handler(message: str) -> str:
    m = message.lower()
    if "pause" in m or "stop" in m:
        spotify.pause()
        return "Paused"
    if "skip" in m or "next" in m:
        spotify.next_track()
        return f"Now playing: {spotify.current_track()}"
    if "previous" in m or "go back" in m or "last song" in m:
        spotify.previous_track()
        return f"Now playing: {spotify.current_track()}"
    if "volume up" in m or "turn up" in m or "louder" in m:
        spotify.set_volume(85)
        return "Volume set to 85%"
    if "volume down" in m or "turn down" in m or "quieter" in m or "lower" in m:
        spotify.set_volume(35)
        return "Volume set to 35%"
    if "resume" in m or "play" in m:
        spotify.play()
        return f"Now playing: {spotify.current_track()}"
    # default intent: report what's currently playing
    return spotify.current_track()


_NOTE_ADD_PREFIXES = ["note that ", "note down ", "write down ", "add a note "]


def notes_handler(message: str) -> str:
    lower = message.lower()
    for prefix in _NOTE_ADD_PREFIXES:
        if prefix in lower:
            text = message[lower.index(prefix) + len(prefix):].strip(" :.")
            if text:
                notes.add(text)
                return f"Note saved: {text}"
    todays = notes.get_today()
    if not todays:
        return "No notes for today"
    return "; ".join(n["text"] for n in todays)


# keep injected results short on permission failure — the model wraps a brief
# result far better than a paragraph of System Settings instructions (those go
# to the log via the raised message when debugging)
def _calendar_handler(message: str) -> str:
    from src.calendar_integration import events  # EventKit import deferred
    from src.eventkit.store import AccessDenied
    try:
        return events.today()
    except AccessDenied:
        return "Calendar access not granted"


def _reminders_handler(message: str) -> str:
    from src.reminders import reminders  # EventKit import deferred
    from src.eventkit.store import AccessDenied
    try:
        return reminders.incomplete_summary()
    except AccessDenied:
        return "Reminders access not granted"


def _disabled(tool: str) -> Handler:
    def handler(message: str) -> str:
        return f"{tool} is disabled in config"
    return handler


def build_handlers(config: dict) -> dict[str, Handler]:
    """Handler registry keyed by tool name (the value from is_tool_call)."""

    def launcher_handler(message: str) -> str:
        app = launcher.match_app(message, config.get("allowed_apps", []))
        if app is None:
            return "No matching app in the allowed apps list"
        return launcher.launch(app)

    def weather_handler(message: str) -> str:
        return weather.current(config["weather"]["location"])

    def news_handler(message: str) -> str:
        found = news.headlines(
            config["news"]["rss_feeds"], config["news"].get("max_headlines", 5)
        )
        return "; ".join(found) if found else "No headlines available right now"

    def stocks_handler(message: str) -> str:
        return stocks.quotes(message, config["stocks"]["watchlist"])

    def screen_handler(message: str) -> str:
        return capture.describe()

    handlers = {
        "spotify": spotify_handler,
        "calendar": _calendar_handler,
        "screen": screen_handler,
        "reminders": _reminders_handler,
        "notes": notes_handler,
        "launcher": launcher_handler,
        "weather": weather_handler,
        "news": news_handler,
        "stocks": stocks_handler,
    }

    # feature flags in config.json (launcher has no flag — always on)
    flags = config.get("features", {})
    flag_names = {tool: tool for tool in handlers} | {"screen": "screen_capture"}
    for tool, flag in flag_names.items():
        if not flags.get(flag, True):
            handlers[tool] = _disabled(tool)
    return handlers
