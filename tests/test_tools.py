from datetime import datetime

from src.calendar_integration.events import format_events, format_time
from src.launcher.launcher import match_app
from src.news.news import parse_titles
from src.reminders.reminders import format_reminders
from src.stocks.stocks import extract_symbols
from src.weather.weather import condition_text

APPS = [
    {"name": "Spotify", "path": "/Applications/Spotify.app"},
    {"name": "VS Code", "path": "/Applications/Visual Studio Code.app"},
]


def test_launcher_matches_configured_app():
    assert match_app("Open Spotify for me", APPS)["name"] == "Spotify"
    assert match_app("fire up vs code", APPS)["name"] == "VS Code"


def test_launcher_rejects_unknown_app():
    assert match_app("Open Photoshop", APPS) is None


def test_launcher_prefers_longest_name():
    apps = APPS + [{"name": "Code", "path": "/Applications/Code.app"}]
    assert match_app("open vs code", apps)["name"] == "VS Code"


def test_news_parses_rss_titles():
    xml = """<rss><channel>
        <title>Feed</title>
        <item><title>First story</title></item>
        <item><title>Second story</title></item>
    </channel></rss>"""
    assert parse_titles(xml) == ["First story", "Second story"]


def test_news_parses_atom_titles():
    xml = """<feed xmlns="http://www.w3.org/2005/Atom">
        <title>Feed</title>
        <entry><title>Atom story</title></entry>
    </feed>"""
    assert parse_titles(xml) == ["Atom story"]


def test_stocks_extracts_symbols_in_order():
    assert extract_symbols("How's NVDA doing? Also check AAPL.") == ["NVDA", "AAPL"]


def test_stocks_ignores_common_uppercase_words():
    assert extract_symbols("I think AI is OK") == []


def test_stocks_no_symbols_means_empty():
    assert extract_symbols("how are my stocks today?") == []


def test_weather_condition_mapping():
    assert condition_text(0) == "sunny"
    assert condition_text(53) == "light rain"
    assert condition_text(999) == "unsettled"


def test_calendar_time_format_matches_training_shapes():
    assert format_time(datetime(2026, 7, 17, 9, 0)) == "9am"
    assert format_time(datetime(2026, 7, 17, 15, 30)) == "3:30pm"


def test_calendar_event_formatting():
    events = [("Standup", datetime(2026, 7, 17, 9, 0)), ("Conference", None)]
    assert format_events(events) == "Standup at 9am, Conference (all day)"
    assert format_events([]) == "No events today"


def test_reminders_formatting():
    assert format_reminders(["Call dentist", "Pay rent"]) == "Call dentist, Pay rent"
    assert format_reminders([]) == "No reminders set"


def test_build_handlers_respects_feature_flags():
    from src.assistant.tools import build_handlers

    config = {
        "features": {"weather": False, "screen_capture": False},
        "allowed_apps": [],
        "stocks": {"watchlist": []},
        "news": {"rss_feeds": []},
        "weather": {"location": "Nowhere"},
    }
    handlers = build_handlers(config)
    assert set(handlers) == {
        "spotify", "calendar", "screen", "reminders", "notes",
        "launcher", "weather", "news", "stocks",
    }
    assert handlers["weather"]("any") == "weather is disabled in config"
    assert handlers["screen"]("any") == "screen is disabled in config"
    assert handlers["launcher"]("open something") == "No matching app in the allowed apps list"
