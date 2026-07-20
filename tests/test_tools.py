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


def test_nws_condition_mapping():
    from src.weather.weather import nws_condition

    assert nws_condition("Thunderstorm") == "stormy"
    assert nws_condition("Thunderstorms and Rain") == "stormy"
    assert nws_condition("Fog/Mist") == "foggy"
    assert nws_condition("Partly Cloudy") == "partly cloudy"
    assert nws_condition("Light Rain") == "light rain"
    assert nws_condition("Fair") == "sunny"
    assert nws_condition("Smoke") == "smoke"  # unmapped → lowercase verbatim


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


def test_spotify_status_question_does_not_start_playback(monkeypatch):
    # "what's playing?" contains "play" — it must read status, never call play()
    from src.assistant import tools
    calls = []
    monkeypatch.setattr(tools.spotify, "is_running", lambda: True)
    monkeypatch.setattr(tools.spotify, "current_track", lambda: "Song by Artist")
    monkeypatch.setattr(tools.spotify, "play", lambda: calls.append("play"))

    for q in ["What's playing?", "What song is this?", "Who sings this?",
              "What's the current song?"]:
        assert tools.spotify_handler(q) == "Song by Artist"
    assert calls == []


def test_spotify_closed_app_only_launches_for_play(monkeypatch):
    from src.assistant import tools
    monkeypatch.setattr(tools.spotify, "is_running", lambda: False)
    monkeypatch.setattr(tools.spotify, "play", lambda: None)
    monkeypatch.setattr(tools.spotify, "current_track", lambda: "Song by Artist")

    assert tools.spotify_handler("Pause the music.") == "Spotify isn't running"
    assert tools.spotify_handler("Skip this song.") == "Spotify isn't running"
    assert "Song by Artist" in tools.spotify_handler("Play some music.")


def test_extract_play_query():
    from src.assistant.tools import extract_play_query

    assert extract_play_query("Play Bohemian Rhapsody by Queen.") == "Bohemian Rhapsody by Queen"
    assert extract_play_query("play the song Levitating on Spotify") == "Levitating"
    assert extract_play_query("Can you play Anti-Hero?") == "Anti-Hero"
    # generic requests mean resume, not search
    assert extract_play_query("Play some music.") is None
    assert extract_play_query("Play something") is None
    assert extract_play_query("Resume the music.") is None


def test_spotify_specific_song_searches_and_plays(monkeypatch):
    from src.assistant import tools

    played = []
    monkeypatch.setattr(tools.spotify, "is_running", lambda: True)
    monkeypatch.setattr(tools.spotify, "search_track",
                        lambda q, cid, sec: ("spotify:track:abc", "Bohemian Rhapsody by Queen"))
    monkeypatch.setattr(tools.spotify, "play_track", lambda uri: played.append(uri))

    creds = {"client_id": "id", "client_secret": "sec"}
    reply = tools.spotify_handler("Play Bohemian Rhapsody", credentials=creds)
    assert reply == "Now playing: Bohemian Rhapsody by Queen"
    assert played == ["spotify:track:abc"]

    # "Play the next track." must stay a skip, not a search
    monkeypatch.setattr(tools.spotify, "next_track", lambda: None)
    monkeypatch.setattr(tools.spotify, "current_track", lambda: "Song by Artist")
    assert "Song by Artist" in tools.spotify_handler("Play the next track.", credentials=creds)
    assert played == ["spotify:track:abc"]  # unchanged — no second play


def test_spotify_specific_song_without_creds_degrades(monkeypatch):
    from src.assistant import tools
    monkeypatch.setattr(tools.spotify, "is_running", lambda: True)
    reply = tools.spotify_handler("Play Bohemian Rhapsody", credentials=None)
    assert "API keys" in reply


def test_extract_artist_request():
    from src.assistant.tools import extract_artist_request

    assert extract_artist_request("a song by Drake") == "Drake"
    assert extract_artist_request("something by Bruno Mars") == "Bruno Mars"
    assert extract_artist_request("music by The Weeknd") == "The Weeknd"
    # "another" variants — unmatched, these became literal searches and hit
    # metadata ambushes (Weird Al's Bruno Mars parody)
    assert extract_artist_request("another song by Drake") == "Drake"
    assert extract_artist_request("Another song by Bruno Mars.") == "Bruno Mars"
    assert extract_artist_request("play another one by Queen") == "Queen"
    # a specific track must NOT be treated as an artist request
    assert extract_artist_request("Passionfruit by Drake") is None
    assert extract_artist_request("Bohemian Rhapsody by Queen") is None


def test_vague_artist_request_uses_random_pick(monkeypatch):
    from src.assistant import tools

    calls = []
    monkeypatch.setattr(tools.spotify, "is_running", lambda: True)
    monkeypatch.setattr(tools.spotify, "search_artist_track",
                        lambda a, cid, sec: calls.append(a) or ("spotify:track:x", "Some Song by Drake"))
    monkeypatch.setattr(tools.spotify, "search_track",
                        lambda *a: (_ for _ in ()).throw(AssertionError("wrong search")))
    monkeypatch.setattr(tools.spotify, "play_track", lambda uri: None)

    creds = {"client_id": "id", "client_secret": "sec"}
    reply = tools.spotify_handler("Play a song by Drake", credentials=creds)
    assert reply == "Now playing: Some Song by Drake"
    assert calls == ["Drake"]

    # playless phrasing still works ("Another song by X")
    reply = tools.spotify_handler("Another song by Bruno Mars.", credentials=creds)
    assert reply == "Now playing: Some Song by Drake"
    assert calls == ["Drake", "Bruno Mars"]


def test_extract_location():
    from src.assistant.tools import extract_location

    assert extract_location("How is the weather today in New York?") == "New York"
    assert extract_location("What's the weather in Tokyo") == "Tokyo"
    assert extract_location("forecast for San Francisco?") == "San Francisco"
    assert extract_location("Is it cold out in the morning?") is None
    assert extract_location("What's the weather like?") is None


def test_fallback_router_covers_weather_and_spotify():
    from src.assistant.tools import build_fallback_router

    fallback = build_fallback_router()
    assert fallback("How is the weather today in New York?") == "weather"
    assert fallback("What's the temperature outside right now?") == "weather"
    assert fallback("Play Passionfruit by Drake on Spotify") == "spotify"
    # unrouted chat → the model, which holds a real conversation now
    assert fallback("Hello there!") is None
    assert fallback("Tell me a joke") is None
    # …but capability questions stay canned: the model invents features
    assert fallback("What can you do?") == "chat"
    assert fallback("help") == "chat"
    # model_chat=false opts all unrouted chat back into canned replies
    canned = build_fallback_router({"features": {"model_chat": False}})
    assert canned("Hello there!") == "chat"
    assert canned("What can you do?") == "chat"


def test_chat_handler_canned_replies():
    from src.assistant.tools import make_chat_handler

    chat = make_chat_handler({"user": {"name": "Kai"}})
    assert chat("Hello Desktop Helper").startswith("Hey Kai!")
    assert chat("good morning!").startswith("Hey Kai!")
    assert chat("thanks!") == "Anytime!"
    assert "calendar" in chat("What can you do?")
    assert chat("Tell me a joke").startswith("I'm not sure how to help")


def test_verbatim_replies_template_and_passthrough():
    from src.assistant.tools import build_verbatim

    v = build_verbatim()
    assert set(v) == {"calendar", "reminders", "notes", "spotify", "stocks",
                      "launcher", "memory", "chat"}
    assert v["launcher"]("VS Code launched") == "VS Code is open."
    assert v["launcher"]("Now playing: X by Y") == "Now playing: X by Y"
    assert v["spotify"]("Love All (with JAY-Z) by Drake") == \
        "Current track: Love All (with JAY-Z) by Drake."
    assert v["spotify"]("Paused") == "Music paused."
    assert v["spotify"]("Now playing: X by Y") == "Now playing: X by Y"
    assert v["stocks"]("AAPL: $333.74 (-0.3%)") == "Your stocks: AAPL: $333.74 (-0.3%)."
    assert v["calendar"]("Standup at 9am") == "Today: Standup at 9am."
    assert v["calendar"]("No events today") == "Your calendar is clear today."
    assert v["calendar"]("Calendar access not granted") == "Calendar access not granted"
    assert v["reminders"]("Call dentist, Pay rent") == "Your reminders: Call dentist, Pay rent."
    assert v["notes"]("Note saved: buy milk") == "Note saved: buy milk"
    assert v["notes"]("Grocery list: milk, eggs") == "From your notes: Grocery list: milk, eggs."


def test_launcher_delegates_play_requests_to_spotify(monkeypatch):
    # the model routes "Play {song}" to launcher (trained launcher prompts are
    # "Open/Start {Name}") — the handler must hand it to spotify
    from src.assistant import tools

    played = []
    monkeypatch.setattr(tools.spotify, "is_running", lambda: True)
    monkeypatch.setattr(tools.spotify, "search_track",
                        lambda q, cid, sec: ("spotify:track:abc", f"{q} by Somebody"))
    monkeypatch.setattr(tools.spotify, "play_track", lambda uri: played.append(uri))

    config = {"allowed_apps": [{"name": "Spotify", "path": "/Applications/Spotify.app"}],
              "spotify": {"client_id": "id", "client_secret": "sec"},
              "stocks": {"watchlist": []}, "news": {"rss_feeds": []},
              "weather": {"location": "X"}}
    handlers = tools.build_handlers(config)

    reply = handlers["launcher"]("Play Bohemian Rhapsody by Queen.")
    assert reply == "Now playing: Bohemian Rhapsody by Queen by Somebody"
    assert played == ["spotify:track:abc"]

    # "…on Spotify" contains the app name — must still play the song, not
    # launch the app (found live: it opened Spotify and stopped there)
    reply = handlers["launcher"]("Play Bohemian Rhapsody on Spotify.")
    assert reply == "Now playing: Bohemian Rhapsody by Somebody"

    # a real launch request still launches, not searches
    assert tools.launcher.match_app("Open Spotify", config["allowed_apps"]) is not None


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
        "launcher", "weather", "news", "stocks", "memory", "chat",
    }
    assert handlers["weather"]("any") == "weather is disabled in config"
    assert handlers["screen"]("any") == "screen is disabled in config"
    assert handlers["launcher"]("open something") == "No matching app in the allowed apps list"
