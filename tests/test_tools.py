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
    # screenshot phrasing is outside the training grammar → keyword net
    assert fallback("Tell me about the screenshot I just took") == "screen"
    assert fallback("what's in my last screen shot?") == "screen"
    # reminder create phrasing + casing/list read variants the model misses
    assert fallback("Remind me to go for a run in an hour") == "reminders"
    assert fallback("check my reminders") == "reminders"
    assert fallback("what's on my Work list?") == "reminders"
    assert fallback("remind me what the weather is") == "weather"  # weather wins
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
                      "launcher", "files", "memory", "chat"}
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


def test_screen_handler_screenshot_intent(monkeypatch, tmp_path):
    from src.assistant import tools
    from src.assistant.tools import build_handlers

    config = {
        "features": {},
        "allowed_apps": [],
        "stocks": {"watchlist": []},
        "news": {"rss_feeds": []},
        "weather": {"location": "Nowhere"},
    }
    shot = tmp_path / "Screenshot test.png"
    monkeypatch.setattr(tools.capture, "latest_screenshot", lambda: shot)
    monkeypatch.setattr(tools.capture, "describe_image",
                        lambda p: f"Text visible in the screenshot {p.name}: hi")
    monkeypatch.setattr(tools.capture, "describe", lambda: "Live screen")

    handlers = build_handlers(config)
    # screenshot phrasing → the user's own file, not a live capture
    assert "Screenshot test.png" in handlers["screen"]("describe the screenshot I took")
    # live phrasing → live capture
    assert handlers["screen"]("what's on my screen?") == "Live screen"
    # no screenshots on disk → honest fallback message
    monkeypatch.setattr(tools.capture, "latest_screenshot", lambda: None)
    assert "No screenshots found" in handlers["screen"]("my latest screenshot?")


def test_extract_file_query():
    from src.assistant.tools import extract_file_query

    # possessive locate phrases + explicit file nouns extract the term
    assert extract_file_query("Find my resume.") == "resume"
    assert extract_file_query("Where is my budget spreadsheet") == "budget"
    assert extract_file_query("Show me the files named tax") == "tax"
    assert extract_file_query("locate report.pdf") == "report.pdf"
    assert extract_file_query("find my tax documents") == "tax"
    # an extension alone signals a file request (no possessive needed)
    assert extract_file_query("find invoice.pdf") == "invoice.pdf"
    # not a file request → None (must not steal generic questions)
    assert extract_file_query("Where is the coffee shop?") is None
    assert extract_file_query("find a good time to meet") is None
    assert extract_file_query("What's the weather?") is None
    assert extract_file_query("find my files") is None  # bare noun, no term


def test_extract_file_term():
    from src.assistant.tools import extract_file_term

    # the phrasings that failed live — third-person, "most recent version of"
    assert extract_file_term("Find Kai's resume") == "resume"
    assert extract_file_term("Find the most recent version of kai's resume") == "resume"
    assert extract_file_term("Find my resume") == "resume"
    assert extract_file_term("locate report.pdf") == "report.pdf"
    assert extract_file_term("find the report") == "report"
    assert extract_file_term("pull up the budget spreadsheet") == "budget"
    assert extract_file_term("where is my invoice located") == "invoice"


def test_time_based_file_queries():
    from src.assistant.tools import (build_fallback_router, extract_recent_days,
                                     is_recent_files_query)

    # recency window parsing
    assert extract_recent_days("files I worked on last week") == 7
    assert extract_recent_days("what did I work on a couple weeks ago") == 14
    assert extract_recent_days("files from 3 days ago") == 3
    assert extract_recent_days("2 weeks ago") == 14
    assert extract_recent_days("recent documents") == 7
    assert extract_recent_days("find my resume") is None
    # routing needs a file/work signal so calendar time queries aren't stolen
    assert is_recent_files_query("files I worked on last week") is True
    assert is_recent_files_query("what's on my calendar this week") is False

    fb = build_fallback_router()
    # "what did I work on…" contains "what did i" but must beat the memory rule
    assert fb("what did I work on a couple weeks ago") == "files"
    assert fb("what did I ask you earlier") == "memory"
    assert fb("what's on my calendar this week") is None


def test_pre_router_forces_unambiguous_file_queries():
    from src.assistant.tools import build_pre_router

    pre = build_pre_router()
    # explicit file noun or a real extension → force files (override model)
    assert pre("where is my resume file") == "files"
    assert pre("find my budget document") == "files"
    assert pre("locate invoice.pdf") == "files"
    # softer phrasing (no noun/extension) → let the model try; fallback nets it
    assert pre("find my resume") is None
    assert pre("where is my budget") is None
    # not a file query at all → None
    assert pre("what's the weather in Tokyo") is None
    assert pre("remind me to call mom") is None


def test_fallback_router_routes_file_search():
    from src.assistant.tools import build_fallback_router

    fallback = build_fallback_router()
    assert fallback("Find my resume") == "files"
    assert fallback("where is my budget file") == "files"
    # generic questions still fall through to the model, not files
    assert fallback("Where is the nearest coffee shop?") is None


def test_files_handler_searches_and_reports(monkeypatch):
    from src.assistant import tools
    from src.assistant.tools import build_handlers

    config = {"features": {}, "allowed_apps": [], "stocks": {"watchlist": []},
              "news": {"rss_feeds": []}, "weather": {"location": "X"},
              "files": {"max_results": 2}}
    captured = {}
    monkeypatch.setattr(tools.files, "find",
                        lambda q, n: captured.update(q=q, n=n) or f"Found: {q}")
    handlers = build_handlers(config)

    assert handlers["files"]("Find my resume") == "Found: resume"
    assert captured == {"q": "resume", "n": 2}  # term extracted, max_results passed
    # no extractable term → asks for one, never shells out
    assert "part of the file name" in handlers["files"]("find my files")


def test_extract_reminder():
    from src.assistant.tools import (extract_reminder, extract_reminder_list,
                                     strip_due_text)

    # create intent → (reminder text, list name or None); read → None
    assert extract_reminder("Remind me to call the dentist") == ("call the dentist", None)
    assert extract_reminder("set a reminder to buy milk") == ("buy milk", None)
    assert extract_reminder("What are my reminders?") is None

    # list targeting — matched against the user's actual lists at runtime
    assert extract_reminder("remind me to buy milk on my Groceries list") \
        == ("buy milk", "Groceries")
    assert extract_reminder("add a reminder to finish the report to my Work list") \
        == ("finish the report", "Work")
    # list name on a READ request too
    assert extract_reminder_list("what's on my Work list?") == "Work"
    assert extract_reminder_list("check my reminders") is None

    # the due-date phrase is stripped from the title (components come from
    # NSDataDetector separately), including any dangling connector word
    assert strip_due_text("call the dentist tomorrow at 3pm", "tomorrow at 3pm") \
        == "call the dentist"
    assert strip_due_text("buy milk", None) == "buy milk"


def test_reminders_reply_passes_through_create_confirmation():
    from src.assistant.tools import build_verbatim

    reply = build_verbatim()["reminders"]
    assert reply("Reminder added: buy milk") == "Reminder added: buy milk"
    assert reply("Return library books") == "Your reminders: Return library books."
    assert reply("No reminders set") == "You don't have any reminders set."


def test_reprompt_builder_covers_screen():
    from src.assistant.tools import build_reprompts

    reprompts = build_reprompts()
    prompt = reprompts["screen"]("Code in front. On screen: def main():")
    assert "Code in front. On screen: def main():" in prompt
    assert "what I'm looking at" in prompt


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
        "launcher", "weather", "news", "stocks", "files", "memory", "chat",
    }
    assert handlers["weather"]("any") == "weather is disabled in config"
    assert handlers["screen"]("any") == "screen is disabled in config"
    assert handlers["launcher"]("open something") == "No matching app in the allowed apps list"
