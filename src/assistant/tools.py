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
import re
from collections.abc import Callable

from src.file_finder import files
from src.launcher import launcher
from src.news import news
from src.notes import notes
from src.screen_capture import capture
from src.spotify import spotify
from src.stocks import stocks
from src.weather import weather

Handler = Callable[[str], str]


# "play …" requests with only these after the verb mean "resume", not a search
_GENERIC_PLAY = {
    "", "music", "some music", "the music", "something", "a song", "some songs",
    "songs", "anything", "spotify", "it",
}


def extract_play_query(message: str) -> str | None:
    """The specific thing to play, or None for a bare resume.

    "Play Bohemian Rhapsody by Queen." → "Bohemian Rhapsody by Queen"
    "Play some music."                 → None
    """
    lower = message.lower()
    idx = lower.find("play ")
    if idx == -1:
        return None
    query = message[idx + len("play "):].strip().rstrip(".!?")
    for prefix in ("the song ", "the track ", "song ", "track ", "me "):
        if query.lower().startswith(prefix):
            query = query[len(prefix):]
    if query.lower().endswith(" on spotify"):
        query = query[: -len(" on spotify")].rstrip()
    if query.lower() in _GENERIC_PLAY:
        return None
    return query or None


# file search has no model tool token (like memory) — the fallback router is
# its only path, so extract_file_query doubles as the intent gate AND the
# term extractor. Intent is kept HIGH-PRECISION: an explicit file noun, or a
# POSSESSIVE locate phrase ("find my", "where's my", "locate"). Bare "find"/
# "where is" are too broad ("where is the coffee shop" is not a file search)
# and would steal generic questions from the model.
_FILE_NOUN_RE = re.compile(r"\b(?:files?|folders?|documents?)\b", re.I)
_FILE_INTENT_RE = re.compile(
    r"\b(?:find my|locate|where(?:'s| is| are) my)\b", re.I)
# a token like "invoice.pdf" is an unambiguous file reference on its own —
# curated to user-document types (not code extensions, which are dev noise)
_FILE_EXT_RE = re.compile(
    r"\b[\w-]+\.(?:pdf|docx?|xlsx?|pptx?|txt|md|csv|rtf|pages|numbers|key|"
    r"zip|png|jpe?g|gif|heic|mov|mp4|mp3|wav|json|ya?ml)\b", re.I)
# leading scaffolding stripped to reach the search term
_FILE_STRIP_RE = re.compile(
    r"^(?:can you |could you |please )?"
    r"(?:find|locate|look for|search for|show me|where(?:'s| is| are))\s+"
    r"(?:my |the |a |any |all )?",
    re.I,
)
_FILE_NOUN_PREFIX_RE = re.compile(
    r"^(?:files?|folders?|documents?)\s+(?:named |called |about |for |with )", re.I)
# trailing type words ("budget spreadsheet" → search "budget"): the type isn't
# part of the filename, and mdfind -name matches the name, not the kind
_FILE_TYPE_TAIL_RE = re.compile(
    r"\s+(?:spreadsheets?|documents?|files?|folders?|photos?|pictures?|docs?)$",
    re.I)
_FILE_LEADING_MY_RE = re.compile(r"^my\s+", re.I)
# a query that reduces to a bare type noun ("find my files") carries no actual
# search term — nothing to hand mdfind
_FILE_BARE_NOUNS = {"file", "files", "folder", "folders", "document",
                    "documents", "doc", "docs", "photo", "photos",
                    "picture", "pictures"}


def extract_file_query(message: str) -> str | None:
    """The filename term to search for, or None when it isn't a file request.

    "Find my resume."               → "resume"
    "Where is my budget spreadsheet" → "budget"
    "Show me the files named tax"    → "tax"
    "Where is the coffee shop?"      → None (no possessive / file noun)
    "Find my files"                  → None (no actual search term)
    """
    text = message.strip().rstrip(".!?")
    lower = text.lower()
    if not (_FILE_NOUN_RE.search(lower) or _FILE_INTENT_RE.search(lower)
            or _FILE_EXT_RE.search(text)):
        return None
    query = _FILE_STRIP_RE.sub("", text).strip()
    query = _FILE_NOUN_PREFIX_RE.sub("", query).strip()
    query = _FILE_TYPE_TAIL_RE.sub("", query).strip()
    query = _FILE_LEADING_MY_RE.sub("", query).strip()
    if query.lower() in _FILE_BARE_NOUNS:
        return None
    return query or None


def spotify_handler(message: str, credentials: dict | None = None) -> str:
    m = message.lower()
    # status questions first — "what's playing?" contains "play" and must not
    # fall through to the play command (found live: it started playback)
    if "what" in m or "who" in m or "current" in m:
        return spotify.current_track()
    # only an explicit play intent may auto-launch Spotify; every other action
    # against a closed app would either lie ("Paused") or launch it by surprise
    if not spotify.is_running() and not ("play" in m or "resume" in m):
        return "Spotify isn't running"

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
    # specific song request? ordered after the transport controls so
    # "Play the next track." stays a skip, not a search for "the next track"
    query = extract_play_query(message)
    if query is None and extract_artist_request(message) is not None:
        query = message  # "Another song by Bruno Mars" — no "play", still a request
    if query is not None:
        if not credentials or not credentials.get("client_id"):
            return "Add Spotify API keys to config.json to play specific songs"
        artist = extract_artist_request(query)
        if artist is not None:
            found = spotify.search_artist_track(artist, credentials["client_id"],
                                                credentials["client_secret"])
        else:
            found = spotify.search_track(query, credentials["client_id"],
                                         credentials["client_secret"])
        if found is None:
            return f"No Spotify results for {query}"
        uri, display = found
        spotify.play_track(uri)
        return f"Now playing: {display}"
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


def build_handlers(config: dict, memory=None) -> dict[str, Handler]:
    """Handler registry keyed by tool name (the value from is_tool_call).

    `memory` (ChromaMemory) backs the "memory" pseudo-tool — no model token
    exists for it; it's reached only via the fallback router."""

    def launcher_handler(message: str) -> str:
        # play-queries win BEFORE app matching: "Play Bohemian Rhapsody on
        # Spotify" contains the app name "Spotify" and was matching/launching
        # the app instead of playing the song. The model routes "Play {Name}"
        # here (trained launcher prompts are "Open/Start {Name}"), so this
        # handler owns the disambiguation: a play request is a song.
        if extract_play_query(message) is not None:
            return spotify_handler(message, config.get("spotify"))
        app = launcher.match_app(message, config.get("allowed_apps", []))
        if app is None:
            return "No matching app in the allowed apps list"
        return launcher.launch(app)

    def weather_handler(message: str) -> str:
        location = extract_location(message) or config["weather"]["location"]
        return weather.current(location)

    def news_handler(message: str) -> str:
        found = news.headlines(
            config["news"]["rss_feeds"], config["news"].get("max_headlines", 5)
        )
        return "; ".join(found) if found else "No headlines available right now"

    def stocks_handler(message: str) -> str:
        return stocks.quotes(message, config["stocks"]["watchlist"])

    def screen_handler(message: str) -> str:
        # "the screenshot I just took" → the user's own newest Cmd+Shift
        # capture; anything else → a live look at the current screen
        m = message.lower()
        if "screenshot" in m or "screen shot" in m:
            latest = capture.latest_screenshot()
            if latest is not None:
                return capture.describe_image(latest)
            return "No screenshots found in your screenshots folder"
        return capture.describe()

    def files_handler(message: str) -> str:
        query = extract_file_query(message)
        if not query:
            return "Tell me part of the file name to search for"
        max_results = config.get("files", {}).get("max_results", 5)
        return files.find(query, max_results)

    def memory_handler(message: str) -> str:
        if memory is None:
            return "Memory isn't set up"
        hits = memory.search(message, n=2)
        if not hits:
            return "We haven't talked about anything like that yet"
        parts = []
        for m in hits:
            reply = m["response"]
            if len(reply) > 120:
                reply = reply[:117] + "…"
            parts.append(f'you asked “{m["message"]}” and I said “{reply}”')
        return "Earlier " + "; before that, ".join(parts) + "."

    handlers = {
        "spotify": lambda m: spotify_handler(m, config.get("spotify")),
        "calendar": _calendar_handler,
        "screen": screen_handler,
        "reminders": _reminders_handler,
        "notes": notes_handler,
        "launcher": launcher_handler,
        "weather": weather_handler,
        "news": news_handler,
        "stocks": stocks_handler,
        "files": files_handler,    # fallback-router only — no model token
        "memory": memory_handler,  # fallback-router only — no model token
        "chat": make_chat_handler(config),  # fallback-router only
    }

    # feature flags in config.json (launcher has no flag — always on)
    flags = config.get("features", {})
    flag_names = {tool: tool for tool in handlers} | {"screen": "screen_capture"}
    for tool, flag in flag_names.items():
        if not flags.get(flag, True):
            handlers[tool] = _disabled(tool)
    return handlers


# -- verbatim replies ---------------------------------------------------------
# Fact-heavy tools skip the model's paraphrase entirely: the dispatcher routes
# via the model, then templates the reply straight from the real result, so
# event titles / reminder items / note text come out word-perfect. Fallback
# strings ("access not granted", "disabled in config", errors, empty results)
# pass through untemplated since they're already sentences about the situation.

def _is_fallback(result: str) -> bool:
    return ("not granted" in result or "disabled in config" in result
            or "error:" in result or "not available" in result)


def _calendar_reply(result: str) -> str:
    if result == "No events today":
        return "Your calendar is clear today."
    if _is_fallback(result):
        return result
    return f"Today: {result}."


def _reminders_reply(result: str) -> str:
    if result == "No reminders set":
        return "You don't have any reminders set."
    if _is_fallback(result):
        return result
    return f"Your reminders: {result}."


def _notes_reply(result: str) -> str:
    if result == "No notes for today":
        return "You haven't written any notes today."
    if _is_fallback(result) or result.startswith("Note saved:"):
        return result
    return f"From your notes: {result}."


def _spotify_reply(result: str) -> str:
    if result == "Paused":
        return "Music paused."
    # handler results that are already sentences pass through
    if result.startswith(("Now playing:", "Volume set", "Spotify isn't", "Nothing playing")) \
            or _is_fallback(result):
        return result
    # bare "Track by Artist" status — neutral phrasing, we don't know play state
    return f"Current track: {result}."


def _stocks_reply(result: str) -> str:
    if _is_fallback(result):
        return result
    return f"Your stocks: {result}."


def _launcher_reply(result: str) -> str:
    if result.endswith(" launched"):
        return f"{result[: -len(' launched')]} is open."
    # fallbacks and spotify-delegated replies are already sentences
    return result


def build_pre_router(config: dict | None = None) -> "Callable[[str], str | None]":
    """High-precision routing that runs BEFORE the model.

    File search has no model tool token, and file-search phrasing collides hard
    with the trained tools — measured live, "where is my resume" argmaxes
    reminders and "find the file called report" argmaxes launcher, both
    confidently enough to clear the gate, so the fallback router (which only
    fires when the model emits NO call) never gets a turn. The cure is to
    intercept the *unambiguous* file queries — those naming a file noun
    ("...my budget document") or a real extension ("invoice.pdf") — and route
    them to files without asking the model at all. Softer phrasings ("find my
    resume", no noun/extension) stay best-effort via the fallback router.

    Returns the tool name to force, or None to let the model route normally."""
    def pre_route(message: str) -> str | None:
        if extract_file_query(message) is None:
            return None  # not a file request (no clean term to search)
        if _FILE_NOUN_RE.search(message.lower()) or _FILE_EXT_RE.search(message):
            return "files"
        return None
    return pre_route


def build_fallback_router(config: dict | None = None) -> Handler:
    """Routes messages the model failed to route (no [CALL] emitted at all).

    Tool patterns are deliberately narrow: only ones we've *seen* the model
    miss. Unrouted chat goes to the model — SmolLM2 holds a real conversation
    — except capability questions, which stay canned: asked what it can do,
    the model invents its own feature list (Wikipedia, Apple Watch). Set
    features.model_chat=false to canned-reply all unrouted chat instead."""
    features = (config or {}).get("features", {})
    model_chat = features.get("model_chat", True)

    def fallback(message: str) -> str | None:
        lower = message.lower()
        # memory questions FIRST — "what did you play earlier?" contains
        # "play …" and would otherwise become a spotify search for "earlier"
        if any(k in lower for k in ("earlier", "last time", "what did i",
                                    "what did you", "remember", "we talk")):
            return "memory"
        # "Play {song} on Spotify" — outside training distribution, produced
        # gibberish chat instead of a tool call; "Another song by X" has no
        # "play" at all but is still a music request
        if extract_play_query(message) is not None or "spotify" in lower \
                or extract_artist_request(message) is not None:
            return "spotify"
        # "How is the weather in New York?" — naming a location makes the
        # model treat it as a general question (argmax = text, no call)
        if "weather" in lower or "forecast" in lower or "temperature" in lower:
            return "weather"
        # "Tell me about the screenshot I just took" — the tool grammar in
        # training was live-screen questions; the screenshot phrasing is OOD
        if "screenshot" in lower or "screen shot" in lower:
            return "screen"
        # "Find my resume" / "where is my budget file" — no model token for
        # file search (like memory), so the keyword router is its only path.
        # Checked after the other tools so their phrasings win first.
        if extract_file_query(message) is not None:
            return "files"
        if _is_capability_question(message):
            return "chat"
        return None if model_chat else "chat"
    return fallback


# -- small talk ---------------------------------------------------------------

_GREETINGS = ("hello", "hi", "hey", "yo", "sup", "good morning",
              "good afternoon", "good evening", "what's up", "whats up")

_CAPABILITIES = ("I can check your calendar, reminders, weather, news, and "
                 "stocks, read your notes, describe your screen, find your "
                 "files, play music on Spotify, and remember what we've "
                 "talked about.")


def _is_greeting(message: str) -> bool:
    m = message.lower().strip(" .!?,")
    return any(m == g or m.startswith(g + " ") or m.startswith(g + ",")
               for g in _GREETINGS)


def _is_capability_question(message: str) -> bool:
    m = message.lower()
    return ("what can you do" in m or "who are you" in m
            or "what are you" in m or m.strip(" .!?") == "help")


def make_chat_handler(config: dict) -> Handler:
    """Canned replies for messages the fallback router sends to "chat" —
    capability questions always, everything unrouted when model_chat=false."""
    name = config.get("user", {}).get("name", "there")

    def chat_handler(message: str) -> str:
        m = message.lower()
        if _is_greeting(message):
            return f"Hey {name}! {_CAPABILITIES}"
        if "thank" in m:
            return "Anytime!"
        if _is_capability_question(message):
            return _CAPABILITIES
        return ("I'm not sure how to help with that one. "
                f"{_CAPABILITIES}")

    return chat_handler


# vague "…{something generic} by {artist}" — asks for *an* artist song, not a
# specific track; "Passionfruit by Drake" won't match (non-generic lead).
# "another …" variants matter: unmatched they became literal searches, and
# Spotify's catalog contains ambushes like Weird Al's "Another Tattoo
# (Parody of … by B.o.B feat. Bruno Mars)".
_ARTIST_REQUEST_RE = re.compile(
    r"^(?:play )?(?:another song|another track|another one|another|"
    r"a different song|something else|a song|a track|something|some music|"
    r"music|songs|anything)\s+by\s+(.+)$",
    re.IGNORECASE,
)


def extract_artist_request(text: str) -> str | None:
    match = _ARTIST_REQUEST_RE.match(text.strip().rstrip(".!?"))
    return match.group(1).strip() if match else None


# "in/for {Capitalized Place}" — capitalization keeps "in the morning" out
_LOCATION_RE = re.compile(r"\b(?:in|for) ([A-Z][\w.'-]*(?: [A-Z][\w.'-]*)*)")


def extract_location(message: str) -> str | None:
    match = _LOCATION_RE.search(message)
    return match.group(1).rstrip("?!.") if match else None


def build_reprompts() -> dict[str, Callable[[str], str]]:
    """Tool → prompt-builder for the dispatcher's reprompt mode: the real
    result becomes a fresh chat prompt, answered with the model's preserved
    instruct ability. For tools whose result should be understood, not
    recited — the user asked what's HAPPENING on screen, not for a read-back
    of the app list."""
    def screen_prompt(result: str) -> str:
        return (f"{result}\n"
                "Based on that, tell me in one or two sentences what I'm "
                "looking at and what's happening.")
    return {"screen": screen_prompt}


def build_verbatim() -> dict[str, Handler]:
    """Tool → reply template for the dispatcher's verbatim mode.

    Criterion: tools whose replies name arbitrary proper nouns (event titles,
    track/artist names, app names) or numbers that must be exact (money).
    Weather, news, and screen keep the model's voice for now — revisit if
    their garbling grates."""
    return {
        "calendar": _calendar_reply,
        "reminders": _reminders_reply,
        "notes": _notes_reply,
        "spotify": _spotify_reply,
        "stocks": _stocks_reply,
        "launcher": _launcher_reply,
        "files": lambda result: result,   # handler already returns a sentence
        "memory": lambda result: result,  # handler already returns a sentence
        "chat": lambda result: result,
    }
