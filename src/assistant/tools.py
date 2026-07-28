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
# document KINDS ("budget spreadsheet", "vacation photos") — as unambiguous a
# file signal as a bare "document", but not in _FILE_NOUN_RE because they're
# also the type-tail that gets stripped from the search term. The pre-router
# treats them as a force signal so "where's my budget spreadsheet" doesn't fall
# through to the model (which confidently mis-routes it to reminders).
_FILE_KIND_RE = re.compile(
    r"\b(?:spreadsheets?|presentations?|slideshows?|photos?|pictures?|"
    r"screenshots?|pdfs?)\b", re.I)
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


# term extraction for a message the MODEL already routed to files (no gate — it's
# already been classified as a file query, so pull the best search term). Handles
# the phrasings extract_file_query's strict routing gate deliberately rejects:
# "Find Kai's resume", "the most recent version of …".
_TERM_VERB_RE = re.compile(
    r"^(?:can you |could you |please )?"
    r"(?:find|locate|look\s+for|search\s+for|show\s+me|pull\s+up|get\s+me|"
    r"dig\s+up|track\s+down|where(?:'s| is| are)|i\s+(?:need|want)(?:\s+to\s+find)?)"
    r"\s+", re.I)
_TERM_QUALIFIERS_RE = re.compile(
    r"^(?:"
    r"(?:the|my|a|an|any|all)\s+"
    r"|most\s+recent\s+|latest\s+|newest\s+|recent\s+|current\s+|old\s+"
    r"|final\s+|signed\s+"
    r"|version\s+of\s+|copy\s+of\s+"
    r"|[\w][\w.'-]*'s\s+"          # third-person possessive ("Kai's")
    r")+", re.I)
_TERM_TRAILING_RE = re.compile(
    r"\s+(?:located|saved|stored|somewhere|please|file|document)\s*$", re.I)


def extract_file_term(message: str) -> str | None:
    """Best filename search term from a file query, no routing gate.

    "Find Kai's resume"                      → "resume"
    "Find the most recent version of my CV"  → "CV"
    "locate the budget spreadsheet"          → "budget"
    """
    text = message.strip().rstrip(".!?")
    text = _TERM_VERB_RE.sub("", text).strip()
    text = _TERM_TRAILING_RE.sub("", text).strip()
    text = _TERM_QUALIFIERS_RE.sub("", text).strip()
    text = _FILE_NOUN_PREFIX_RE.sub("", text).strip()
    text = _FILE_TYPE_TAIL_RE.sub("", text).strip()
    if text.lower() in _FILE_BARE_NOUNS:
        return None
    return text or None


# time-based file search ("files I worked on last week", "recent documents") —
# a recency window in days, or None. Distinct from name search (extract_file_query)
_RECENT_N_DAYS_RE = re.compile(r"\b(\d+)\s+days?\s+(?:ago|back)\b", re.I)
_RECENT_N_WEEKS_RE = re.compile(r"\b(\d+)\s+weeks?\s+(?:ago|back)\b", re.I)
_RECENCY_MAP = [
    (r"\btoday\b", 1),
    (r"\byesterday\b", 2),
    (r"\b(?:this|last|past|the\s+last|the\s+past)\s+week\b", 7),
    (r"\b(?:a|one)\s+week\s+(?:ago|back)\b", 7),
    (r"\bcouple\s+(?:of\s+)?weeks?\b", 14),
    (r"\btwo\s+weeks?\b", 14),
    (r"\b(?:this|last|past)\s+month\b", 30),
    (r"\b(?:a|one)\s+month\s+(?:ago|back)\b", 30),
    (r"\b(?:recently|lately|these\s+days)\b", 7),
    (r"\brecent\s+(?:files?|docs?|documents?|work|stuff)\b", 7),
]
# a file/work signal, so "what's on my calendar this week" doesn't become a
# file search — the recency phrase alone isn't enough to route to files
_FILE_WORK_WORDS = ("file", "document", " doc", "work on", "working on",
                    "worked on", "been working", "edited", "opened", "saved")


def extract_recent_days(message: str) -> int | None:
    """The recency window in days for a time-based file query, or None."""
    lower = message.lower()
    if (m := _RECENT_N_DAYS_RE.search(lower)):
        return int(m.group(1))
    if (m := _RECENT_N_WEEKS_RE.search(lower)):
        return int(m.group(1)) * 7
    for pattern, days in _RECENCY_MAP:
        if re.search(pattern, lower):
            return days
    return None


def is_recent_files_query(message: str) -> bool:
    """A time-based FILE query (recency phrase + a file/work signal) — used for
    routing, so calendar/reminder time queries aren't stolen."""
    lower = message.lower()
    return extract_recent_days(message) is not None \
        and any(w in lower for w in _FILE_WORK_WORDS)


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


# "Remind me to call the dentist" → create; anything else → read the list.
# Ordered longest-first so "add a reminder to" strips before "add a reminder".
_REMINDER_ADD_PREFIXES = ["remind me to ", "remind me ", "set a reminder to ",
                          "set a reminder ", "add a reminder to ",
                          "add a reminder ", "create a reminder to ",
                          "reminder to "]
# "…on/in/to my <name> list" targets a specific list; the name is matched
# against the user's ACTUAL lists at runtime (reminders._find_calendar)
_REMINDER_LIST_RE = re.compile(
    r"\b(?:on|in|to|from)\s+(?:my\s+|the\s+)?(.+?)\s+list\b", re.I)
_TRAILING_CONNECTORS = ("at", "on", "by", "for", "to", "in")


def extract_reminder_list(message: str) -> str | None:
    """The reminder-list name in the message ("…on my Work list"), or None."""
    m = _REMINDER_LIST_RE.search(message)
    return m.group(1).strip() if m else None


def extract_reminder(message: str) -> tuple[str, str | None] | None:
    """(text to remind about, list name or None) for a create request, else
    None. The list phrase is stripped from the text; the due date is left in for
    the handler to parse via NSDataDetector."""
    lower = message.lower()
    for prefix in _REMINDER_ADD_PREFIXES:
        if prefix in lower:
            text = message[lower.index(prefix) + len(prefix):].strip(" :.")
            list_name = extract_reminder_list(text)
            if list_name:
                text = _REMINDER_LIST_RE.sub("", text).strip(" :.")
            return (text, list_name) if text else None
    return None


# delete / reschedule intent — new write ops (Kai 2026-07-26: overdue clutter
# needs cleaning up from chat, and stale reminders need moving to a new day)
_REMINDER_WORD_RE = re.compile(r"\breminders?\b", re.I)
_REMINDER_DELETE_RE = re.compile(r"\b(?:delete|remove|clear)\b", re.I)
_REMINDER_OVERDUE_RE = re.compile(
    r"\b(?:overdue|past due|passed|expired|old)\b", re.I)
_REMINDER_RESCHED_RE = re.compile(
    r"\b(?:reschedule|move|push|change)\b", re.I)
_DUE_TODAY_READ_RE = re.compile(r"\btoday\b", re.I)  # "due today", "for today"
# words that are part of the request, never part of a reminder's title
_REMINDER_STOPWORDS = {
    "delete", "remove", "clear", "reschedule", "move", "push", "change",
    "the", "my", "a", "an", "all", "any", "please", "can", "you",
    "reminder", "reminders", "one", "ones", "that", "which", "where",
    "have", "has", "had", "already", "passed", "overdue", "past", "expired",
    "old", "due", "date", "day", "preexisting", "pre-existing", "existing",
    "to", "of", "for", "up",
}


def _reminder_title_words(message: str, date_text: str | None = None) -> str | None:
    """Whatever's left of the message once request words and the date phrase
    are stripped — the title fragment to match against real reminders."""
    text = message
    if date_text:
        text = text.replace(date_text, " ", 1)
    words = [w for w in re.findall(r"[A-Za-z0-9'-]+", text)
             if w.lower() not in _REMINDER_STOPWORDS]
    return " ".join(words) or None


def extract_reminder_delete(message: str) -> tuple[str | None, bool] | None:
    """(title fragment or None, overdue_only) for a delete request, else None.
    Needs both a delete verb and the word reminder(s) — high precision."""
    if not (_REMINDER_DELETE_RE.search(message)
            and _REMINDER_WORD_RE.search(message)):
        return None
    overdue = bool(_REMINDER_OVERDUE_RE.search(message))
    return _reminder_title_words(message), overdue


def is_reminder_reschedule(message: str) -> bool:
    """A reschedule request needs a move verb and the word reminder(s). The
    title fragment and new due date are extracted in the handler (the date must
    be parsed first so its words don't pollute the title match)."""
    return bool(_REMINDER_RESCHED_RE.search(message)
                and _REMINDER_WORD_RE.search(message))


def strip_due_text(text: str, date_text: str | None) -> str:
    """Remove the matched date phrase + any dangling connector so the title reads
    cleanly ("call the dentist tomorrow at 3pm" → "call the dentist")."""
    if date_text:
        text = text.replace(date_text, "", 1)
    words = text.strip(" ,.").split()
    while words and words[-1].lower() in _TRAILING_CONNECTORS:
        words.pop()
    return " ".join(words)


def _reminders_handler(message: str) -> str:
    from src.reminders import reminders  # EventKit import deferred
    from src.eventkit.store import AccessDenied
    try:
        # delete / reschedule BEFORE create: "move my reminder to 5pm" contains
        # the create prefix "reminder to " and would otherwise add a new one
        deletion = extract_reminder_delete(message)
        if deletion is not None:
            title_q, overdue = deletion
            return reminders.remove(title_query=title_q, only_overdue=overdue,
                                    list_name=extract_reminder_list(message))
        if is_reminder_reschedule(message):
            due, date_text = reminders.parse_due(message)
            if due is None:
                return "When should I move it to?"
            title_q = _reminder_title_words(message, date_text)
            if not title_q:
                return "Which reminder should I move?"
            return reminders.reschedule(title_q, due, date_text,
                                        list_name=extract_reminder_list(message))
        parsed = extract_reminder(message)
        if parsed is not None:
            text, list_name = parsed
            due, date_text = reminders.parse_due(text)
            title = strip_due_text(text, date_text)
            if not title:
                return "What should the reminder say?"
            return reminders.add(title, due=due, due_text=date_text,
                                 list_name=list_name)
        # "due today" reads filter to today (live complaint: asking what's due
        # today listed reminders from three different days)
        if _DUE_TODAY_READ_RE.search(message):
            return reminders.due_today_summary(extract_reminder_list(message))
        # read — optionally from a named list ("what's on my Work list?")
        return reminders.incomplete_summary(extract_reminder_list(message))
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
        # 4, not 5: the reply lands in a fixed-height slot on the orb surface,
        # and a fifth line clipped. Four hits is plenty to recognise the one
        # you meant, which is what a file search is actually for.
        max_results = config.get("files", {}).get("max_results", 4)
        # a recency phrase ("last week", "recent files") → time search; the
        # window wins over any name, so "files I worked on last week" isn't a
        # literal name search for "worked on last week"
        days = extract_recent_days(message)
        if days is not None:
            return files.recent(days, max_results)
        # extract_file_term (not extract_file_query) — routing already happened
        # (model token or fallback), so pull the term without the strict gate
        query = extract_file_term(message)
        if not query:
            return "Tell me part of the file name to search for"
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
    if result.startswith(("Reminder added", "Couldn't save")):
        return result  # a create confirmation — already a full sentence
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
        # reminder create / list-add FIRST: the files token over-fires on
        # "add milk to my shopping list" (the word "list" pulls it to files),
        # and because the model emits a call the fallback never gets to correct
        # it. These phrasings are high-precision (an add-prefix or "…to my X
        # list"), so forcing reminders here is safe and symmetric with files.
        if extract_reminder(message) is not None \
                or extract_reminder_list(message) is not None:
            return "reminders"
        # delete/reschedule are UNTRAINED write phrasings — the model has never
        # seen them, so don't let it guess ("delete the run reminder" must not
        # become a file search)
        if extract_reminder_delete(message) is not None \
                or is_reminder_reschedule(message):
            return "reminders"
        if extract_file_query(message) is None:
            return None  # not a file request (no clean term to search)
        lower = message.lower()
        if (_FILE_NOUN_RE.search(lower) or _FILE_KIND_RE.search(lower)
                or _FILE_EXT_RE.search(message)):
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
        # time-based FILE query FIRST — "what did I work on last week" contains
        # "what did i" and would otherwise be stolen by the memory rule below
        if is_recent_files_query(message):
            return "files"
        # memory questions — "what did you play earlier?" contains "play …" and
        # would otherwise become a spotify search for "earlier"
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
        if extract_file_query(message) is not None \
                or is_recent_files_query(message):
            return "files"
        # "Remind me to go for a run in an hour" — reminder CREATE phrasing was
        # never trained (the model saw only a couple of READ prompts), so it
        # falls to chat and produces garbage; casing variants of reads miss too.
        # Checked after weather so "remind me what the weather is" stays weather.
        if extract_reminder(message) is not None \
                or extract_reminder_list(message) is not None \
                or "reminder" in lower or "to-do" in lower or "to do list" in lower:
            return "reminders"
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
