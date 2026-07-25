"""
Synthetic tool-call training data.

Faithfulness redesign (2026-07-17): result content is generated compositionally
(random names, invented proper nouns, course codes, arbitrary numbers/times) so
it almost never repeats across examples. With the old small fixed pools the
model could reach low loss by memorising pool items instead of reading the
[RESULT] block — which is exactly the unfaithful behaviour observed at
inference. High-entropy content makes copying from context the only low-loss
strategy. Replies echo result content *verbatim* (no case changes) to maximise
the copy signal.

Result string *formats* are a contract with src/assistant/tools.py — the real
handlers emit these exact shapes, including the fallback strings ("Calendar
access not granted", "X is disabled in config"), which are trained here too so
the model wraps real failure modes gracefully.
"""
import argparse
import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Prompt pools (routing signal — unchanged from the 8/8-routing run)
# ---------------------------------------------------------------------------

_CALENDAR_PROMPTS = [
    "What's on my calendar today?",
    "What is on my calendar today?",
    "What's on my calendar?",
    "What do I have scheduled today?",
    "What do I have scheduled?",
    "Any meetings today?",
    "Do I have any meetings today?",
    "What are my events for today?",
    "What events do I have today?",
    "What's my schedule look like?",
    "What is my schedule for today?",
    "What does my day look like?",
    "Do I have anything on today?",
    "Do I have anything scheduled?",
    "What meetings do I have?",
    "What meetings are coming up?",
    "Check my calendar.",
    "Check my calendar for today.",
    "Pull up my calendar.",
    "Look at my calendar.",
    "What's coming up today?",
    "What's next on my calendar?",
    "Am I busy today?",
    "Am I free this afternoon?",
    "Do I have time this morning?",
    "What's on my agenda?",
    "What is on my agenda today?",
    "Show me my agenda.",
    "Any appointments today?",
    "Do I have any appointments?",
    "When's my next meeting?",
    "What's my first meeting today?",
    "Is my afternoon booked?",
    "What have I got on today?",
]

_SCREEN_PROMPTS = [
    "What's on my screen?",
    "What is on my screen?",
    "What's on my screen right now?",
    "What am I looking at?",
    "What am I looking at on screen?",
    "Describe my screen.",
    "Describe what's on my screen.",
    "Take a screenshot and describe it.",
    "What's currently on my display?",
    "What is displayed on my screen?",
    "What am I working on?",
    "What am I working on right now?",
    "Can you see my screen?",
    "Can you look at my screen?",
    "What's open on my computer?",
    "What apps are open on my screen?",
    "What does my screen show?",
    "What's shown on my display?",
    "Read my screen.",
    "What windows do I have open?",
    "Capture my screen.",
    "Tell me what's on my screen.",
    "What's up on my monitor?",
    "Look at my screen and tell me what you see.",
]

_REMINDERS_PROMPTS = [
    "What are my reminders?",
    "What are my reminders for today?",
    "What do I need to remember today?",
    "What do I need to remember?",
    "Check my reminders.",
    "Check my reminders for today.",
    "Any reminders for today?",
    "Do I have any reminders today?",
    "What's on my reminder list?",
    "What is on my reminders list?",
    "Do I have any reminders set?",
    "Are there any reminders set?",
    "Show me my reminders.",
    "Pull up my reminders.",
    "List my reminders.",
    "What should I not forget today?",
    "What do I have to do today?",
    "What's on my to-do list?",
    "What tasks do I have?",
    "Remind me what I need to do.",
    "Do I have anything to take care of?",
    "What am I supposed to do today?",
]

_NOTES_PROMPTS = [
    "What did I note today?",
    "What did I write in my notes?",
    "Check my notes.",
    "Check my notes for today.",
    "What's in my notes?",
    "What is in my notes?",
    "Show me today's notes.",
    "Show me my notes.",
    "Did I write anything down?",
    "Did I make any notes today?",
    "What are my notes?",
    "What notes do I have?",
    "Pull up my notes.",
    "Open my notes.",
    "Read my notes.",
    "What have I noted recently?",
    "What have I written down?",
    "Anything in my notes?",
    "What did I jot down?",
    "Go through my notes.",
    "What notes did I take today?",
]

_SPOTIFY_PROMPTS = [
    "What's playing?",
    "What's playing on Spotify?",
    "What is playing right now?",
    "What song is this?",
    "What song is playing?",
    "What track is this?",
    "Pause the music.",
    "Pause Spotify.",
    "Pause the song.",
    "Skip this song.",
    "Skip this track.",
    "Next song.",
    "Play the next track.",
    "Turn up the volume.",
    "Turn the music up.",
    "Turn down the volume.",
    "Turn the music down.",
    "Lower the volume.",
    "Play something.",
    "Play some music.",
    "Resume the music.",
    "What artist is this?",
    "Who sings this?",
    "Who is this by?",
    "Stop the music.",
    "Stop Spotify.",
    "What's the current song?",
    "What am I listening to?",
]

_LAUNCHER_TEMPLATES = [
    "Open {a}.",
    "Launch {a}.",
    "Start {a}.",
    "Can you open {a}?",
    "Fire up {a}.",
    "Open up {a} for me.",
    "Get {a} open.",
    "Bring up {a}.",
]

_WEATHER_PROMPTS = [
    "What's the weather like?",
    "What is the weather like today?",
    "How's the weather today?",
    "How is the weather outside?",
    "Will it rain today?",
    "Is it going to rain?",
    "Do I need a jacket?",
    "Do I need a coat today?",
    "What's the temperature outside?",
    "What is the temperature right now?",
    "How warm is it outside?",
    "Is it cold out?",
    "Is it warm out today?",
    "What's the forecast?",
    "What is the forecast for today?",
    "What's the forecast looking like?",
    "Should I bring an umbrella?",
    "Do I need an umbrella today?",
    "How hot is it today?",
    "How cold is it out?",
    "What's the weather looking like?",
    "Is it sunny out?",
    "What's it like outside?",
    "Give me the weather.",
]

_NEWS_PROMPTS = [
    "What's in the news?",
    "What is in the news today?",
    "Any news today?",
    "Is there any news?",
    "What's happening in the world?",
    "What is happening in the world today?",
    "Give me the headlines.",
    "Give me today's headlines.",
    "What are the top stories?",
    "What are the top headlines?",
    "Any tech news?",
    "Any news in tech today?",
    "What's going on today?",
    "What's going on in the news?",
    "Catch me up on the news.",
    "Catch me up on today's news.",
    "Show me the news.",
    "Read me the headlines.",
    "What's the latest news?",
    "Any breaking news?",
    "What stories are trending?",
    "Update me on the news.",
]

_STOCK_PROMPT_TEMPLATES = [
    "How's {s} doing?",
    "Is {s} up or down?",
    "What's {s} at?",
    "What's {s} trading at?",
    "Check {s} for me.",
    "How's {s} today?",
]

_STOCKS_GENERAL_PROMPTS = [
    "Check my watchlist.",
    "Is the market up today?",
    "How are my stocks?",
    "How's the market today?",
    "What are my stocks doing?",
    "Give me a market update.",
]


# ---------------------------------------------------------------------------
# High-entropy content generators
# ---------------------------------------------------------------------------

_FIRST_NAMES = [
    "Sarah", "Jake", "Tessa", "Marcus", "Priya", "Daniel", "Ines", "Tom",
    "Amara", "Leo", "Nadia", "Chris", "Yuki", "Omar", "Elena", "Sam",
    "Ravi", "Clara", "Diego", "Maja", "Felix", "Aisha", "Ben", "Lena",
    "Kofi", "Rosa", "Ethan", "Zara", "Noah", "Ida", "Mateo", "June",
    "Anders", "Bilal", "Greta", "Hana", "Ivan", "Joelle", "Kai", "Luca",
]

_LAST_NAMES = [
    "Okafor", "Lindqvist", "Marchetti", "Novak", "Reyes", "Tanaka",
    "Osei", "Bergstrom", "Kaur", "Delgado", "Fischer", "Haddad",
    "Ivanova", "Johansson", "Kimura", "Laurent", "Moreau", "Nakamura",
    "Oduya", "Petrov", "Quintana", "Rossi", "Sato", "Thorne",
    "Ueda", "Vasquez", "Weber", "Xu", "Yilmaz", "Zhang",
]

_SYLLABLES = [
    "ka", "ro", "ven", "tal", "mir", "zo", "len", "dar", "fi", "nex",
    "bel", "tur", "sa", "gri", "pol", "dun", "cha", "vor", "li", "mak",
    "os", "quin", "ther", "ul", "brin", "cor", "del", "fen", "gal", "hyr",
]

_NOUNS = [
    "budget", "insurance", "groceries", "laptop", "roadmap", "invoice",
    "passport", "timesheet", "onboarding", "retro", "sprint", "design",
    "marketing", "hiring", "security", "billing", "backlog", "pricing",
    "launch", "audit", "contract", "renewal", "prototype", "pipeline",
    "migration", "rollout", "offsite", "training", "survey", "handoff",
    "dashboard", "cleanup", "release", "quarterly", "compliance", "vendor",
]

_DEPTS = ["CS", "BME", "SSW", "EE", "MA", "PHYS", "BIO", "CHE", "HUM", "MGT"]

_REAL_APPS = [
    "Chrome", "Slack", "Calculator", "Figma", "Finder", "VS Code", "Safari",
    "Spotify", "Terminal", "Notion", "Xcode", "Discord", "Mail", "Photos",
    "Messages", "Preview", "Zoom", "Notes", "Obsidian", "Calendar",
]

_REAL_TICKERS = [
    "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "INTC",
    "NFLX", "DIS", "BA", "JPM", "V", "WMT", "KO", "PEP", "COST", "PLTR",
    "UBER", "SHOP", "CRM", "ORCL", "QCOM", "MU",
]

# must match src/weather/weather.py's condition vocabulary exactly
_WEATHER_CONDITIONS = [
    "sunny", "partly cloudy", "overcast", "foggy", "light rain", "rainy",
    "heavy rain", "freezing rain", "snowy", "stormy", "unsettled",
]

_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _name() -> str:
    if random.random() < 0.5:
        return f"{random.choice(_FIRST_NAMES)} {random.choice(_LAST_NAMES)}"
    return random.choice(_FIRST_NAMES)


def _pword() -> str:
    """Invented proper noun ("Venkaro") — can only be produced by copying."""
    n = random.randint(2, 3)
    return "".join(random.choice(_SYLLABLES) for _ in range(n)).capitalize()


def _noun() -> str:
    return random.choice(_NOUNS)


def _course() -> str:
    sep = random.choice(["", " "])
    return f"{random.choice(_DEPTS)}{sep}{random.randint(100, 799)}"


def _time() -> str:
    hour = random.randint(1, 12)
    suffix = random.choice(["am", "pm"])
    if random.random() < 0.5:
        return f"{hour}{suffix}"
    return f"{hour}:{random.choice([5, 10, 15, 20, 30, 40, 45, 50]):02d}{suffix}"


def _app() -> str:
    return random.choice(_REAL_APPS) if random.random() < 0.7 else _pword()


def _ticker() -> str:
    if random.random() < 0.6:
        return random.choice(_REAL_TICKERS)
    return "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(random.randint(2, 5)))


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _wrap(tool: str, result: str, reply: str, prompt: str) -> tuple[str, str]:
    return prompt, f"[CALL: {tool}][RESULT]{result}[/RESULT]\n{reply}"


def _disabled(tool: str, prompt: str) -> tuple[str, str]:
    return _wrap(tool, f"{tool} is disabled in config",
                 f"The {tool} tool is disabled in your config.", prompt)


# ---------------------------------------------------------------------------
# Per-tool example builders
# ---------------------------------------------------------------------------

def _event_title() -> str:
    return random.choice([
        lambda: f"1:1 with {_name()}",
        lambda: f"Lunch with {_name()}",
        lambda: f"Call with {_name()}",
        lambda: f"{_noun().capitalize()} review",
        lambda: f"{_noun().capitalize()} sync",
        lambda: f"{_noun().capitalize()} planning",
        lambda: f"{_pword()} demo",
        lambda: f"{_course()} lecture",
        lambda: f"{_course()} office hours",
        lambda: f"Interview with {_name()}",
        lambda: f"Dentist",
        lambda: f"Doctor appointment",
        lambda: f"{_pword()} kickoff",
        lambda: f"Gym with {_name()}",
    ])()


def _calendar() -> tuple[str, str]:
    prompt = random.choice(_CALENDAR_PROMPTS)
    roll = random.random()
    if roll < 0.05:
        return _wrap("calendar", "Calendar access not granted",
                     "I can't see your calendar — access hasn't been granted yet.", prompt)
    if roll < 0.15:
        return _wrap("calendar", "No events today", "Your calendar is clear today.", prompt)

    entries = []
    for _ in range(random.randint(1, 3)):
        title = _event_title()
        if random.random() < 0.15:
            entries.append(f"{title} (all day)")
        else:
            entries.append(f"{title} at {_time()}")
    result = ", ".join(entries)
    reply = random.choice([
        f"You have {_join(entries)} today.",
        f"Today: {_join(entries)}.",
        f"On your calendar: {_join(entries)}.",
    ])
    return _wrap("calendar", result, reply, prompt)


def _screen() -> tuple[str, str]:
    prompt = random.choice(_SCREEN_PROMPTS)
    if random.random() < 0.03:
        return _disabled("screen", prompt)

    front = _app()
    others = random.sample([a for a in _REAL_APPS if a != front], random.randint(0, 4))
    if random.random() < 0.3:
        others.append(_pword())
    random.shuffle(others)

    if not others:
        result = f"{front} in front"
        reply = random.choice([
            f"{front} is the only app in front right now.",
            f"You're in {front}.",
        ])
    else:
        result = f"{front} in front, also open: {', '.join(others)}"
        reply = random.choice([
            f"You're in {front}, with {_join(others)} open.",
            f"{front} is in front; {_join(others)} are also open.",
            f"Right now {front} is in front, and you also have {_join(others)} open.",
        ])
    return _wrap("screen", result, reply, prompt)


def _task() -> str:
    return random.choice([
        lambda: f"Call {_name()}",
        lambda: f"Email {_name()}",
        lambda: f"Pay {_noun()} bill",
        lambda: f"Submit {_noun()} report",
        lambda: f"Buy {_noun()}",
        lambda: f"Renew {_noun()}",
        lambda: f"Book {_noun()} appointment",
        lambda: f"Pick up {_noun()}",
        lambda: f"{_course()} homework",
        lambda: f"{_course()} Final Exam",
        lambda: f"Review pull request from {_name()}",
        lambda: f"Get {_noun()} checked",
        lambda: f"Bring {_noun()} just in case",
        lambda: f"{_pword()} - {_noun()} follow-up",
    ])()


def _reminders() -> tuple[str, str]:
    prompt = random.choice(_REMINDERS_PROMPTS)
    roll = random.random()
    if roll < 0.05:
        return _wrap("reminders", "Reminders access not granted",
                     "I can't see your reminders — access hasn't been granted yet.", prompt)
    if roll < 0.15:
        return _wrap("reminders", "No reminders set",
                     "You don't have any reminders set right now.", prompt)

    items = [_task() for _ in range(random.randint(1, 4))]
    result = ", ".join(items)
    if len(items) == 1:
        reply = f"You have one reminder: {items[0]}."
    else:
        reply = random.choice([
            f"You have {len(items)} reminders: {_join(items)}.",
            f"On your list: {_join(items)}.",
        ])
    return _wrap("reminders", result, reply, prompt)


def _notes() -> tuple[str, str]:
    prompt = random.choice(_NOTES_PROMPTS)
    if random.random() < 0.15:
        return _wrap("notes", "No notes for today",
                     "You haven't written anything in your notes yet today.", prompt)

    content = random.choice([
        lambda: f"{_noun().capitalize()} ideas: {_noun()}, {_noun()}, {_noun()}",
        lambda: f"Meeting with {_name()}: action item — {_task()}",
        lambda: f"{_course()} exam on {random.choice(_DAYS)} — review {_noun()}",
        lambda: f"Grocery list: {_noun()}, {_noun()}, {_noun()}",
        lambda: f"Don't forget: {_task()} before {random.choice(_DAYS)}",
        lambda: f"Bug: {_pword()} fails when {_noun()} is empty",
        lambda: f"Book recommendation from {_name()}: {_pword()}",
    ])()
    reply = random.choice([
        f"Your notes say: {content}.",
        f"You wrote down: {content}.",
        f"From your notes today: {content}.",
    ])
    return _wrap("notes", content, reply, prompt)


def _track() -> tuple[str, str]:
    title = random.choice([
        lambda: f"{_pword()} {_noun().capitalize()}",
        lambda: f"{_noun().capitalize()} {_noun().capitalize()}",
        lambda: _pword(),
    ])()
    artist = random.choice([
        lambda: _name(),
        lambda: f"The {_pword()}s",
        lambda: _pword(),
    ])()
    return title, artist


def _spotify() -> tuple[str, str]:
    prompt = random.choice(_SPOTIFY_PROMPTS)
    if random.random() < 0.03:
        return _disabled("spotify", prompt)

    action = random.choice(["status", "pause", "skip", "volume_up", "volume_down"])
    title, artist = _track()
    vol = random.randint(4, 100)

    if action == "status":
        result = f"{title} by {artist}, volume {vol}%"
        reply = f"{title} by {artist} is playing at {vol}% volume."
    elif action == "pause":
        result = "Paused"
        reply = "Music paused."
    elif action == "skip":
        result = f"Now playing: {title} by {artist}"
        reply = f"Skipped — now playing {title} by {artist}."
    elif action == "volume_up":
        result = f"Volume set to {vol}%"
        reply = f"Volume turned up to {vol}%."
    else:
        result = f"Volume set to {vol}%"
        reply = f"Volume turned down to {vol}%."
    return _wrap("spotify", result, reply, prompt)


def _launcher() -> tuple[str, str]:
    app = _app()
    prompt = random.choice(_LAUNCHER_TEMPLATES).format(a=app)
    if random.random() < 0.1:
        return _wrap("launcher", "No matching app in the allowed apps list",
                     f"I couldn't find {app} in your allowed apps list.", prompt)
    return _wrap("launcher", f"{app} launched", f"{app} is open.", prompt)


def _weather() -> tuple[str, str]:
    prompt = random.choice(_WEATHER_PROMPTS)
    roll = random.random()
    if roll < 0.03:
        return _disabled("weather", prompt)
    if roll < 0.06:
        loc = _pword()
        return _wrap("weather", f"weather error: location not found: {loc}",
                     f"I couldn't look up the weather for {loc}.", prompt)

    temp = random.randint(5, 105)
    condition = random.choice(_WEATHER_CONDITIONS)
    result = f"{temp}°F, {condition}"
    if "rain" in condition or condition in ("stormy", "snowy"):
        reply = f"It's {temp}°F and {condition} — you might want an umbrella."
    elif temp < 50:
        reply = f"It's {temp}°F and {condition} — a jacket is a good idea."
    else:
        reply = random.choice([
            f"It's {temp}°F and {condition} outside.",
            f"Currently {temp}°F and {condition}.",
        ])
    return _wrap("weather", result, reply, prompt)


def _company() -> str:
    return _pword() + random.choice(["", " Labs", " Systems", "Corp", " Tech"])


def _headline() -> str:
    return random.choice([
        lambda: f"{_company()} announces new {_noun()} {random.choice(['platform', 'service', 'tool'])}",
        lambda: f"{_company()} shares {random.choice(['rise', 'fall'])} {round(random.uniform(1, 12), 1)}% after {_noun()} report",
        lambda: f"{_name()} named CEO of {_company()}",
        lambda: f"Scientists report {_noun()} breakthrough at {_pword()} University",
        lambda: f"{_pword()} City approves new {_noun()} plan",
        lambda: f"{_company()} recalls {_noun()} product over safety concerns",
        lambda: f"{_company()} to cut {random.randint(2, 90) * 100} jobs in {_noun()} shakeup",
        lambda: f"New study links {_noun()} to {_noun()} risks",
    ])()


def _news() -> tuple[str, str]:
    prompt = random.choice(_NEWS_PROMPTS)
    if random.random() < 0.05:
        return _wrap("news", "No headlines available right now",
                     "I couldn't fetch any headlines right now.", prompt)

    headlines = [_headline() for _ in range(random.randint(2, 4))]
    result = "; ".join(headlines)
    reply = random.choice([
        f"Here are today's top stories: {'; '.join(headlines)}.",
        f"In the news: {'; '.join(headlines)}.",
    ])
    return _wrap("news", result, reply, prompt)


def _stock_entry(sym: str) -> tuple[str, str]:
    price = round(random.uniform(2, 1500), 2)
    change = round(random.uniform(-6, 6), 1)
    direction = "+" if change >= 0 else ""
    trend = "up" if change >= 0 else "down"
    return f"{sym}: ${price} ({direction}{change}%)", f"{sym} is {trend} {abs(change)}% at ${price}"


def _stocks() -> tuple[str, str]:
    if random.random() < 0.5:
        sym = _ticker()
        prompt = random.choice(_STOCK_PROMPT_TEMPLATES).format(s=sym)
        if random.random() < 0.05:
            return _wrap("stocks", f"{sym}: no data",
                         f"I couldn't get data for {sym}.", prompt)
        result, reply = _stock_entry(sym)
    else:
        prompt = random.choice(_STOCKS_GENERAL_PROMPTS)
        symbols = [_ticker() for _ in range(random.randint(1, 3))]
        entries, parts = zip(*[_stock_entry(s) for s in symbols])
        result = ", ".join(entries)
        reply = "; ".join(parts) + "."
    return _wrap("stocks", result, reply, prompt)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# No-tool chat — routing contrast (run-6 backlog item, finally trained)
# ---------------------------------------------------------------------------
# greetings/thanks/small-talk must produce a PLAIN reply with no [CALL] token.
# in the OPT era these were handled by a decoding-time confidence gate because
# the model half-wanted spotify on "Hello." — now the contrast is in the data.

_CHAT_EXAMPLES = [
    (["Hello!", "Hello.", "Hi!", "Hi there!", "Hey!", "hey", "Yo!", "Hello Desktop Helper",
      "Good morning!", "Good morning", "Good afternoon!", "Good evening!"],
     ["Hey! What can I do for you?",
      "Hi! Need your calendar, the weather, or some music?",
      "Hello! How can I help?",
      "Hey there — what do you need?"]),
    (["Thanks!", "Thank you!", "Thanks so much!", "thx", "Appreciate it!", "Perfect, thanks."],
     ["Anytime!", "You're welcome!", "Happy to help!", "Of course!"]),
    (["How are you?", "How's it going?", "What's up?", "How are you doing today?"],
     ["Doing great — ready to help. What do you need?",
      "All good here! What can I do for you?"]),
    (["What can you do?", "What are you able to do?", "Help", "What do you do?",
      "Who are you?", "What are you?"],
     ["I'm your desktop assistant — I can check your calendar, reminders, weather, "
      "news, and stocks, read your notes, describe your screen, and play music on Spotify.",
      "I can manage your calendar and reminders, fetch weather, news, and stock prices, "
      "read notes, describe your screen, and control Spotify."]),
    (["Goodbye!", "Bye!", "See you later!", "Good night!", "gtg, bye"],
     ["See you later!", "Bye! I'll be here.", "Good night!"]),
]


def _chat() -> tuple[str, str]:
    prompts, replies = random.choice(_CHAT_EXAMPLES)
    return random.choice(prompts), random.choice(replies)


# -- file finder --------------------------------------------------------------
# Deliberately wide phrasing coverage — the whole reason for this tool token is
# that keyword routing missed natural queries ("find the most recent version of
# kai's resume", "where is X located"). Third-person ("kai's"), "the/most
# recent/latest", "where is … located", and bare "find X" are all here.
_FILE_TOPICS = [
    "resume", "CV", "cover letter", "budget", "invoice", "receipt", "report",
    "quarterly report", "presentation", "slides", "essay", "thesis", "contract",
    "lease", "tax return", "tax documents", "spreadsheet", "meeting notes",
    "project proposal", "grocery list", "reading list", "screenshot", "photo",
    "vacation photos", "boarding pass", "W2", "pay stub", "insurance form",
    "signed contract", "project plan", "budget spreadsheet",
]

_FILE_TEMPLATES = [
    "Find my {t}", "Find the {t}", "Find {who} {t}",
    "Find the most recent version of {who2} {t}",
    "Find the latest {t}", "Find the newest {t}",
    "Where is my {t}", "Where's my {t}", "Where is the {t} located",
    "Where did I save my {t}", "Where do I have my {t}",
    "Locate my {t}", "Locate the {t}", "Locate {who} {t}",
    "Search for my {t}", "Search for the {t}",
    "Look for my {t}", "Look for the {t}",
    "Pull up my {t}", "Can you find my {t}", "Can you find the {t}",
    "Find the file called {t}", "Do I have a {t} file",
    "I need to find my {t}", "Get me my {t}", "Get me the {t}",
    "Track down my {t}", "Dig up my {t}",
]

_FILE_EXTS = [".pdf", ".docx", ".pages", ".txt", ".md", ".xlsx", ".pptx", ".key"]
_FILE_DIRS = ["~/Documents", "~/Downloads", "~/Desktop", "~/Documents/Work",
              "~/Documents/Personal"]


def _filename(topic: str) -> str:
    base = topic.lower().replace("the ", "").replace("last year's ", "") \
                .replace("'s", "").replace("my ", "").strip().replace(" ", "_")
    stem = random.choice([base, base.capitalize(), f"{_name()}_{base}",
                          f"{base}_v2", f"{base}_final", f"{base}_2024",
                          f"{base}_draft"])
    return stem + random.choice(_FILE_EXTS)


_RECENT_FILE_PROMPTS = [
    "What files did I work on last week?", "What did I work on last week?",
    "What did I work on a couple weeks ago?", "Show me my recent files",
    "Show me the files I worked on recently", "What have I been working on lately?",
    "Files I edited yesterday", "What documents did I edit this week?",
    "Recent documents", "What files have I changed recently?",
    "Pull up my recent documents", "What did I work on a few days ago?",
    "Files from the last week", "What was I working on yesterday?",
    "What did I edit in the last couple days?", "Show me recently modified files",
]
_RECENT_WINDOWS = ["last day", "last couple days", "last week", "last two weeks",
                   "last month", "last 3 days"]


def _recent_files_result() -> str:
    window = random.choice(_RECENT_WINDOWS)
    n = random.randint(1, 4)
    listing = ", ".join(f"{_filename(random.choice(_FILE_TOPICS))} "
                        f"({random.choice(_FILE_DIRS)})" for _ in range(n))
    return f"{n} file{'s' if n > 1 else ''} from the {window}: {listing}"


def _files() -> tuple[str, str]:
    # time-based ("what did I work on last week?") vs name-based file queries
    if random.random() < 0.25:
        prompt = random.choice(_RECENT_FILE_PROMPTS)
        result = _recent_files_result()
        return _wrap("files", result, result, prompt)
    if random.random() < 0.15:
        # explicit filename with an extension
        name = _filename(random.choice(_FILE_TOPICS))
        prompt = random.choice([f"Find {name}", f"Where is {name}",
                                f"Locate {name}", f"Pull up {name}"])
        topic_for_result = name
    else:
        template = random.choice(_FILE_TEMPLATES)
        topic = random.choice(_FILE_TOPICS)
        prompt = template.format(t=topic, who=f"{_name()}'s",
                                 who2=random.choice(["my", f"{_name()}'s"]))
        topic_for_result = topic

    if random.random() < 0.12:
        term = topic_for_result.split()[-1].split(".")[0]
        return _wrap("files", f"No files found matching '{term}'",
                     f"I couldn't find any files matching '{term}'.", prompt)

    n = random.randint(1, 4)
    listing = ", ".join(f"{_filename(topic_for_result)} ({random.choice(_FILE_DIRS)})"
                        for _ in range(n))
    result = f"Found {n} file{'s' if n > 1 else ''}: {listing}"
    return _wrap("files", result, result, prompt)  # files is verbatim → reply = result


_BUILDERS = [_calendar, _screen, _reminders, _notes, _spotify,
             _launcher, _weather, _news, _stocks, _files, _chat]


def generate(count: int, seed: int | None = None) -> list[dict]:
    if seed is not None:
        random.seed(seed)
    examples = []
    for _ in range(count):
        builder = random.choice(_BUILDERS)
        prompt, response = builder()
        examples.append({"prompt": prompt, "response": response})
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic tool-call training examples")
    parser.add_argument("--output", default="data/tool_calls.jsonl")
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    examples = generate(args.count, seed=args.seed)
    with open(out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} examples → {out}")


if __name__ == "__main__":
    main()
