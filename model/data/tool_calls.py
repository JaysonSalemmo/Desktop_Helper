import argparse
import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Data pools
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

_CALENDAR_EVENTS = [
    "Standup at 9am",
    "1:1 with manager at 2pm",
    "Design review at 3pm",
    "Team lunch at 12pm",
    "Sprint planning at 10am",
    "All-hands at 4pm",
    "Code review at 11am",
    "Interview at 1pm",
    "Doctor appointment at 3:30pm",
    "Dentist at 10:30am",
    "Retrospective at 9:30am",
    "Product demo at 2:30pm",
    "Budget review at 3pm",
    "Happy hour at 5pm",
    "Onboarding call at 11am",
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

_SCREEN_RESULTS = [
    ("VS Code with a Python file open, browser tab with Stack Overflow",
     "You have VS Code open with a Python file and a Stack Overflow tab in your browser."),
    ("Terminal with pytest output showing 3 failing tests",
     "It looks like you're running tests — there are 3 failing tests in your terminal."),
    ("Slack with several unread messages in the #engineering channel",
     "You have Slack open with unread messages in #engineering."),
    ("Chrome with Gmail open, 4 unread emails",
     "You have Gmail open in Chrome with 4 unread emails."),
    ("Figma with a UI design file open",
     "You have a UI design file open in Figma."),
    ("Zoom meeting in progress, screen sharing active",
     "You're in a Zoom meeting with screen sharing active."),
    ("Spotify desktop app open, music paused",
     "Spotify is open and music is currently paused."),
    ("Notes app with a shopping list visible",
     "You have Notes open with what looks like a shopping list."),
    ("Two terminal windows and a browser with documentation",
     "You have two terminal windows open and a browser showing documentation."),
    ("Calendar app showing this week's schedule",
     "You have your calendar app open showing this week's schedule."),
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

_REMINDERS_ITEMS = [
    "Call dentist",
    "Buy groceries",
    "Submit timesheet",
    "Call mom",
    "Pay rent",
    "Pick up dry cleaning",
    "Reply to Sarah's email",
    "Book flight for conference",
    "Renew gym membership",
    "Take medication at 8pm",
    "Water the plants",
    "Back up laptop",
    "Schedule car service",
    "Send invoice to client",
    "Review pull request from Jake",
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

_NOTES_CONTENT = [
    ("Meeting notes: discussed Q3 roadmap, action item — update the backlog",
     "Your notes have a record of a meeting about the Q3 roadmap. Action item: update the backlog."),
    ("Ideas for the new feature: dark mode toggle, keyboard shortcuts, export to PDF",
     "You jotted down some feature ideas: dark mode toggle, keyboard shortcuts, and export to PDF."),
    ("Grocery list: milk, eggs, bread, coffee, apples",
     "Your notes have a grocery list: milk, eggs, bread, coffee, and apples."),
    ("Passwords note: do not forget to rotate API keys before Friday",
     "You have a note reminding you to rotate API keys before Friday."),
    ("Book recommendations from James: Atomic Habits, The Pragmatic Programmer",
     "You noted some book recommendations from James: Atomic Habits and The Pragmatic Programmer."),
    ("Bug investigation: race condition in the auth middleware, repros under high load",
     "There's a note about a bug — a race condition in the auth middleware that repros under high load."),
    ("No notes for today",
     "You haven't written anything in your notes yet today."),
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

_SPOTIFY_TRACKS = [
    ("Bohemian Rhapsody", "Queen"),
    ("Hotel California", "Eagles"),
    ("Stairway to Heaven", "Led Zeppelin"),
    ("Smells Like Teen Spirit", "Nirvana"),
    ("Lose Yourself", "Eminem"),
    ("Blinding Lights", "The Weeknd"),
    ("Shape of You", "Ed Sheeran"),
    ("Levitating", "Dua Lipa"),
    ("As It Was", "Harry Styles"),
    ("Anti-Hero", "Taylor Swift"),
    ("Flowers", "Miley Cyrus"),
    ("Watermelon Sugar", "Harry Styles"),
]

# launcher prompts are generated per-app from templates so phrasing varies while
# the app name (the thing that must route correctly) stays prominent
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

_LAUNCHER_APPS = [
    ("Chrome", "Chrome is open."),
    ("Slack", "Slack is now open."),
    ("Calculator", "Calculator is open."),
    ("Figma", "Figma is open."),
    ("Finder", "Finder is open."),
    ("VS Code", "VS Code is open."),
    ("Safari", "Safari is open."),
    ("Spotify", "Spotify is open."),
    ("Terminal", "Terminal is open."),
    ("Notion", "Notion is open."),
    ("Xcode", "Xcode is open."),
    ("Discord", "Discord is open."),
    ("Mail", "Mail is open."),
    ("Photos", "Photos is open."),
    ("Messages", "Messages is open."),
    ("Preview", "Preview is open."),
    ("Zoom", "Zoom is open."),
    ("Notes", "Notes is open."),
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

_WEATHER_CONDITIONS = [
    (72, "sunny", False),
    (65, "partly cloudy", False),
    (58, "windy", False),
    (80, "hot and humid", False),
    (55, "overcast", True),
    (62, "light rain", True),
    (45, "cold and clear", False),
    (70, "mild and breezy", False),
    (88, "very hot", False),
    (50, "foggy", True),
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

_NEWS_HEADLINES = [
    "Fed holds interest rates steady amid inflation concerns",
    "Apple announces new MacBook lineup at WWDC",
    "Scientists report breakthrough in battery technology",
    "Global markets rally on strong jobs report",
    "New climate agreement signed by 40 nations",
    "OpenAI releases new model with improved reasoning",
    "Google updates search with AI-powered summaries",
    "Electric vehicle sales hit record high in Q1",
    "SpaceX launches next Starlink batch successfully",
    "Major cybersecurity breach affects thousands of accounts",
    "Housing market shows signs of cooling",
    "Nvidia reports record quarterly revenue",
    "Remote work trends stabilising at hybrid model",
    "New study links screen time to sleep disruption",
    "Tech layoffs continue at major firms",
]

_STOCK_SYMBOLS = [
    "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "INTC",
    "NFLX", "DIS", "BA", "JPM", "V", "WMT", "KO", "PEP", "COST", "PLTR",
    "UBER", "SHOP", "CRM", "ORCL", "QCOM", "MU",
]

# templates for symbol-specific prompts — expanded across the full symbol list
# below so the model learns tool routing generalises to any ticker, not a fixed few
_STOCK_PROMPT_TEMPLATES = [
    "How's {s} doing?",
    "Is {s} up or down?",
    "What's {s} at?",
    "What's {s} trading at?",
    "Check {s} for me.",
    "How's {s} today?",
]

# (symbol, prompt) pairs — one prompt per symbol, cycling through templates
_STOCKS_SPECIFIC = [
    (sym, _STOCK_PROMPT_TEMPLATES[i % len(_STOCK_PROMPT_TEMPLATES)].format(s=sym))
    for i, sym in enumerate(_STOCK_SYMBOLS)
]

# prompts that don't reference a specific symbol
_STOCKS_GENERAL_PROMPTS = [
    "Check my watchlist.",
    "Is the market up today?",
    "How are my stocks?",
    "How's the market today?",
    "What are my stocks doing?",
    "Give me a market update.",
]


# ---------------------------------------------------------------------------
# Per-tool example builders
# ---------------------------------------------------------------------------

def _calendar() -> tuple[str, str]:
    prompt = random.choice(_CALENDAR_PROMPTS)
    n = random.randint(0, 3)
    if n == 0:
        result = "No events today"
        reply = "Your calendar is clear today."
    else:
        events = random.sample(_CALENDAR_EVENTS, n)
        result = ", ".join(events)
        if n == 1:
            reply = f"You have {events[0].lower()} today."
        else:
            head = ", ".join(e.lower() for e in events[:-1])
            reply = f"You have {head}, and {events[-1].lower()} today."
    return prompt, f"[CALL: calendar][RESULT]{result}[/RESULT]\n{reply}"


def _screen() -> tuple[str, str]:
    prompt = random.choice(_SCREEN_PROMPTS)
    result_text, reply = random.choice(_SCREEN_RESULTS)
    return prompt, f"[CALL: screen][RESULT]{result_text}[/RESULT]\n{reply}"


def _reminders() -> tuple[str, str]:
    prompt = random.choice(_REMINDERS_PROMPTS)
    n = random.randint(0, 4)
    if n == 0:
        result = "No reminders set"
        reply = "You don't have any reminders set right now."
    else:
        items = random.sample(_REMINDERS_ITEMS, n)
        result = ", ".join(items)
        if n == 1:
            reply = f"You have one reminder: {items[0].lower()}."
        else:
            head = ", ".join(i.lower() for i in items[:-1])
            reply = f"You have {n} reminders: {head}, and {items[-1].lower()}."
    return prompt, f"[CALL: reminders][RESULT]{result}[/RESULT]\n{reply}"


def _notes() -> tuple[str, str]:
    prompt = random.choice(_NOTES_PROMPTS)
    result_text, reply = random.choice(_NOTES_CONTENT)
    return prompt, f"[CALL: notes][RESULT]{result_text}[/RESULT]\n{reply}"


def _spotify() -> tuple[str, str]:
    prompt = random.choice(_SPOTIFY_PROMPTS)
    action = random.choice(["status", "pause", "skip", "volume_up", "volume_down"])
    track, artist = random.choice(_SPOTIFY_TRACKS)
    vol = random.randint(3, 10) * 10

    if action == "status":
        result = f"{track} by {artist}, volume {vol}%"
        reply = f"{track} by {artist} is playing at {vol}% volume."
    elif action == "pause":
        result = "Paused"
        reply = "Music paused."
    elif action == "skip":
        next_track, next_artist = random.choice(_SPOTIFY_TRACKS)
        result = f"Now playing: {next_track} by {next_artist}"
        reply = f"Skipped — now playing {next_track} by {next_artist}."
    elif action == "volume_up":
        result = f"Volume set to {vol}%"
        reply = f"Volume turned up to {vol}%."
    else:
        result = f"Volume set to {vol}%"
        reply = f"Volume turned down to {vol}%."

    return prompt, f"[CALL: spotify][RESULT]{result}[/RESULT]\n{reply}"


def _launcher() -> tuple[str, str]:
    app, reply = random.choice(_LAUNCHER_APPS)
    prompt = random.choice(_LAUNCHER_TEMPLATES).format(a=app)
    result = f"{app} launched"
    return prompt, f"[CALL: launcher][RESULT]{result}[/RESULT]\n{reply}"


def _weather() -> tuple[str, str]:
    prompt = random.choice(_WEATHER_PROMPTS)
    temp, condition, raining = random.choice(_WEATHER_CONDITIONS)
    result = f"{temp}°F, {condition}"

    if raining:
        reply = f"It's {temp}°F and {condition} — you might want an umbrella."
    elif temp < 55:
        reply = f"It's {temp}°F and {condition} — a jacket is a good idea."
    elif temp > 80:
        reply = f"It's {temp}°F and {condition} out today."
    else:
        reply = f"It's {temp}°F and {condition} outside."

    return prompt, f"[CALL: weather][RESULT]{result}[/RESULT]\n{reply}"


def _news() -> tuple[str, str]:
    prompt = random.choice(_NEWS_PROMPTS)
    n = random.randint(2, 4)
    headlines = random.sample(_NEWS_HEADLINES, n)
    result = "; ".join(headlines)
    items = "; ".join(h.lower() for h in headlines)
    reply = f"Here are today's top stories: {items}."
    return prompt, f"[CALL: news][RESULT]{result}[/RESULT]\n{reply}"


def _stock_entry(sym: str) -> tuple[str, str]:
    price = round(random.uniform(80, 900), 2)
    change = round(random.uniform(-3.5, 3.5), 1)
    direction = "+" if change >= 0 else ""
    trend = "up" if change >= 0 else "down"
    return f"{sym}: ${price} ({direction}{change}%)", f"{sym} is {trend} {abs(change)}% at ${price}"


def _stocks() -> tuple[str, str]:
    if random.random() < 0.5:
        # specific-symbol prompt — result must reference that symbol
        sym, prompt = random.choice(_STOCKS_SPECIFIC)
        result, reply = _stock_entry(sym)
    else:
        # general prompt — show 1-3 random symbols from watchlist
        prompt = random.choice(_STOCKS_GENERAL_PROMPTS)
        symbols = random.sample(_STOCK_SYMBOLS, random.randint(1, 3))
        entries, parts = zip(*[_stock_entry(s) for s in symbols])
        result = ", ".join(entries)
        reply = "; ".join(parts) + "."
    return prompt, f"[CALL: stocks][RESULT]{result}[/RESULT]\n{reply}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_BUILDERS = [_calendar, _screen, _reminders, _notes, _spotify,
             _launcher, _weather, _news, _stocks]


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
    parser.add_argument("--count", type=int, default=3000)
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
