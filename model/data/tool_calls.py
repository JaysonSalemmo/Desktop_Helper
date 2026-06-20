import argparse
import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Data pools
# ---------------------------------------------------------------------------

_CALENDAR_PROMPTS = [
    "What's on my calendar today?",
    "What do I have scheduled today?",
    "Any meetings today?",
    "What are my events for today?",
    "What's my schedule look like?",
    "Do I have anything on today?",
    "What meetings do I have?",
    "Check my calendar.",
    "What's coming up today?",
    "Am I busy today?",
    "What's on my agenda?",
    "Any appointments today?",
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
    "What am I looking at?",
    "Describe my screen.",
    "What's currently on my display?",
    "What am I working on?",
    "Can you see my screen?",
    "What's open on my computer?",
    "What does my screen show?",
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
    "What do I need to remember today?",
    "Check my reminders.",
    "Any reminders for today?",
    "What's on my reminder list?",
    "Do I have any reminders set?",
    "Show me my reminders.",
    "What should I not forget today?",
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
    "Check my notes.",
    "What's in my notes?",
    "Show me today's notes.",
    "Did I write anything down?",
    "What are my notes?",
    "Pull up my notes.",
    "What have I noted recently?",
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
    "What song is this?",
    "What's on?",
    "Pause the music.",
    "Skip this song.",
    "Turn up the volume.",
    "Turn down the volume.",
    "Play something.",
    "What artist is this?",
    "Stop the music.",
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

_LAUNCHER_PROMPTS = [
    "Open Chrome.",
    "Launch Slack.",
    "Open the calculator.",
    "Start Figma.",
    "Open Finder.",
    "Launch VS Code.",
    "Open Safari.",
    "Start Spotify.",
    "Open Terminal.",
    "Launch Notion.",
    "Open Xcode.",
    "Start Discord.",
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
]

_WEATHER_PROMPTS = [
    "What's the weather like?",
    "How's the weather today?",
    "Will it rain today?",
    "Do I need a jacket?",
    "What's the temperature outside?",
    "Is it cold out?",
    "What's the forecast?",
    "Should I bring an umbrella?",
    "How hot is it today?",
    "What's the weather looking like?",
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
    "Any news today?",
    "What's happening in the world?",
    "Give me the headlines.",
    "What are the top stories?",
    "Any tech news?",
    "What's going on today?",
    "Catch me up on the news.",
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

# (symbol, prompt) pairs — prompt references the symbol explicitly
_STOCKS_SPECIFIC = [
    ("AAPL",  "How's AAPL doing?"),
    ("AAPL",  "Is AAPL up or down?"),
    ("NVDA",  "What's NVDA at?"),
    ("NVDA",  "How's NVDA today?"),
    ("MSFT",  "Check MSFT for me."),
    ("MSFT",  "How's MSFT doing?"),
    ("GOOGL", "What's GOOGL at?"),
    ("AMZN",  "How's AMZN doing?"),
    ("TSLA",  "What's TSLA trading at?"),
    ("META",  "How's META today?"),
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

_STOCK_SYMBOLS = ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]


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
    # match the prompt to the app so examples are coherent
    prompt = next(
        (p for p in _LAUNCHER_PROMPTS if app.lower() in p.lower()),
        f"Open {app}.",
    )
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
    parser.add_argument("--count", type=int, default=500)
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
