"""
Morning briefing — weather + today's calendar + headlines in one message.

Pure composition from the live tool backends: no model involved, so it's
fast and word-perfect by construction. Sections degrade independently — a
dead feed or missing permission drops or annotates its own line without
taking down the rest.

The *_fn params exist for tests; production callers just pass config.
"""
from datetime import datetime


def _default_weather(config: dict) -> str:
    from src.weather import weather
    return weather.current(config["weather"]["location"])


def _default_calendar(config: dict) -> str:
    from src.calendar_integration import events  # EventKit import deferred
    from src.eventkit.store import AccessDenied
    try:
        return events.today()
    except AccessDenied:
        return "access not granted"


def _default_news(config: dict) -> list[str]:
    from src.news import news
    return news.headlines(config["news"]["rss_feeds"],
                          max_headlines=3)


def _greeting(name: str, now: datetime) -> str:
    if now.hour < 12:
        part = "morning"
    elif now.hour < 18:
        part = "afternoon"
    else:
        part = "evening"
    date = f"{now:%A}, {now:%B} {now.day}"
    return f"Good {part}, {name} — {date}."


def compose_sections(config: dict, weather_fn=_default_weather,
                     calendar_fn=_default_calendar, news_fn=_default_news,
                     now: datetime | None = None) -> list[str]:
    """The briefing as separate sections (greeting, weather, calendar,
    headlines) — the chat panel shows each as its own bubble."""
    now = now or datetime.now()
    flags = config.get("features", {})
    sections = [_greeting(config["user"]["name"], now)]

    if flags.get("weather", True):
        try:
            sections.append(f"Weather: {weather_fn(config)}")
        except Exception:
            pass  # a dead section shouldn't kill the briefing

    if flags.get("calendar", True):
        try:
            sections.append(f"Calendar: {calendar_fn(config)}")
        except Exception:
            pass

    if flags.get("news", True):
        try:
            headlines = news_fn(config)
            if headlines:
                sections.append("Headlines:\n" + "\n".join(f"• {h}" for h in headlines))
        except Exception:
            pass

    return sections


def compose(config: dict, **kwargs) -> str:
    """The briefing as one text block (TUI, tests, logs)."""
    return "\n".join(compose_sections(config, **kwargs))
