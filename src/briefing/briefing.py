"""
Morning briefing — weather + today's calendar, when there's something to say.

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


_EMPTY_AGENDA = ("no events", "nothing scheduled", "access not granted")


def _is_empty_agenda(agenda: str) -> bool:
    low = agenda.strip().lower()
    return not low or any(low.startswith(p) for p in _EMPTY_AGENDA)


def compose_sections(config: dict, weather_fn=_default_weather,
                     calendar_fn=_default_calendar, news_fn=None,
                     now: datetime | None = None) -> list[str]:
    """The briefing as separate sections (greeting, weather, and today's
    calendar only if it has events). `news_fn` is accepted and ignored — the
    headlines section was removed; the parameter stays so existing callers and
    tests don't break."""
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
            agenda = calendar_fn(config)
            # only when there IS something — "No events today" is noise, and
            # headlines were dropped entirely (2026-07-27, Kai): the briefing
            # should be a glance, not a wall of text
            if agenda and not _is_empty_agenda(agenda):
                sections.append(f"Calendar: {agenda}")
        except Exception:
            pass

    return sections


def compose(config: dict, **kwargs) -> str:
    """The briefing as one text block (TUI, tests, logs)."""
    return "\n".join(compose_sections(config, **kwargs))
