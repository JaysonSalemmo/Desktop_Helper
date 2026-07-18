"""
Today's calendar events via the shared EventKit store.

Result string matches the training format: "Standup at 9am, Team lunch at 12pm"
or "No events today".
"""
from datetime import datetime, time

from EventKit import EKEntityTypeEvent
from Foundation import NSDate

from src.eventkit.store import request_access


def format_time(dt: datetime) -> str:
    # "9am" / "9:30am" — the shapes the model saw in training
    hour = dt.strftime("%I").lstrip("0")
    suffix = dt.strftime("%p").lower()
    if dt.minute:
        return f"{hour}:{dt.minute:02d}{suffix}"
    return f"{hour}{suffix}"


def format_events(events: list[tuple[str, datetime | None]]) -> str:
    """events: (title, start time) pairs; start=None means all-day."""
    if not events:
        return "No events today"
    parts = []
    for title, start in events:
        parts.append(f"{title} (all day)" if start is None else f"{title} at {format_time(start)}")
    return ", ".join(parts)


def _nsdate(dt: datetime) -> NSDate:
    return NSDate.dateWithTimeIntervalSince1970_(dt.timestamp())


def today() -> str:
    store = request_access(EKEntityTypeEvent)
    now = datetime.now()
    start = _nsdate(datetime.combine(now.date(), time.min))
    end = _nsdate(datetime.combine(now.date(), time.max))

    predicate = store.predicateForEventsWithStartDate_endDate_calendars_(start, end, None)
    found = store.eventsMatchingPredicate_(predicate) or []

    events: list[tuple[str, datetime | None]] = []
    for ev in found:
        title = str(ev.title() or "Untitled")
        if ev.isAllDay():
            events.append((title, None))
        else:
            events.append((title, datetime.fromtimestamp(ev.startDate().timeIntervalSince1970())))
    events.sort(key=lambda e: (e[1] is not None, e[1] or datetime.min))
    return format_events(events)
