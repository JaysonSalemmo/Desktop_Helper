"""
macOS Reminders via the shared EventKit store (src/eventkit/store.py).

Supports reading and creating reminders, targeting a specific list, and setting
a natural-language due date. The user's lists are discovered at runtime (never
hardcoded), so list-targeting adapts to whatever lists they actually have.

Result string matches the training format: "Call dentist, Buy groceries" or
"No reminders set".
"""
import re
import threading

from EventKit import EKEntityTypeReminder

from src.eventkit.store import request_access

FETCH_TIMEOUT = 15


def format_reminders(titles: list[str]) -> str:
    return ", ".join(titles) if titles else "No reminders set"


def list_names() -> list[str]:
    """Titles of the user's reminder lists."""
    store = request_access(EKEntityTypeReminder)
    return [str(c.title()) for c in store.calendarsForEntityType_(EKEntityTypeReminder)]


def _find_calendar(store, name: str | None):
    """The reminder list whose title best matches `name` (case-insensitive
    substring), or None. Longest title wins so a specific name isn't shadowed."""
    if not name:
        return None
    low = name.lower()
    matches = [c for c in store.calendarsForEntityType_(EKEntityTypeReminder)
               if low in str(c.title()).lower()]
    return max(matches, key=lambda c: len(str(c.title()))) if matches else None


def _components_from(nsdate):
    from Foundation import (NSCalendar, NSCalendarUnitDay, NSCalendarUnitHour,
                            NSCalendarUnitMinute, NSCalendarUnitMonth,
                            NSCalendarUnitYear)
    units = (NSCalendarUnitYear | NSCalendarUnitMonth | NSCalendarUnitDay
             | NSCalendarUnitHour | NSCalendarUnitMinute)
    return NSCalendar.currentCalendar().components_fromDate_(units, nsdate)


# "in an hour", "in 30 minutes", "in 2 days" — relative durations NSDataDetector
# does NOT recognize, so they're parsed here instead
_RELATIVE_RE = re.compile(
    r"\bin\s+(a|an|\d+)\s+(minutes?|mins?|hours?|hrs?|days?|weeks?)\b", re.I)
_RELATIVE_KWARG = {"minute": "minutes", "min": "minutes", "hour": "hours",
                   "hr": "hours", "day": "days", "week": "weeks"}


def _parse_relative(text: str):
    from datetime import datetime, timedelta

    from Foundation import NSDate
    m = _RELATIVE_RE.search(text)
    if not m:
        return None, None
    n = 1 if m.group(1).lower() in ("a", "an") else int(m.group(1))
    kwarg = _RELATIVE_KWARG.get(m.group(2).lower().rstrip("s"))
    if kwarg is None:
        return None, None
    due = datetime.now() + timedelta(**{kwarg: n})
    return _components_from(NSDate.dateWithTimeIntervalSince1970_(due.timestamp())), m.group(0)


def parse_due(text: str):
    """(dueDateComponents, matched_text) for a natural-language date/time in
    `text`, else (None, None).

    Relative durations first ("in an hour" → now + offset, exact time), since
    NSDataDetector either misses them or defaults them to noon. Then
    NSDataDetector (the system parser that turns "tomorrow at 3pm" into a live
    date in Mail) for absolute-ish dates."""
    comps, matched = _parse_relative(text)
    if comps is not None:
        return comps, matched
    from Foundation import NSDataDetector, NSMakeRange, NSTextCheckingTypeDate
    detector, _err = NSDataDetector.dataDetectorWithTypes_error_(
        NSTextCheckingTypeDate, None)
    if detector is not None:
        matches = detector.matchesInString_options_range_(
            text, 0, NSMakeRange(0, len(text)))
        if matches and matches[0].date() is not None:
            m = matches[0]
            rng = m.range()
            return (_components_from(m.date()),
                    text[rng.location:rng.location + rng.length])
    return None, None


def get_incomplete(list_name: str | None = None) -> list[str]:
    """Incomplete reminder titles — all lists, or just one if `list_name` given."""
    store = request_access(EKEntityTypeReminder)
    cal = _find_calendar(store, list_name)
    calendars = [cal] if cal is not None else None
    predicate = store.predicateForIncompleteRemindersWithDueDateStarting_ending_calendars_(
        None, None, calendars
    )

    done = threading.Event()
    found: list[str] = []

    def callback(reminders):
        found.extend(str(r.title()) for r in (reminders or []) if r.title())
        done.set()

    store.fetchRemindersMatchingPredicate_completion_(predicate, callback)
    if not done.wait(timeout=FETCH_TIMEOUT):
        raise TimeoutError("Reminders fetch timed out")
    return found


def incomplete_summary(list_name: str | None = None) -> str:
    return format_reminders(get_incomplete(list_name))


def add(title: str, due=None, due_text: str | None = None,
        list_name: str | None = None) -> str:
    """Create an incomplete reminder. Optionally in a named list and/or with a
    due date. The store requests FULL (read+write) access, so no extra
    permission beyond reading."""
    from EventKit import EKReminder

    store = request_access(EKEntityTypeReminder)
    cal = _find_calendar(store, list_name)
    missing_list = bool(list_name) and cal is None
    if cal is None:
        cal = store.defaultCalendarForNewReminders()

    reminder = EKReminder.reminderWithEventStore_(store)
    reminder.setTitle_(title)
    reminder.setCalendar_(cal)
    if due is not None:
        reminder.setDueDateComponents_(due)

    ok, error = store.saveReminder_commit_error_(reminder, True, None)
    if not ok:
        return f"Couldn't save the reminder ({error})"

    where = "" if missing_list or not list_name else f" to your {cal.title()} list"
    when = f", due {due_text}" if due_text else ""
    tail = f" (no '{list_name}' list found — used your default)" if missing_list else ""
    return f"Reminder added{where}: {title}{when}{tail}"
