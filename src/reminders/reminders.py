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


def _format_due(components) -> str | None:
    """Readable due string from a reminder's NSDateComponents, or None if it has
    no due date. Date-only reminders (no time set) omit the time so we don't
    surface a bogus '12:00 AM'."""
    if components is None:
        return None
    from Foundation import NSCalendar, NSDateFormatter
    date = NSCalendar.currentCalendar().dateFromComponents_(components)
    if date is None:
        return None
    # an unset hour comes back as NSDateComponentUndefined (a huge int), so a
    # plain 0..23 range check distinguishes a timed reminder from a date-only one
    has_time = 0 <= components.hour() <= 23
    fmt = NSDateFormatter.alloc().init()
    fmt.setDateFormat_("EEE MMM d 'at' h:mm a" if has_time else "EEE MMM d")
    return str(fmt.stringFromDate_(date))


def _fetch_raw(store, predicate) -> list:
    """EKReminder objects for a predicate (the async fetch, synchronized)."""
    done = threading.Event()
    found: list = []

    def callback(reminders):
        found.extend(reminders or [])
        done.set()

    store.fetchRemindersMatchingPredicate_completion_(predicate, callback)
    if not done.wait(timeout=FETCH_TIMEOUT):
        raise TimeoutError("Reminders fetch timed out")
    return [r for r in found if r.title()]


def _label(r) -> str:
    due = _format_due(r.dueDateComponents())
    return f"{r.title()} (due {due})" if due else str(r.title())


def _due_ts(r) -> float | None:
    """Due date as a unix timestamp, or None for no due date."""
    comps = r.dueDateComponents()
    if comps is None:
        return None
    from Foundation import NSCalendar
    date = NSCalendar.currentCalendar().dateFromComponents_(comps)
    return date.timeIntervalSince1970() if date is not None else None


def _calendars(store, list_name):
    cal = _find_calendar(store, list_name)
    return [cal] if cal is not None else None


def _title_matches(title, query: str) -> bool:
    """Every query word appears in the title (case-insensitive word subset).
    NOT substring: the extractor strips request words, so "delete the go for a
    run reminder" queries "go run" — which must still match "go for a run"."""
    t = str(title).lower()
    return all(w in t for w in query.lower().split())


def get_incomplete(list_name: str | None = None) -> list[str]:
    """Incomplete reminders — "title" or "title (due …)" — all lists, or just
    one if `list_name` given."""
    store = request_access(EKEntityTypeReminder)
    predicate = store.predicateForIncompleteRemindersWithDueDateStarting_ending_calendars_(
        None, None, _calendars(store, list_name)
    )
    return [_label(r) for r in _fetch_raw(store, predicate)]


def due_today_summary(list_name: str | None = None) -> str:
    """Only what's actually due TODAY — plus an overdue count so old clutter is
    visible without being listed as today's (the live complaint: "due today"
    answered with reminders from three different days)."""
    from datetime import datetime, timedelta

    from Foundation import NSDate

    store = request_access(EKEntityTypeReminder)
    cals = _calendars(store, list_name)
    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    ns = lambda dt: NSDate.dateWithTimeIntervalSince1970_(dt.timestamp())

    today = _fetch_raw(store, store.predicateForIncompleteRemindersWithDueDateStarting_ending_calendars_(
        ns(start), ns(start + timedelta(days=1)), cals))
    overdue = _fetch_raw(store, store.predicateForIncompleteRemindersWithDueDateStarting_ending_calendars_(
        NSDate.distantPast(), ns(start), cals))

    if not today and not overdue:
        return "Nothing due today"
    core = (f"Due today: {', '.join(_label(r) for r in today)}"
            if today else "Nothing due today")
    if overdue:
        n = len(overdue)
        core += f" — plus {n} overdue reminder{'s' if n > 1 else ''}"
    return core


def remove(title_query: str | None = None, only_overdue: bool = False,
           list_name: str | None = None) -> str:
    """Delete incomplete reminders by title match and/or overdue-ness. At least
    one filter must be present — a bare "delete my reminders" wiping everything
    is not a thing this does."""
    import time as _time

    if not title_query and not only_overdue:
        return "Tell me which reminders to delete — by name, or 'the overdue ones'"

    store = request_access(EKEntityTypeReminder)
    predicate = store.predicateForIncompleteRemindersWithDueDateStarting_ending_calendars_(
        None, None, _calendars(store, list_name))
    matches = _fetch_raw(store, predicate)
    if title_query:
        matches = [r for r in matches if _title_matches(r.title(), title_query)]
    if only_overdue:
        now = _time.time()
        matches = [r for r in matches
                   if (ts := _due_ts(r)) is not None and ts < now]
    if not matches:
        return "No matching reminders to delete"

    deleted = []
    for r in matches:
        label = _label(r)
        ok, _err = store.removeReminder_commit_error_(r, True, None)
        if ok:
            deleted.append(label)
    if not deleted:
        return "Couldn't delete those reminders"
    n = len(deleted)
    return f"Deleted {n} reminder{'s' if n > 1 else ''}: {', '.join(deleted)}"


def reschedule(title_query: str, due, due_text: str | None,
               list_name: str | None = None) -> str:
    """Move the due date of the reminder matching `title_query`. With several
    matches the oldest-due one moves (the usual intent: push the stale one),
    and the reply says how many were left alone."""
    store = request_access(EKEntityTypeReminder)
    predicate = store.predicateForIncompleteRemindersWithDueDateStarting_ending_calendars_(
        None, None, _calendars(store, list_name))
    matches = [r for r in _fetch_raw(store, predicate)
               if _title_matches(r.title(), title_query)]
    if not matches:
        return f"No reminder matching '{title_query}'"

    matches.sort(key=lambda r: ts if (ts := _due_ts(r)) is not None else float("inf"))
    target = matches[0]
    target.setDueDateComponents_(due)
    ok, error = store.saveReminder_commit_error_(target, True, None)
    if not ok:
        detail = error.localizedDescription() if error is not None else "unknown error"
        return f"Couldn't reschedule ({detail})"
    extra = ""
    if len(matches) > 1:
        others = len(matches) - 1
        extra = f" ({others} other match{'es' if others > 1 else ''} left unchanged)"
    return f"Rescheduled: {target.title()}, now due {due_text}{extra}"


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
    if cal is None:
        # defaultCalendarForNewReminders() legitimately returns nil (default
        # list in an unwritable account, fresh TCC grant, or none configured)
        # — seen live as EKErrorDomain Code=1 "No calendar has been set."
        # Fall back to the first writable list rather than failing the save.
        writable = [c for c in store.calendarsForEntityType_(EKEntityTypeReminder)
                    if c.allowsContentModifications()]
        # prefer the list actually named "Reminders" (the usual default) over
        # whatever EventKit happens to enumerate first
        cal = next((c for c in writable if str(c.title()).lower() == "reminders"),
                   writable[0] if writable else None)
    if cal is None:
        return ("Couldn't save the reminder — no writable Reminders list. "
                "Open the Reminders app and create a list first.")

    reminder = EKReminder.reminderWithEventStore_(store)
    reminder.setTitle_(title)
    reminder.setCalendar_(cal)
    if due is not None:
        reminder.setDueDateComponents_(due)

    ok, error = store.saveReminder_commit_error_(reminder, True, None)
    if not ok:
        # localizedDescription, not the raw NSError repr — the full
        # "Error Domain=EKErrorDomain Code=1 …" dump leaked to the UI once
        detail = error.localizedDescription() if error is not None else "unknown error"
        return f"Couldn't save the reminder ({detail})"

    where = "" if missing_list or not list_name else f" to your {cal.title()} list"
    when = f", due {due_text}" if due_text else ""
    tail = f" (no '{list_name}' list found — used your default)" if missing_list else ""
    return f"Reminder added{where}: {title}{when}{tail}"
