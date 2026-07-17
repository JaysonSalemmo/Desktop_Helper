"""
macOS Reminders via the shared EventKit store (src/eventkit/store.py).

Result string matches the training format: "Call dentist, Buy groceries" or
"No reminders set".
"""
import threading

from EventKit import EKEntityTypeReminder

from src.eventkit.store import request_access

FETCH_TIMEOUT = 15


def format_reminders(titles: list[str]) -> str:
    return ", ".join(titles) if titles else "No reminders set"


def get_incomplete() -> list[str]:
    """Titles of all incomplete reminders, across all reminder lists."""
    store = request_access(EKEntityTypeReminder)
    predicate = store.predicateForIncompleteRemindersWithDueDateStarting_ending_calendars_(
        None, None, None
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


def incomplete_summary() -> str:
    return format_reminders(get_incomplete())
