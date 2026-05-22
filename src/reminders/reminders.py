"""
macOS Reminders integration via pyobjc EventKit.
Shares the same framework as calendar_integration.
"""

# TODO: implement using EventKit via pyobjc
# EventKit handles both Calendar and Reminders on macOS.
#
# Pattern:
#   from EventKit import EKEventStore, EKReminderPriority
#   store = EKEventStore.alloc().init()
#   store.requestAccessToEntityType_completion_(EKEntityTypeReminder, callback)
#
# Will require user to grant Reminders permission on first run (macOS prompt).


def get_incomplete() -> list[dict]:
    """Return all incomplete reminders."""
    raise NotImplementedError("Reminders integration not yet implemented.")


def add(title: str, due_date=None) -> None:
    """Create a new reminder."""
    raise NotImplementedError("Reminders integration not yet implemented.")


def complete(title: str) -> None:
    """Mark a reminder as complete by title."""
    raise NotImplementedError("Reminders integration not yet implemented.")
