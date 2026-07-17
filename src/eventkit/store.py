"""
Shared EKEventStore for calendar + reminders (both live in EventKit).

One store instance per process — macOS ties permission grants to the store, and
calendar + reminders would otherwise each init their own. Access requests are
async in EventKit; `request_access` blocks on the callback so callers stay
synchronous. First use triggers the macOS permission prompt (attributed to the
hosting terminal app); a denial raises `AccessDenied` with instructions instead
of failing cryptically.
"""
import threading

from EventKit import EKEntityTypeEvent, EKEntityTypeReminder, EKEventStore

_store: EKEventStore | None = None
_store_lock = threading.Lock()
_granted: dict[int, bool] = {}  # entity type → already granted this process


class AccessDenied(RuntimeError):
    pass


def store() -> EKEventStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = EKEventStore.alloc().init()
        return _store


def request_access(entity_type: int) -> EKEventStore:
    """Ensure access for the entity type, returning the shared store."""
    s = store()
    if _granted.get(entity_type):
        return s

    done = threading.Event()
    outcome = {"granted": False}

    def callback(granted, error):
        outcome["granted"] = bool(granted)
        done.set()

    # macOS 14+ split the request API; fall back to the pre-14 one if absent
    if entity_type == EKEntityTypeEvent and hasattr(s, "requestFullAccessToEventsWithCompletion_"):
        s.requestFullAccessToEventsWithCompletion_(callback)
    elif entity_type == EKEntityTypeReminder and hasattr(s, "requestFullAccessToRemindersWithCompletion_"):
        s.requestFullAccessToRemindersWithCompletion_(callback)
    else:
        s.requestAccessToEntityType_completion_(entity_type, callback)

    kind = "Calendars" if entity_type == EKEntityTypeEvent else "Reminders"
    if not done.wait(timeout=60) or not outcome["granted"]:
        raise AccessDenied(
            f"{kind} access not granted — allow it under "
            f"System Settings → Privacy & Security → {kind}"
        )
    _granted[entity_type] = True
    return s
