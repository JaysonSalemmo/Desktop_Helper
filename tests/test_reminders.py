from datetime import datetime, timedelta


def test_parse_due_relative_durations():
    """Relative durations NSDataDetector misses ('in an hour') must resolve to
    now + offset, at the exact time."""
    from src.reminders import reminders

    for phrase, delta in (("in an hour", timedelta(hours=1)),
                          ("in 30 minutes", timedelta(minutes=30)),
                          ("in 2 hours", timedelta(hours=2)),
                          ("in 3 days", timedelta(days=3))):
        comps, matched = reminders.parse_due(phrase)
        assert comps is not None, phrase
        assert matched.lower() in phrase.lower()
        expected = datetime.now() + delta
        assert comps.hour() == expected.hour  # exact time, not noon


def test_parse_due_absolute_and_none():
    from src.reminders import reminders

    comps, matched = reminders.parse_due("tomorrow at 3pm")
    assert comps is not None and comps.hour() == 15
    assert reminders.parse_due("buy groceries") == (None, None)
