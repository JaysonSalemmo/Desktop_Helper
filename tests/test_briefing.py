from datetime import datetime

from src.briefing.briefing import compose

CONFIG = {
    "user": {"name": "Kai"},
    "features": {"weather": True, "calendar": True, "news": True},
}
MORNING = datetime(2026, 7, 18, 8, 30)


def test_full_briefing():
    text = compose(
        CONFIG,
        weather_fn=lambda c: "73°F, heavy rain",
        calendar_fn=lambda c: "FIFA World Cup 3rd Place Match at 9am",
        news_fn=lambda c: ["Headline one", "Headline two"],
        now=MORNING,
    )
    assert text == (
        "Good morning, Kai — Saturday, July 18.\n"
        "Weather: 73°F, heavy rain\n"
        "Calendar: FIFA World Cup 3rd Place Match at 9am"
    )  # headlines removed 2026-07-27 — see test_headlines_are_gone


def test_sections_degrade_independently():
    def boom(c):
        raise RuntimeError("dead feed")

    text = compose(CONFIG, weather_fn=boom,
                   calendar_fn=lambda c: "Standup at 9am", now=MORNING)
    assert "Weather" not in text          # the dead section drops out alone…
    assert "Calendar: Standup at 9am" in text  # …and the rest still renders
    assert text.startswith("Good morning, Kai")


def test_feature_flags_skip_sections():
    config = {**CONFIG, "features": {"weather": False, "calendar": True, "news": False}}
    text = compose(config, calendar_fn=lambda c: "Standup at 9am", now=MORNING)
    assert "Weather" not in text
    assert "Calendar" in text


def test_calendar_section_only_appears_when_there_are_events():
    # Kai 2026-07-27: an empty agenda is noise in a glanceable briefing
    config = {**CONFIG, "features": {"weather": False, "calendar": True}}
    assert "Calendar" in compose(config, calendar_fn=lambda c: "Standup at 9am",
                                 now=MORNING)
    for empty in ("No events today", "", "access not granted"):
        text = compose(config, calendar_fn=lambda c, e=empty: e, now=MORNING)
        assert "Calendar" not in text, f"{empty!r} should be omitted"


def test_headlines_are_gone():
    # the news section was removed — it dominated the orb's message slot
    config = {**CONFIG, "features": {"weather": False, "calendar": False,
                                     "news": True}}
    text = compose(config, now=MORNING)
    assert "Headlines" not in text


def test_evening_greeting():
    text = compose({**CONFIG, "features": {"weather": False, "calendar": False, "news": False}},
                   now=datetime(2026, 7, 18, 20, 0))
    assert text.startswith("Good evening, Kai")
