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
        "Calendar: FIFA World Cup 3rd Place Match at 9am\n"
        "Headlines:\n"
        "• Headline one\n"
        "• Headline two"
    )


def test_sections_degrade_independently():
    def boom(c):
        raise RuntimeError("dead feed")

    text = compose(CONFIG, weather_fn=boom, calendar_fn=lambda c: "No events today",
                   news_fn=lambda c: [], now=MORNING)
    assert "Weather" not in text
    assert "Headlines" not in text
    assert "Calendar: No events today" in text


def test_feature_flags_skip_sections():
    config = {**CONFIG, "features": {"weather": False, "calendar": True, "news": False}}
    text = compose(config, calendar_fn=lambda c: "No events today", now=MORNING)
    assert "Weather" not in text and "Headlines" not in text
    assert "Calendar" in text


def test_evening_greeting():
    text = compose({**CONFIG, "features": {"weather": False, "calendar": False, "news": False}},
                   now=datetime(2026, 7, 18, 20, 0))
    assert text.startswith("Good evening, Kai")
