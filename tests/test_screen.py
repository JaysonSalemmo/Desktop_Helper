import os
import time

from PIL import Image, ImageDraw, ImageFont

from src.screen_capture import capture
from src.screen_capture.capture import (_MAX_SNIPPET_CHARS, _compose, _snippet,
                                        describe_image, latest_screenshot, ocr)


def _render(tmp_path, text: str):
    # big dark text on white — easy mode for Vision, which is the point:
    # this tests our plumbing, not Apple's recognizer
    image = Image.new("RGB", (1200, 200), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 64)
    except OSError:
        font = ImageFont.load_default(64)
    draw.text((40, 60), text, fill="black", font=font)
    path = tmp_path / "rendered.png"
    image.save(path)
    return path


def test_ocr_reads_rendered_text(tmp_path):
    path = _render(tmp_path, "Meeting notes for Project Falcon")
    recognized = " ".join(ocr(path)).lower()
    assert "meeting notes" in recognized
    assert "falcon" in recognized


def test_ocr_missing_file_returns_empty():
    assert ocr("/nonexistent/nope.png") == []


def test_snippet_dedupes_and_drops_noise():
    lines = ["  OK  ", "OK", "x", "", "Inbox (3)", "Inbox (3)", "Reply all"]
    assert _snippet(lines) == "OK; Inbox (3); Reply all"


def test_snippet_caps_length_at_word_boundary():
    lines = [f"window title number {i} with several words" for i in range(30)]
    text = _snippet(lines)
    assert len(text) <= _MAX_SNIPPET_CHARS + 1  # +1 for the ellipsis
    assert text.endswith("…")
    assert not text[:-1].endswith(" ")  # cut at a word boundary, not mid-word


def test_latest_screenshot_picks_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "screenshots_dir", lambda: tmp_path)
    older = tmp_path / "Screenshot 2026-07-19 at 09.00.00.png"
    newer = tmp_path / "Screen Shot 2026-07-20 at 14.30.00.png"  # legacy name
    ignored = tmp_path / "vacation.png"  # not a screenshot name
    for f in (older, newer, ignored):
        f.touch()
    now = time.time()
    os.utime(older, (now - 100, now - 100))
    os.utime(newer, (now, now))
    assert latest_screenshot() == newer


def test_latest_screenshot_none_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "screenshots_dir", lambda: tmp_path)
    assert latest_screenshot() is None


def test_describe_image_reads_text_and_names_file(tmp_path):
    path = _render(tmp_path, "Quarterly budget review")
    result = describe_image(path)
    assert "rendered.png" in result
    assert "budget" in result.lower()


def test_describe_image_no_text(tmp_path):
    blank = tmp_path / "Screenshot blank.png"
    Image.new("RGB", (200, 200), "white").save(blank)
    assert "no readable text" in describe_image(blank)


def test_compose_with_and_without_text():
    assert _compose("Safari", ["Anki"], []) == "Safari in front, also open: Anki"
    assert _compose("Safari", [], ["Checkout", "Total $42.10"]) == \
        "Safari in front. On screen: Checkout; Total $42.10"
    assert _compose(None, [], []) == "No apps open"
    assert _compose(None, ["Anki", "Blender"], []) == "Open apps: Anki, Blender"
