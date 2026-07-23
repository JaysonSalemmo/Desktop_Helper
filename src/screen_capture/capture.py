"""
Screen capture module.
Captures the screen (or finds the user's own screenshots) and turns what's
visible into text the local model can describe.
"""
import subprocess

import pyautogui
from pathlib import Path

SCREENSHOT_PATH = Path(__file__).parent.parent.parent / "data" / "screenshot.png"

# cap on the OCR snippet handed to the model: the context window is 1024
# tokens and the prompt + reply need their share; ~700 chars ≈ 190 tokens.
# generous on purpose — the model DESCRIBES this text (reprompt path), it
# doesn't recite it, and a description needs enough context to say what the
# screen is actually about
_MAX_SNIPPET_CHARS = 700

# macOS names user screenshots "Screenshot <date> at <time>.png" (Ventura+)
# or "Screen Shot … .png" (older); both live in the configured capture folder
_SCREENSHOT_GLOBS = ("Screenshot*.png", "Screen Shot*.png")


def capture() -> Path:
    """Take a screenshot and save it. Returns the file path."""
    SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    screenshot = pyautogui.screenshot()
    screenshot.save(SCREENSHOT_PATH)
    return SCREENSHOT_PATH


def ocr(image_path: Path) -> list[str]:
    """Recognized text lines from an image via Apple's Vision framework.

    Fully on-device (no cloud), like everything else in this project. Returns
    lines in Vision's reading order; empty list on any failure.
    """
    from Foundation import NSURL
    import Vision

    url = NSURL.fileURLWithPath_(str(image_path))
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    # accurate (not fast): screenshots are full of small UI text, and a one-off
    # request per user question can afford the extra second
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)

    success, _error = handler.performRequests_error_([request], None)
    if not success:
        return []
    lines = []
    for observation in request.results() or []:
        candidates = observation.topCandidates_(1)
        if candidates and len(candidates):
            lines.append(str(candidates[0].string()))
    return lines


def _snippet(lines: list[str]) -> str:
    """Compress OCR lines into one bounded string: strip, drop 1-char UI noise
    (menu bar glyphs, single icons), dedupe preserving order, truncate at a
    word boundary."""
    seen = set()
    kept = []
    for line in lines:
        line = line.strip()
        if len(line) <= 1 or line in seen:
            continue
        seen.add(line)
        kept.append(line)
    text = "; ".join(kept)
    if len(text) > _MAX_SNIPPET_CHARS:
        text = text[:_MAX_SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
    return text


def _compose(front_name: str | None, others: list[str], lines: list[str]) -> str:
    if front_name is None:
        base = "Open apps: " + ", ".join(others) if others else "No apps open"
    elif not others:
        base = f"{front_name} in front"
    else:
        base = f"{front_name} in front, also open: {', '.join(others)}"

    text = _snippet(lines)
    return f"{base}. On screen: {text}" if text else base


def screenshots_dir() -> Path:
    """Where the user's own Cmd+Shift screenshots land: the folder configured
    in macOS (defaults read com.apple.screencapture location), else Desktop."""
    try:
        out = subprocess.run(
            ["defaults", "read", "com.apple.screencapture", "location"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            path = Path(out.stdout.strip()).expanduser()
            if path.is_dir():
                return path
    except Exception:
        pass  # unset key / sandbox weirdness → the macOS default
    return Path.home() / "Desktop"


def latest_screenshot() -> Path | None:
    """Newest screenshot the user took themselves, or None."""
    folder = screenshots_dir()
    candidates = [p for pattern in _SCREENSHOT_GLOBS for p in folder.glob(pattern)]
    return max(candidates, key=lambda p: p.stat().st_mtime, default=None)


def describe_image(image_path: Path) -> str:
    """OCR a specific image (e.g. the user's own screenshot) into description
    material for the model."""
    image_path = Path(image_path)
    text = _snippet(ocr(image_path))
    if not text:
        return f"The screenshot {image_path.name} has no readable text"
    return f"Text visible in the screenshot {image_path.name}: {text}"


def describe() -> str:
    """Text description of what's on screen for the model to respond to.

    Frontmost + open apps via NSWorkspace (no permission needed), plus the
    screen's visible text via screenshot + local OCR. OCR needs the Screen
    Recording permission — until it's granted (or if Vision fails) the app
    list still goes through, so the tool never comes back empty.
    """
    from AppKit import NSApplicationActivationPolicyRegular, NSWorkspace

    workspace = NSWorkspace.sharedWorkspace()
    front = workspace.frontmostApplication()
    front_name = str(front.localizedName()) if front else None

    others = sorted(
        str(app.localizedName())
        for app in workspace.runningApplications()
        if app.activationPolicy() == NSApplicationActivationPolicyRegular
        and str(app.localizedName()) != front_name
    )

    try:
        lines = ocr(capture())
    except Exception:
        lines = []  # a broken screenshot shouldn't take the app list with it

    return _compose(front_name, others, lines)
