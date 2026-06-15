"""
Screen capture module.
Captures the screen and passes a description to the local model via the tool dispatcher.
"""
import pyautogui
from pathlib import Path

SCREENSHOT_PATH = Path(__file__).parent.parent.parent / "data" / "screenshot.png"


def capture() -> Path:
    """Take a screenshot and save it. Returns the file path."""
    SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    screenshot = pyautogui.screenshot()
    screenshot.save(SCREENSHOT_PATH)
    return SCREENSHOT_PATH


def describe() -> str:
    """Capture screen and return a text description for the model to respond to.

    Phase 4: wire this to an OCR pass or a lightweight vision encoder so the
    local transformer receives a text representation of what's on screen.
    """
    raise NotImplementedError("Screen capture tool not yet wired to local model.")
