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
    """Text description of what's on screen for the model to respond to.

    Reports the frontmost app and other visible apps via NSWorkspace — real
    data with no screen-recording permission needed. A true visual description
    (OCR / vision encoder over capture()) remains future work.
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

    if front_name is None:
        return "Open apps: " + ", ".join(others) if others else "No apps open"
    if not others:
        return f"{front_name} in front"
    return f"{front_name} in front, also open: {', '.join(others)}"
