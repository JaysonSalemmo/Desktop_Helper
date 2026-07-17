"""
App launcher — opens apps from the `allowed_apps` list in config.json.

Only apps explicitly listed in config can be launched; the user's message is
matched against the configured names, never used to build a path directly.
"""
import subprocess


def match_app(message: str, allowed_apps: list[dict]) -> dict | None:
    """Find the configured app whose name appears in the message.

    Longest name wins so "VS Code" beats a hypothetical "Code" entry.
    """
    lower = message.lower()
    candidates = [app for app in allowed_apps if app["name"].lower() in lower]
    if not candidates:
        return None
    return max(candidates, key=lambda app: len(app["name"]))


def launch(app: dict) -> str:
    subprocess.run(["open", app["path"]], check=True, capture_output=True)
    return f"{app['name']} launched"
