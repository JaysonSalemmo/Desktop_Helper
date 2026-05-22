"""
Spotify control via AppleScript (no API key required).
Controls the locally running Spotify desktop app.
"""
import subprocess


def _run(script: str) -> str:
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()


def play() -> None:
    _run('tell application "Spotify" to play')


def pause() -> None:
    _run('tell application "Spotify" to pause')


def next_track() -> None:
    _run('tell application "Spotify" to next track')


def previous_track() -> None:
    _run('tell application "Spotify" to previous track')


def current_track() -> str:
    """Return a string describing the currently playing track."""
    name = _run('tell application "Spotify" to name of current track')
    artist = _run('tell application "Spotify" to artist of current track')
    if name and artist:
        return f"{name} by {artist}"
    return "Nothing playing or Spotify is not running."


def set_volume(level: int) -> None:
    """Set Spotify volume 0–100."""
    level = max(0, min(100, level))
    _run(f'tell application "Spotify" to set sound volume to {level}')
