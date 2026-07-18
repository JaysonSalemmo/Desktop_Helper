"""
Spotify control via AppleScript, plus Web API track search.

Playback is always local (AppleScript against the desktop app — no auth).
The Web API is used only for search (text → track URI), which works with the
client-credentials flow: just the app id/secret from config, no OAuth login.
"""
import base64
import subprocess
import time

import requests

TOKEN_URL = "https://accounts.spotify.com/api/token"
SEARCH_URL = "https://api.spotify.com/v1/search"
API_TIMEOUT = 10

# client-credentials token cache — tokens last an hour, searches are cheap
_token: dict = {"value": None, "expires": 0.0}

# osascript can block indefinitely (e.g. an unanswered Automation permission
# prompt, or Spotify mid-launch) — a hung tool call would freeze the whole
# assistant turn, so every call gets a hard timeout
TIMEOUT = 15


def _run(script: str) -> str:
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return ""
    return result.stdout.strip()


def is_running() -> bool:
    """True if the Spotify app is open. Checked before read-only queries so
    `tell application "Spotify"` doesn't auto-launch it just to answer
    "what's playing?" — only an explicit play should start the app."""
    return subprocess.run(["pgrep", "-x", "Spotify"], capture_output=True).returncode == 0


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
    if not is_running():
        return "Spotify isn't running"
    name = _run('tell application "Spotify" to name of current track')
    artist = _run('tell application "Spotify" to artist of current track')
    if name and artist:
        return f"{name} by {artist}"
    return "Nothing playing right now"


def set_volume(level: int) -> None:
    """Set Spotify volume 0–100."""
    level = max(0, min(100, level))
    _run(f'tell application "Spotify" to set sound volume to {level}')


def play_track(uri: str) -> None:
    """Play a specific track by Spotify URI (spotify:track:...)."""
    _run(f'tell application "Spotify" to play track "{uri}"')


def _api_token(client_id: str, client_secret: str) -> str:
    if _token["value"] and time.time() < _token["expires"]:
        return _token["value"]
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    resp = requests.post(
        TOKEN_URL,
        data={"grant_type": "client_credentials"},
        headers={"Authorization": f"Basic {auth}"},
        timeout=API_TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()
    _token["value"] = payload["access_token"]
    _token["expires"] = time.time() + payload.get("expires_in", 3600) - 60
    return _token["value"]


def search_track(query: str, client_id: str, client_secret: str) -> tuple[str, str] | None:
    """Top search hit for the query → (track URI, "Title by Artist"), or None."""
    token = _api_token(client_id, client_secret)
    resp = requests.get(
        SEARCH_URL,
        params={"q": query, "type": "track", "limit": 1},
        headers={"Authorization": f"Bearer {token}"},
        timeout=API_TIMEOUT,
    )
    resp.raise_for_status()
    items = resp.json().get("tracks", {}).get("items", [])
    if not items:
        return None
    track = items[0]
    artists = ", ".join(a["name"] for a in track["artists"])
    return track["uri"], f"{track['name']} by {artists}"
