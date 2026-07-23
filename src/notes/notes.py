"""
Quick note capture module.
Saves timestamped notes to a local JSON file.
"""
import json
from datetime import datetime

from src.paths import user_data_path

# writable per-user data (Application Support when frozen, project dir from
# source) — a project-relative path would write into the read-only .app bundle
NOTES_PATH = user_data_path("data", "notes.json")


def _load() -> list:
    if not NOTES_PATH.exists():
        return []
    with open(NOTES_PATH) as f:
        return json.load(f)


def _save(notes: list) -> None:
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTES_PATH, "w") as f:
        json.dump(notes, f, indent=4)


def add(text: str) -> None:
    """Save a new timestamped note."""
    notes = _load()
    notes.append({"timestamp": datetime.now().isoformat(), "text": text})
    _save(notes)


def get_today() -> list[dict]:
    """Return all notes from today."""
    today = datetime.now().date().isoformat()
    return [n for n in _load() if n["timestamp"].startswith(today)]
