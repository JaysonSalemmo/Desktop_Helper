"""
Quick note capture module.
Saves timestamped notes to a local JSON file.
"""
import json
from datetime import datetime
from pathlib import Path

NOTES_PATH = Path(__file__).parent.parent.parent / "data" / "notes.json"


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


def summarize() -> str:
    """Summarize today's notes — will be routed through the local model (Phase 4)."""
    raise NotImplementedError("Note summarization not yet wired to local model.")
