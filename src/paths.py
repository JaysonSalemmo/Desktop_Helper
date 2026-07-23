"""
Path resolution that works both from source (`uv run`) and inside a frozen
PyInstaller bundle.

Two kinds of path, because a frozen .app is read-only and the 3.4GB checkpoint
is too big to embed:

- **Resource** paths — read-only assets shipped with the code (the tokenizer
  directory, the config template, the icon). From source they're project
  files; frozen they're unpacked under ``sys._MEIPASS``.
- **User-data** paths — writable per-user state (``config.json``, the model
  checkpoint). From source these stay in the project directory so dev behavior
  is unchanged; frozen they live in ``~/Library/Application Support/Desktop
  Helper``.

The design point: when NOT frozen, ``resource_dir()`` and ``user_data_dir()``
both return the project root, so every existing path resolves exactly as it did
before this module existed — nothing about running from source changes.
"""
import sys
from pathlib import Path

APP_NAME = "Desktop Helper"

# src/paths.py → parent is src/, its parent is the repo root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def resource_dir() -> Path:
    """Base directory for read-only bundled assets."""
    if is_frozen():
        return Path(sys._MEIPASS)  # PyInstaller's unpacked-resources dir
    return _PROJECT_ROOT


def resource_path(*parts: str) -> Path:
    """A read-only bundled asset, e.g. resource_path('model', 'hf_tokenizer')."""
    return resource_dir().joinpath(*parts)


def user_data_dir() -> Path:
    """Writable per-user directory (created if missing). Project root from
    source; Application Support when frozen."""
    if is_frozen():
        base = Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        base = _PROJECT_ROOT
    base.mkdir(parents=True, exist_ok=True)
    return base


def user_data_path(*parts: str) -> Path:
    """A writable per-user path, e.g. user_data_path('config.json')."""
    return user_data_dir().joinpath(*parts)
