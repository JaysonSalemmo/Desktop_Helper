"""
File finder — locate the user's files by name via macOS Spotlight (mdfind).

Read-only and index-free: Spotlight already maintains the index, so a query is
a single `mdfind` call. Two deliberate choices shape the results:

- **Scope home, drop the noise.** A bare `mdfind -name resume` over the whole
  home floods with library/cache/dependency files that happen to contain the
  term — ~/Library, hidden dirs, `node_modules`, `venv`, `__pycache__`, build
  artifacts (`.pyc` and friends). The user means a document, so those are
  filtered out.
- **Most-recently-modified first.** "Find my resume" almost always means the
  copy you touched last, not the oldest draft — so matches are sorted by mtime
  and the freshest few are surfaced.

Coverage is Spotlight's: `mdfind` only returns what the index holds, so a file
in a folder excluded from indexing (some dev trees are) won't appear.
"""
import os
import subprocess
import time
from pathlib import Path

HOME = Path.home()

# how long to let Spotlight run before giving up (indexed → normally instant)
_MDFIND_TIMEOUT = 5

# path components that mark a result as noise rather than a user document
_NOISE_DIRS = {"Library", "node_modules", "__pycache__", "site-packages",
               "venv", "DerivedData", ".build",
               # our own PyInstaller output dominated "recent documents" with
               # BUNDLE-00.toc / COLLECT-00.toc and bundled site-packages
               "build", "dist", ".venv", ".git", "Caches"}
# build/compiled artifacts — never what someone searches for by name
_NOISE_SUFFIXES = {".pyc", ".pyo", ".class", ".o", ".so", ".toc", ".pkg",
                   ".dylib", ".egg-info"}


def _is_noise(path: Path) -> bool:
    parts = path.parts
    if any(part in _NOISE_DIRS for part in parts):
        return True
    # any hidden directory/file in the path (~/.git, ~/.venv, ~/.Trash, …)
    if any(part.startswith(".") for part in parts):
        return True
    return path.suffix.lower() in _NOISE_SUFFIXES


def _mdfind(query: str) -> list[Path]:
    """Raw Spotlight filename matches under HOME. List form (no shell) so a
    query with spaces or shell metacharacters is passed literally, never
    interpreted. Any failure → empty list; file search never crashes a turn."""
    try:
        out = subprocess.run(
            ["mdfind", "-onlyin", str(HOME), "-name", query],
            capture_output=True, text=True, timeout=_MDFIND_TIMEOUT,
        )
    except Exception:
        return []
    if out.returncode != 0:
        return []
    return [Path(line) for line in out.stdout.splitlines() if line]


def _short_dir(path: Path) -> str:
    """A location you can read at a glance.

    Full paths are the enemy of a one-glance answer: a result list of
    `~/Documents/Side Projects/Desktop_Helper/build/Desktop Helper` entries
    wraps over several lines each and buries the filename. Shallow paths are
    shown whole; deeper ones collapse to `…/<folder>`, which is what you
    actually need to tell two copies apart."""
    shown = _human(path)
    parts = shown.split("/")
    if len(parts) <= 3:           # "~", "Documents", "Work"
        return shown
    return "…/" + parts[-1]


def _format(paths: list[Path]) -> str:
    """One file per line: name first, location second."""
    return "\n".join(f"{p.name}  ·  {_short_dir(p.parent)}" for p in paths)


def _human(path: Path) -> str:
    """~-abbreviated path for display: /Users/kai/Documents → ~/Documents."""
    try:
        return "~/" + str(path.relative_to(HOME))
    except ValueError:
        return str(path)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0  # vanished between the index and the stat — sort it last


# -- direct filesystem walk: the Spotlight-independent ground truth -----------
# mdfind results are filtered by the caller's file-access permissions, and the
# filtering is SILENT — the frozen app got zero results for a file that exists,
# with no permission prompt ever shown (mdfind never touches the folders, so
# TCC never asks). Walking the user-document dirs directly both finds the files
# AND fires the proper "Desktop Helper would like to access…" prompt once.
_WALK_DIRS = ("Documents", "Desktop", "Downloads")
_WALK_DEPTH = 5        # levels below each root — documents live shallow
_WALK_MAX_DIRS = 4000  # runaway guard for pathological trees
_WALK_TIME_BUDGET = 8  # seconds — a reply beats an exhaustive scan


def _walk_search(match, max_hits: int = 200) -> list[Path]:
    """Files under the user-document dirs for which `match(Path) -> bool`,
    bounded in depth, breadth, and time. Noise-pruned like the mdfind path."""
    hits: list[Path] = []
    visited = 0
    deadline = time.monotonic() + _WALK_TIME_BUDGET
    for base in _WALK_DIRS:
        root = HOME / base
        if not root.is_dir():
            continue
        base_depth = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root):
            visited += 1
            if visited > _WALK_MAX_DIRS or time.monotonic() > deadline:
                return hits
            here = Path(dirpath)
            if len(here.parts) - base_depth >= _WALK_DEPTH:
                dirnames[:] = []  # depth cap: don't descend further
            else:
                dirnames[:] = [d for d in dirnames
                               if not d.startswith(".") and d not in _NOISE_DIRS]
            for name in filenames:
                path = here / name
                if not _is_noise(path) and match(path):
                    hits.append(path)
                    if len(hits) >= max_hits:
                        return hits
    return hits


def search(query: str, max_results: int = 5) -> list[Path]:
    """Matching paths, Library noise removed, newest first, capped.

    Spotlight first (indexed → instant, covers all of home); if it returns
    NOTHING, fall back to the direct walk — an empty mdfind is
    indistinguishable from silent permission filtering, so every empty
    answer gets verified against the real filesystem."""
    query = query.strip()
    if not query:
        return []
    matches = [p for p in _mdfind(query) if not _is_noise(p)]
    if not matches:
        q = query.lower()
        matches = _walk_search(lambda p: q in p.name.lower())
    matches.sort(key=_mtime, reverse=True)
    return matches[:max_results]


def find(query: str, max_results: int = 5) -> str:
    """One display sentence for the dispatcher. `query` is the already-extracted
    search term (see tools.extract_file_query). Paths are facts — this reads
    back as-is (verbatim), the model never paraphrases it."""
    results = search(query, max_results)
    if not results:
        return f"No files found matching '{query}'"
    noun = "file" if len(results) == 1 else "files"
    return f"Found {len(results)} {noun}:\n{_format(results)}"


def _mdfind_modified_since(days: int) -> list[Path]:
    """Paths whose content was modified within the last `days` days (Spotlight
    date query — the raw expression form, not -name)."""
    seconds = int(days * 86400)
    try:
        out = subprocess.run(
            ["mdfind", "-onlyin", str(HOME),
             f"kMDItemContentModificationDate >= $time.now(-{seconds})"],
            capture_output=True, text=True, timeout=_MDFIND_TIMEOUT,
        )
    except Exception:
        return []
    if out.returncode != 0:
        return []
    return [Path(line) for line in out.stdout.splitlines() if line]


def _window_label(days: int) -> str:
    return {1: "last day", 2: "last couple days", 7: "last week",
            14: "last two weeks", 30: "last month"}.get(days, f"last {days} days")


def search_recent(days: int, max_results: int = 5) -> list[Path]:
    """Actual files (not folders) modified in the last `days` days, noise
    removed, newest first, capped. Same Spotlight-then-walk policy as
    search(): an empty mdfind answer is verified against the real disk."""
    matches = [p for p in _mdfind_modified_since(days)
               if not _is_noise(p) and p.is_file()]
    if not matches:
        cutoff = time.time() - days * 86400
        matches = _walk_search(lambda p: _mtime(p) >= cutoff)
    matches.sort(key=_mtime, reverse=True)
    return matches[:max_results]


def recent(days: int, max_results: int = 5) -> str:
    """One display sentence: the files you worked on within the last `days`
    days. Time-based counterpart to find() (which searches by name)."""
    results = search_recent(days, max_results)
    window = _window_label(days)
    if not results:
        return f"No files worked on in the {window}"
    noun = "file" if len(results) == 1 else "files"
    return f"{len(results)} {noun} from the {window}:\n{_format(results)}"
