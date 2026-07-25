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
import subprocess
from pathlib import Path

HOME = Path.home()

# how long to let Spotlight run before giving up (indexed → normally instant)
_MDFIND_TIMEOUT = 5

# path components that mark a result as noise rather than a user document
_NOISE_DIRS = {"Library", "node_modules", "__pycache__", "site-packages",
               "venv", "DerivedData", ".build"}
# build/compiled artifacts — never what someone searches for by name
_NOISE_SUFFIXES = {".pyc", ".pyo", ".class", ".o", ".so"}


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


def search(query: str, max_results: int = 5) -> list[Path]:
    """Matching paths, Library noise removed, newest first, capped."""
    query = query.strip()
    if not query:
        return []
    matches = [p for p in _mdfind(query) if not _is_noise(p)]
    matches.sort(key=_mtime, reverse=True)
    return matches[:max_results]


def find(query: str, max_results: int = 5) -> str:
    """One display sentence for the dispatcher. `query` is the already-extracted
    search term (see tools.extract_file_query). Paths are facts — this reads
    back as-is (verbatim), the model never paraphrases it."""
    results = search(query, max_results)
    if not results:
        return f"No files found matching '{query}'"
    listed = ", ".join(f"{p.name} ({_human(p.parent)})" for p in results)
    noun = "file" if len(results) == 1 else "files"
    return f"Found {len(results)} {noun}: {listed}"


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
    removed, newest first, capped."""
    matches = [p for p in _mdfind_modified_since(days)
               if not _is_noise(p) and p.is_file()]
    matches.sort(key=_mtime, reverse=True)
    return matches[:max_results]


def recent(days: int, max_results: int = 5) -> str:
    """One display sentence: the files you worked on within the last `days`
    days. Time-based counterpart to find() (which searches by name)."""
    results = search_recent(days, max_results)
    window = _window_label(days)
    if not results:
        return f"No files worked on in the {window}"
    listed = ", ".join(f"{p.name} ({_human(p.parent)})" for p in results)
    noun = "file" if len(results) == 1 else "files"
    return f"{len(results)} {noun} from the {window}: {listed}"
