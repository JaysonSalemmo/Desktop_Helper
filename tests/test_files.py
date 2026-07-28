from pathlib import Path

from src.file_finder import files


def test_search_filters_library_and_sorts_by_mtime(tmp_path, monkeypatch):
    # three real files with controlled mtimes + one Library path that must drop
    old = tmp_path / "resume_old.pdf"
    new = tmp_path / "resume.pdf"
    lib = tmp_path / "Library" / "Caches" / "resume.pdf"
    lib.parent.mkdir(parents=True)
    for f in (old, new, lib):
        f.write_text("x")
    import os
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))

    monkeypatch.setattr(files, "_mdfind", lambda q: [old, lib, new])
    results = files.search("resume")
    assert results == [new, old]  # Library dropped, newest first


def test_search_respects_max_results(monkeypatch):
    paths = [Path(f"/Users/x/f{i}.txt") for i in range(10)]
    monkeypatch.setattr(files, "_mdfind", lambda q: paths)
    monkeypatch.setattr(files, "_mtime", lambda p: 0.0)
    assert len(files.search("f", max_results=3)) == 3


def test_search_empty_query_skips_spotlight(monkeypatch):
    called = []
    monkeypatch.setattr(files, "_mdfind", lambda q: called.append(q) or [])
    assert files.search("   ") == []
    assert called == []  # never shelled out on a blank term


def test_human_abbreviates_home(monkeypatch):
    monkeypatch.setattr(files, "HOME", Path("/Users/kai"))
    assert files._human(Path("/Users/kai/Documents")) == "~/Documents"
    assert files._human(Path("/tmp/elsewhere")) == "/tmp/elsewhere"


def test_find_formats_sentence(monkeypatch):
    monkeypatch.setattr(files, "HOME", Path("/Users/kai"))
    monkeypatch.setattr(files, "search", lambda q, n=5: [
        Path("/Users/kai/Documents/resume.pdf"),
        Path("/Users/kai/Desktop/resume_old.pdf"),
    ])
    out = files.find("resume")
    # one file per LINE, name first then location: a comma-run of
    # "name (path)" was hard to scan and overflowed the orb's message slot
    assert out == ("Found 2 files:\n"
                   "resume.pdf  ·  ~/Documents\n"
                   "resume_old.pdf  ·  ~/Desktop")


def test_find_singular_and_none(monkeypatch):
    monkeypatch.setattr(files, "HOME", Path("/Users/kai"))
    monkeypatch.setattr(files, "search",
                        lambda q, n=5: [Path("/Users/kai/Downloads/tax.pdf")])
    assert files.find("tax").startswith("Found 1 file:\ntax.pdf")

    monkeypatch.setattr(files, "search", lambda q, n=5: [])
    assert files.find("nope") == "No files found matching 'nope'"


def test_mdfind_bad_query_returns_empty():
    # real subprocess path: garbage that yields no results must not raise
    assert files._mdfind("\x00zzz-no-such-file-anywhere-zzz") == []


def test_recent_filters_folders_and_noise(tmp_path, monkeypatch):
    monkeypatch.setattr(files, "HOME", tmp_path)
    doc = tmp_path / "report.pdf"
    doc.write_text("x")
    folder = tmp_path / "somedir"
    folder.mkdir()
    lib = tmp_path / "Library" / "Caches"
    lib.mkdir(parents=True)
    libfile = lib / "cache.pdf"
    libfile.write_text("x")
    monkeypatch.setattr(files, "_mdfind_modified_since",
                        lambda days: [doc, folder, libfile])
    out = files.recent(7)
    assert "report.pdf" in out
    assert "somedir" not in out    # folders excluded (not a file)
    assert "cache.pdf" not in out  # Library noise excluded


def test_recent_none_and_window_labels(monkeypatch):
    monkeypatch.setattr(files, "_mdfind_modified_since", lambda days: [])
    # empty mdfind now falls back to the walk — pin it empty too
    monkeypatch.setattr(files, "_walk_search", lambda match, max_hits=200: [])
    assert files.recent(7) == "No files worked on in the last week"
    assert files.recent(14) == "No files worked on in the last two weeks"
    assert files.recent(5) == "No files worked on in the last 5 days"


def test_walk_fallback_when_spotlight_returns_nothing(tmp_path, monkeypatch):
    # the live failure: the file EXISTS but mdfind silently returns zero
    # (permission-filtered) — the walk must find it from the real disk
    docs = tmp_path / "Documents"
    docs.mkdir()
    resume = docs / "Kai_Villamor_Resume.pdf"
    resume.write_text("x")
    hidden = docs / ".hidden_resume.pdf"
    hidden.write_text("x")
    noise = docs / "node_modules"
    noise.mkdir()
    (noise / "resume.js").write_text("x")

    monkeypatch.setattr(files, "HOME", tmp_path)
    monkeypatch.setattr(files, "_mdfind", lambda q: [])  # Spotlight stonewalls

    out = files.find("resume")
    assert "Kai_Villamor_Resume.pdf" in out
    assert ".hidden_resume" not in out
    assert "resume.js" not in out


def test_walk_fallback_for_recent_files(tmp_path, monkeypatch):
    docs = tmp_path / "Documents"
    docs.mkdir()
    fresh = docs / "today.md"
    fresh.write_text("x")

    monkeypatch.setattr(files, "HOME", tmp_path)
    monkeypatch.setattr(files, "_mdfind_modified_since", lambda days: [])

    out = files.recent(7)
    assert "today.md" in out


def test_deep_paths_collapse_so_the_name_stays_readable(monkeypatch):
    # a full path like ~/Documents/Side Projects/Desktop_Helper/build/... wraps
    # over several lines and buries the filename it's meant to locate
    monkeypatch.setattr(files, "HOME", Path("/Users/kai"))
    assert files._short_dir(Path("/Users/kai/Documents")) == "~/Documents"
    assert files._short_dir(Path("/Users/kai/Documents/Work")) == "~/Documents/Work"
    assert files._short_dir(
        Path("/Users/kai/Documents/Side Projects/App/src/menubar")) == "…/menubar"


def test_build_output_is_not_a_document(monkeypatch):
    # "recent documents" was dominated by our own PyInstaller output
    monkeypatch.setattr(files, "HOME", Path("/Users/kai"))
    for junk in ("Documents/App/build/Desktop Helper/BUNDLE-00.toc",
                 "Documents/App/dist/DesktopHelper/_internal/version.py",
                 "Documents/App/build/COLLECT-00.toc"):
        assert files._is_noise(Path("/Users/kai") / junk), junk
    assert not files._is_noise(Path("/Users/kai/Documents/resume.pdf"))
