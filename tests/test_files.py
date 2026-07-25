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
    assert out == ("Found 2 files: resume.pdf (~/Documents), "
                   "resume_old.pdf (~/Desktop)")


def test_find_singular_and_none(monkeypatch):
    monkeypatch.setattr(files, "HOME", Path("/Users/kai"))
    monkeypatch.setattr(files, "search",
                        lambda q, n=5: [Path("/Users/kai/Downloads/tax.pdf")])
    assert files.find("tax").startswith("Found 1 file: tax.pdf")

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
    assert files.recent(7) == "No files worked on in the last week"
    assert files.recent(14) == "No files worked on in the last two weeks"
    assert files.recent(5) == "No files worked on in the last 5 days"
