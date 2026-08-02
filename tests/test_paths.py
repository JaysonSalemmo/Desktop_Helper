from pathlib import Path

from src import paths


def test_from_source_resolves_to_project_root():
    # not frozen: resource and user-data both anchor at the repo root, so every
    # existing path resolves exactly as before this module existed
    assert not paths.is_frozen()
    root = Path(__file__).resolve().parent.parent
    assert paths.resource_dir() == root
    assert paths.user_data_dir() == root
    assert paths.resource_path("model", "hf_tokenizer") == root / "model" / "hf_tokenizer"
    assert paths.user_data_path("config.json") == root / "config.json"


def test_frozen_splits_resource_and_user_data(monkeypatch, tmp_path):
    monkeypatch.setattr(paths.sys, "frozen", True, raising=False)
    monkeypatch.setattr(paths.sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    monkeypatch.setattr(paths.Path, "home", classmethod(lambda cls: tmp_path / "home"))

    # resources come from the unpacked bundle
    assert paths.resource_dir() == tmp_path / "bundle"
    # user data goes to Application Support, and the dir is created
    expected = tmp_path / "home" / "Library" / "Application Support" / "Buddy"
    assert paths.user_data_dir() == expected
    assert expected.is_dir()
