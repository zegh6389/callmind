"""Config path anchoring (F18)."""

from __future__ import annotations

from pathlib import Path

from callmind.config import Settings


def test_default_paths_anchor_to_data_dir(monkeypatch):
    monkeypatch.delenv("CALLMIND_MEMORY_DB_PATH", raising=False)
    monkeypatch.delenv("CALLMIND_KB_DIR", raising=False)
    monkeypatch.setenv("CALLMIND_DATA_DIR", "C:/callmind-data")
    s = Settings()
    assert Path(s.memory_db_path).is_absolute()
    assert Path(s.memory_db_path).parent == Path("C:/callmind-data")
    assert Path(s.kb_dir).is_absolute()
    assert Path(s.kb_dir).parent == Path("C:/callmind-data")


def test_explicit_relative_path_anchored_too(monkeypatch):
    monkeypatch.setenv("CALLMIND_DATA_DIR", "C:/callmind-data")
    monkeypatch.setenv("CALLMIND_MEMORY_DB_PATH", "custom.db")
    s = Settings()
    assert Path(s.memory_db_path) == Path("C:/callmind-data/custom.db")


def test_absolute_path_untouched(monkeypatch):
    monkeypatch.setenv("CALLMIND_MEMORY_DB_PATH", "C:/my/own/callmind.db")
    s = Settings()
    assert s.memory_db_path == "C:/my/own/callmind.db"