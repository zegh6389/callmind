"""Config path anchoring (F18)."""

from __future__ import annotations

from pathlib import Path

from callmind.config import Settings


def test_default_paths_anchor_to_data_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("CALLMIND_MEMORY_DB_PATH", raising=False)
    monkeypatch.delenv("CALLMIND_KB_DIR", raising=False)
    data_dir = tmp_path / "callmind-data"
    monkeypatch.setenv("CALLMIND_DATA_DIR", str(data_dir))
    s = Settings()
    assert Path(s.memory_db_path).is_absolute()
    assert Path(s.memory_db_path).parent == data_dir
    assert Path(s.kb_dir).is_absolute()
    assert Path(s.kb_dir).parent == data_dir


def test_explicit_relative_path_anchored_too(monkeypatch, tmp_path):
    data_dir = tmp_path / "callmind-data"
    monkeypatch.setenv("CALLMIND_DATA_DIR", str(data_dir))
    monkeypatch.setenv("CALLMIND_MEMORY_DB_PATH", "custom.db")
    s = Settings()
    assert Path(s.memory_db_path) == data_dir / "custom.db"


def test_absolute_path_untouched(monkeypatch, tmp_path):
    abs_path = tmp_path / "own" / "callmind.db"
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CALLMIND_MEMORY_DB_PATH", str(abs_path))
    s = Settings()
    assert s.memory_db_path == str(abs_path)