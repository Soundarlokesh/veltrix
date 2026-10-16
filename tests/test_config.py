from datetime import date
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, Settings, app_today


def test_defaults_point_at_data_folder(monkeypatch):
    for name in ["CORPUS_DIR", "MANIFEST_PATH", "DB_PATH", "VECTOR_DIR"]:
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env()
    assert settings.db_path == PROJECT_ROOT / "data" / "veltrix.db"
    assert settings.corpus_dir == PROJECT_ROOT / "data" / "corpus"
    assert settings.manifest_path == PROJECT_ROOT / "data" / "manifest.json"


def test_no_api_key_by_default(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert Settings.from_env().groq_api_key is None


def test_app_today_uses_env(monkeypatch):
    monkeypatch.setenv("APP_TODAY", "2026-09-15")
    assert app_today() == date(2026, 9, 15)


def test_app_today_rejects_bad_format(monkeypatch):
    monkeypatch.setenv("APP_TODAY", "15/09/2026")
    with pytest.raises(ValueError):
        app_today()


def test_env_overrides_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "x.db"))
    assert Settings.from_env().db_path == Path(tmp_path / "x.db")
