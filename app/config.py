"""Runtime configuration, read from environment variables only.

No secrets have defaults. Paths default to the ./data folder so that
`docker compose up` works after ./data is replaced with the assessment copy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _today() -> date:
    raw = _env("APP_TODAY")
    if raw is None:
        return date.today()
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"APP_TODAY must be YYYY-MM-DD, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path
    manifest_path: Path
    db_path: Path
    vector_dir: Path
    llm_provider: str
    groq_model: str
    groq_api_key: str | None
    embedding_model: str
    tesseract_cmd: str | None
    log_level: str
    state_db: Path
    llm_timeout_seconds: float
    rag_top_k: int
    gemini_api_key: str | None
    gemini_model: str

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = PROJECT_ROOT / "data"
        return cls(
            corpus_dir=Path(_env("CORPUS_DIR", str(data_dir / "corpus"))),
            manifest_path=Path(_env("MANIFEST_PATH", str(data_dir / "manifest.json"))),
            db_path=Path(_env("DB_PATH", str(data_dir / "veltrix.db"))),
            vector_dir=Path(_env("VECTOR_DIR", str(data_dir / "index"))),
            llm_provider=_env("LLM_PROVIDER", "mock"),
            groq_model=_env("GROQ_MODEL", ""),
            groq_api_key=_env("GROQ_API_KEY"),
            embedding_model=_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            tesseract_cmd=_env("TESSERACT_CMD"),
            log_level=_env("LOG_LEVEL", "INFO"),
            state_db=Path(_env("STATE_DB", str(data_dir / "state.db"))),
            llm_timeout_seconds=float(_env("LLM_TIMEOUT_SECONDS", "20")),
            rag_top_k=int(_env("RAG_TOP_K", "12")),
            gemini_api_key=_env("GEMINI_API_KEY"),
            gemini_model=_env("GEMINI_MODEL", ""),
        )


def app_today() -> date:
    """Today's date (APP_TODAY if set, otherwise the system date)."""
    return _today()
