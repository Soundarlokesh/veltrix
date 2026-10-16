"""SQLite connections. Read-only for assessment tables unless a write is explicitly requested."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    return connection


def connect_writable(db_path: Path) -> sqlite3.Connection:
    """Autocommit mode: callers start explicit transactions with BEGIN IMMEDIATE."""
    connection = sqlite3.connect(Path(db_path), timeout=5, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection
