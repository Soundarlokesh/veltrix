"""Read-only lookup of the requester's identity, role and region from veltrix.db.

The database is the only source of role and region (TASK.pdf section 7.1). This module
opens the database read-only and never writes to it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app.policy.models import Requester


def get_requester(db_path: Path, employee_id: str) -> Requester | None:
    employee_id = (employee_id or "").strip()
    if not employee_id:
        return None
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        row = connection.execute(
            "SELECT employee_id, region, role FROM employees WHERE employee_id = ?",
            (employee_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    return Requester(employee_id=row[0], role=row[2], region=row[1])
