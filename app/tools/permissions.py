"""Who may see whose leave data (TASK.pdf sections 3.1 and 6.1). Used by the tools and by the router.

- Employee: own data only.
- Manager: own data and direct reports' data (direct reports only, not indirect).
- HR: one employee at a time, any employee.
- get_team_leave: managers only.
Unknown targets get the same refusal as forbidden ones, so the response reveals nothing.
"""

from __future__ import annotations

import sqlite3

from app.db import leave_repo
from app.policy.models import Requester

FIXED_REFUSAL = "I can't share information about that employee."
BULK_REFUSAL = "I can't share leave data for several employees at once."
BULK_TOKENS = frozenset({"all", "everyone", "everybody", "*", "all employees"})


def _exists(conn: sqlite3.Connection, employee_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM employees WHERE employee_id = ?", (employee_id,)).fetchone()
    return row is not None


def can_view_employee(conn: sqlite3.Connection, viewer: Requester, target_id: str) -> bool:
    if target_id == viewer.employee_id:
        return True
    if viewer.role == "manager":
        return target_id in leave_repo.direct_reports(conn, viewer.employee_id)
    if viewer.role == "hr":
        return _exists(conn, target_id)
    return False


def can_view_team(viewer: Requester) -> bool:
    return viewer.role == "manager"
