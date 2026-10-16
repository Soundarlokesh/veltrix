"""Which access levels each role may read (TASK.pdf section 3.1).

Unknown roles read nothing (fail closed).
"""

from __future__ import annotations

ROLE_READABLE_ACCESS: dict[str, frozenset[str]] = {
    "employee": frozenset({"all"}),
    "manager": frozenset({"all", "managers"}),
    "hr": frozenset({"all", "managers", "hr"}),
}


def can_read(role: str, access_level: str) -> bool:
    return access_level in ROLE_READABLE_ACCESS.get(role, frozenset())
