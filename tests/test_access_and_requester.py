import hashlib
import sqlite3

import pytest

from app.db.employees import get_requester
from app.policy.access import can_read
from tests.helpers import DATA_DIR, ORIGINAL_DB

WORKING_DB = DATA_DIR / "veltrix.db"


@pytest.mark.parametrize(
    "role, access, expected",
    [
        ("employee", "all", True),
        ("employee", "managers", False),
        ("employee", "hr", False),
        ("manager", "all", True),
        ("manager", "managers", True),
        ("manager", "hr", False),
        ("hr", "all", True),
        ("hr", "managers", True),
        ("hr", "hr", True),
        ("contractor", "all", False),  # unknown role reads nothing
        ("", "all", False),
    ],
)
def test_role_access_table(role, access, expected):
    assert can_read(role, access) is expected


@pytest.fixture
def db(corpus_available):
    if not WORKING_DB.is_file():
        pytest.skip("working database copy not present")
    return WORKING_DB


def test_requester_role_and_region_come_from_database(db):
    requester = get_requester(db, "E4005")
    assert requester.region == "DE"
    assert requester.role == "employee"


def test_manager_and_hr_roles_come_from_database(db):
    assert get_requester(db, "E2001").role == "manager"
    assert get_requester(db, "E2001").region == "IN"
    hr = get_requester(db, "E1001")
    assert hr.role == "hr"
    assert hr.region == "IN"


def test_unknown_or_blank_employee_is_none(db):
    assert get_requester(db, "E9999") is None
    assert get_requester(db, "") is None
    assert get_requester(db, "   ") is None


def test_lookup_is_read_only(db):
    # A write through the same read-only URI must fail.
    uri = f"file:{db.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("UPDATE employees SET role = 'hr' WHERE employee_id = 'E2004'")
    finally:
        connection.close()


def test_original_database_is_unchanged_after_lookups(db):
    if not ORIGINAL_DB.is_file():
        pytest.skip("original database not present")
    get_requester(db, "E4005")
    digest = hashlib.sha256(ORIGINAL_DB.read_bytes()).hexdigest()
    assert digest == "a2d151a58038e98e0b5dfa1fd4a906dc359e26c43d16b20cad4ec3e64408c4b5"
