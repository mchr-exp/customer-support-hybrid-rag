"""Milestone 1 tests: database utilities (11+ tests)."""
import json
from src import db


def test_customer_count_is_59():
    assert json.loads(db.run_query_safe("SELECT COUNT(*) AS n FROM Customer"))[0]["n"] == 59


def test_parameterized_query_filters():
    out = json.loads(db.run_query_safe(
        "SELECT FirstName FROM Customer WHERE CustomerId = :cid", {"cid": 1}))
    assert len(out) == 1 and out[0]["FirstName"]


def test_empty_results_return_empty_json():
    assert db.run_query_safe(
        "SELECT * FROM Customer WHERE CustomerId = :cid", {"cid": -1}) == "[]"


def test_output_is_valid_json():
    json.loads(db.run_query_safe("SELECT * FROM Genre"))  # raises if invalid


def test_parameterized_blocks_injection():
    # classic injection payload is treated as a literal value, not SQL
    out = json.loads(db.run_query_safe(
        "SELECT * FROM Customer WHERE LastName = :n", {"n": "x'; DROP TABLE Customer;--"}))
    assert out == []
    # table still intact
    assert json.loads(db.run_query_safe("SELECT COUNT(*) AS n FROM Customer"))[0]["n"] == 59


def test_normalize_phone_international():
    assert db.normalize_phone("+1 (555) 123-4567") == "+15551234567"


def test_normalize_phone_domestic_dashes():
    assert db.normalize_phone("555-123-4567") == "5551234567"


def test_normalize_phone_parens_spaces():
    assert db.normalize_phone("(555) 123 4567") == "5551234567"


def test_normalize_phone_empty():
    assert db.normalize_phone("") == ""


def test_normalize_phone_none():
    assert db.normalize_phone(None) == ""


def test_normalize_phone_preserves_leading_plus():
    assert db.normalize_phone("+44 20 7946 0958").startswith("+")


def test_verify_database_healthy():
    h = db.verify_database()
    assert h["healthy"] is True


def test_verify_database_table_count():
    h = db.verify_database()
    assert h["table_count"] >= 11 and h["customer_count"] == 59
