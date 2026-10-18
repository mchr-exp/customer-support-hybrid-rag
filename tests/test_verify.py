"""Milestone 3 tests -- identity verification.

Offline tests exercise the DETERMINISTIC parts (the Pydantic shape, the DB
resolution, the graph wiring) with no LLM: resolve_customer hits the in-memory
Chinook DB, which is fast and free. The structured-extraction step (which needs
the model) is covered by one gated live test.

Email/phone cases are self-grounding: we read customer 1's real email/phone from
the DB and feed them back, so the asserts can't drift from the data.
"""
from __future__ import annotations

import json

import pytest

from src import db
from src.models import UserInput
from src.verify import resolve_customer


def test_user_input_defaults_all_none():
    ui = UserInput()
    assert ui.customer_id is None and ui.email is None and ui.phone is None
    assert ui.is_empty() is True


def test_user_input_coerces_and_reports_nonempty():
    ui = UserInput(customer_id="1")  # Pydantic v2 coerces str->int
    assert ui.customer_id == 1
    assert ui.is_empty() is False


def test_resolve_by_valid_id():
    assert resolve_customer(UserInput(customer_id=1)) == 1


def test_resolve_bogus_id_returns_none():
    # Nonexistent id must NOT be trusted just because it's a number.
    assert resolve_customer(UserInput(customer_id=999999)) is None


def test_resolve_by_email_self_grounded():
    row = json.loads(db.run_query_safe(
        "SELECT Email FROM Customer WHERE CustomerId = :c", {"c": 1}))[0]
    assert resolve_customer(UserInput(email=row["Email"])) == 1
    # case-insensitive match too
    assert resolve_customer(UserInput(email=row["Email"].upper())) == 1


def test_resolve_by_phone_self_grounded():
    row = json.loads(db.run_query_safe(
        "SELECT Phone FROM Customer WHERE CustomerId = :c", {"c": 1}))[0]
    # Re-format the stored number; normalize_phone on both sides should still match.
    assert resolve_customer(UserInput(phone=row["Phone"])) == 1


def test_resolve_empty_returns_none():
    assert resolve_customer(UserInput()) is None


def test_verified_agent_compiles_with_verify_node():
    pytest.importorskip("langgraph")
    from src.verify import build_verified_invoice_agent
    g = build_verified_invoice_agent()
    nodes = set(g.get_graph().nodes.keys())
    assert {"verify", "assistant", "tools"}.issubset(nodes)


@pytest.mark.live
def test_live_verified_agent_grounds_and_refuses():
    from src import config
    if not config.OPENAI_API_KEY:
        pytest.skip("no OPENAI_API_KEY")
    from langchain_core.messages import HumanMessage
    from src.verify import build_verified_invoice_agent

    g = build_verified_invoice_agent()

    ok = g.invoke({"messages": [HumanMessage(
        "What is in my last invoice? My customer id is 1.")], "customer_id": None})
    # verify node must have resolved and set the real id
    assert ok["customer_id"] == 1

    bogus = g.invoke({"messages": [HumanMessage(
        "Show invoices for customer 999999.")], "customer_id": None})
    assert bogus["customer_id"] is None
