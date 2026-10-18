"""Tests for the Milestone 4 hand-built supervisor (router).

Offline tests (no key, deterministic, free) pin the wiring and the routing
prompt. Live tests prove ROUTING by inspecting which specialist handoff tool the
supervisor actually called for each query.
"""
import os

import pytest

import src.config  # noqa: F401  -- importing triggers load_dotenv() so .env key loads
from src.prompts import SUPERVISOR_SYSTEM_PROMPT
from src.supervisor import build_supervisor

_HAS_KEY = bool(os.getenv("OPENAI_API_KEY"))
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------- offline: wiring + prompt (no LLM) ----------

def test_supervisor_compiles_with_expected_nodes():
    g = build_supervisor()
    nodes = set(g.get_graph().nodes.keys())
    assert {"supervisor", "tools"} <= nodes


def test_supervisor_prompt_names_both_specialists_and_refusal():
    p = SUPERVISOR_SYSTEM_PROMPT
    assert "music_specialist" in p
    assert "invoice_specialist" in p
    # It must instruct refusal of off-topic rather than answering it.
    assert "do not call any specialist" in p.lower() or "not call any specialist" in p.lower()


# ---------- live: routing (needs a real key) ----------

def _handoffs(messages):
    """Collect the names of every tool the supervisor handed off to."""
    names = []
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            names.append(tc["name"])
    return names


@pytest.mark.live
@pytest.mark.skipif(not _HAS_KEY, reason="no OPENAI_API_KEY")
def test_music_query_routes_to_music_specialist():
    from langchain_core.messages import HumanMessage

    g = build_supervisor()
    out = g.invoke(
        {"messages": [HumanMessage(content="What rock albums do you have?")],
         "customer_id": None}
    )
    assert "music_specialist" in _handoffs(out["messages"])


@pytest.mark.live
@pytest.mark.skipif(not _HAS_KEY, reason="no OPENAI_API_KEY")
def test_invoice_query_routes_to_invoice_specialist():
    from langchain_core.messages import HumanMessage

    g = build_supervisor()
    out = g.invoke(
        {"messages": [HumanMessage(content="What invoices does customer 1 have?")],
         "customer_id": None}
    )
    assert "invoice_specialist" in _handoffs(out["messages"])


@pytest.mark.live
@pytest.mark.skipif(not _HAS_KEY, reason="no OPENAI_API_KEY")
def test_offtopic_is_refused_without_any_handoff():
    from langchain_core.messages import HumanMessage

    g = build_supervisor()
    out = g.invoke(
        {"messages": [HumanMessage(content="What is the weather in Seattle?")],
         "customer_id": None}
    )
    # No specialist was invoked at all...
    assert _handoffs(out["messages"]) == []
    # ...and the refusal points the customer back to what it CAN do.
    answer = out["messages"][-1].content.lower()
    assert "music" in answer or "invoice" in answer
