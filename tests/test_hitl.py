"""Milestone 3 -- HITL interrupt/resume tests.

Offline tests pin the deterministic wiring (the routing edge, the compiled graph
shape, the mandatory checkpointer) with no LLM and no network. The live test runs
the real pause -> resume -> verified cycle and is gated behind OPENAI_API_KEY.
"""
import os

import pytest
from langchain_core.messages import HumanMessage

from src import hitl


# ----------------------------- offline wiring -----------------------------

def test_route_after_verify_sends_verified_to_assistant():
    assert hitl.route_after_verify({"customer_id": 1, "messages": []}) == "assistant"


def test_route_after_verify_sends_unverified_to_ask_identity():
    # None, 0, and a missing key all count as "not verified".
    assert hitl.route_after_verify({"customer_id": None, "messages": []}) == "ask_identity"
    assert hitl.route_after_verify({"messages": []}) == "ask_identity"


def test_graph_compiles_with_hitl_nodes():
    graph = hitl.build_hitl_invoice_agent()
    nodes = set(graph.get_graph().nodes.keys())
    # the HITL-specific pieces must be present alongside the usual loop
    assert {"verify", "ask_identity", "assistant", "tools"} <= nodes


def test_graph_has_checkpointer():
    # interrupt()/resume cannot work without a checkpointer.
    graph = hitl.build_hitl_invoice_agent()
    assert graph.checkpointer is not None


# ------------------------------- live cycle -------------------------------

@pytest.mark.live
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
def test_live_pause_then_resume_verifies():
    from langgraph.types import Command

    graph = hitl.build_hitl_invoice_agent()
    cfg = {"configurable": {"thread_id": "test-day9"}}

    # 1) bogus id -> the graph should PAUSE at ask_identity.
    out = graph.invoke(
        {"messages": [HumanMessage(content="Show me invoices for customer 99999.")],
         "customer_id": None},
        cfg,
    )
    assert out.get("__interrupt__"), "graph should interrupt on an unverifiable id"

    # 2) resume with a valid id -> loop back to verify -> customer_id resolved.
    out = graph.invoke(Command(resume="my customer id is 1"), cfg)
    assert out.get("customer_id") == 1
    assert out["messages"][-1].content  # a grounded final answer exists
