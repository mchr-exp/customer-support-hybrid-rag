"""Milestone 4 unit tests -- the music agent's state + graph WIRING.

These are deliberately OFFLINE: no API key, no network, no LLM call. We do not
unit-test the model's decision (non-deterministic + costs money); we test the
deterministic machinery around it:

  - the add_messages reducer appends (state.py)
  - the graph compiles with the expected nodes (agents.py)
  - tools_condition routes: tool_call -> "tools", plain reply -> END
  - ToolNode actually runs a tool, given a SYNTHETIC tool call (no LLM)
  - the grounding system prompt exists and forbids inventing facts

The live LLM behavior (does it CHOOSE the right tool?) is an integration
concern, gated on OPENAI_API_KEY -- see the key-gated test at the bottom, which
is skipped automatically when no key is present.
"""
import json
import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition

from src import tools
from src.agents import MUSIC_SYSTEM_PROMPT, build_music_agent
from src.state import AgentState


# ---------- state.py: the reducer ----------
def test_add_messages_appends_not_overwrites():
    left = [HumanMessage(content="first")]
    right = [HumanMessage(content="second")]
    merged = add_messages(left, right)
    assert [m.content for m in merged] == ["first", "second"]


def test_agent_state_shape():
    # TypedDict keys are a documentation/contract check.
    assert set(AgentState.__annotations__) == {"messages", "customer_id"}


# ---------- agents.py: graph structure ----------
def test_graph_compiles_with_expected_nodes():
    graph = build_music_agent()
    nodes = set(graph.get_graph().nodes.keys())
    # entry/exit sentinels + our two real nodes
    assert {"assistant", "tools"} <= nodes
    assert "__start__" in nodes and "__end__" in nodes


def test_system_prompt_enforces_grounding():
    p = MUSIC_SYSTEM_PROMPT.lower()
    assert "never" in p and "invent" in p          # anti-hallucination seed
    assert "tool" in p                             # instructs tool use


# ---------- the conditional edge: routing logic (no LLM) ----------
def test_tools_condition_routes_to_tools_on_tool_call():
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "get_albums_by_artist",
                     "args": {"artist": "AC/DC"}, "id": "1", "type": "tool_call"}],
    )
    assert tools_condition({"messages": [ai]}) == "tools"


def test_tools_condition_routes_to_end_on_plain_reply():
    ai = AIMessage(content="Here is your answer.")
    assert tools_condition({"messages": [ai]}) == END


# ---------- the tool node: real execution, synthetic call (no LLM) ----------
def test_tool_node_runs_real_tool_and_returns_grounded_data():
    from langgraph.prebuilt import ToolNode

    node = ToolNode(tools.lc_music_tools)
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "get_albums_by_artist",
                     "args": {"artist": "AC/DC"}, "id": "call_1", "type": "tool_call"}],
    )
    out = node.invoke({"messages": [ai]})
    tool_msg = out["messages"][-1]
    # The tool node appended the REAL tool output (a JSON string of albums).
    rows = json.loads(tool_msg.content)
    assert isinstance(rows, list) and len(rows) == 2        # Chinook: AC/DC has 2 albums
    assert any("AC/DC" in str(r.values()) for r in rows)


# ---------- live integration (key-gated, skipped by default) ----------
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"),
                    reason="live LLM test; needs OPENAI_API_KEY")
def test_live_agent_grounds_answer_in_tool_output():
    graph = build_music_agent()
    out = graph.invoke(
        {"messages": [HumanMessage(content="How many albums does AC/DC have?")],
         "customer_id": None})
    answer = out["messages"][-1].content
    assert "2" in answer                                     # grounded, correct


# ========== Milestone 4: invoice agent + the 6 grounding rules ==========
from src.invoice_agent import build_invoice_agent          # noqa: E402
from src.prompts import GROUNDING_RULES, INVOICE_SYSTEM_PROMPT  # noqa: E402
from src.agents import MUSIC_SYSTEM_PROMPT                 # noqa: E402


def test_invoice_agent_compiles():
    graph = build_invoice_agent()
    nodes = set(graph.get_graph().nodes.keys())
    # hand-built graph: same node names as the music agent
    assert {"assistant", "tools"} <= nodes
    assert "__start__" in nodes and "__end__" in nodes


def test_grounding_rules_shared_by_both_agents():
    # The exact same 6-rule block must appear in BOTH prompts (no drift).
    assert GROUNDING_RULES in MUSIC_SYSTEM_PROMPT
    assert GROUNDING_RULES in INVOICE_SYSTEM_PROMPT


def test_grounding_rules_cover_all_six():
    r = GROUNDING_RULES.lower()
    assert "only provide information returned by tool calls" in r   # 1
    assert "exact numbers" in r                                     # 2
    assert "empty results" in r                                     # 3
    assert "scope" in r and "defer" in r                            # 4
    assert "sampled or truncated" in r and "total count" in r       # 5
    assert "training data" in r and "call a tool first" in r        # 6


def test_invoice_prompt_pins_id_to_system_message():
    # Day-8 safety seed: id must come from a system message, never user text.
    p = INVOICE_SYSTEM_PROMPT.lower()
    assert "verified customer id" in p
    assert "system message" in p and "never" in p


def test_invoice_tool_node_runs_real_tool(engine_ignored=None):
    from langgraph.prebuilt import ToolNode

    node = ToolNode(tools.lc_invoice_tools)
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "get_invoices_by_customer_sorted_by_date",
                     "args": {"customer_id": "1"}, "id": "c1", "type": "tool_call"}],
    )
    out = node.invoke({"messages": [ai]})
    rows = json.loads(out["messages"][-1].content)
    assert isinstance(rows, list) and len(rows) > 0            # customer 1 has invoices


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"),
                    reason="live LLM test; needs OPENAI_API_KEY")
def test_live_invoice_agent_refuses_unknown_customer():
    from langchain_core.messages import SystemMessage

    graph = build_invoice_agent()
    out = graph.invoke({"messages": [
        SystemMessage(content="Verified customer ID: 999999"),
        HumanMessage(content="show my last invoice")]})
    # rule 3: empty -> says so, does not fabricate an invoice
    assert "999999" in out["messages"][-1].content or "no invoice" in out["messages"][-1].content.lower()
