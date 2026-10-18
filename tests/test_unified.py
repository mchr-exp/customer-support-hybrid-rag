"""Milestone 7 offline wiring tests for the UNIFIED graph + UI.

Prove the structure and helpers WITHOUT an API key: never call graph.invoke
(which would hit the LLM). The live end-to-end journey is exercised by the
`python -m src.unified` demo and the manual browser walkthrough.
"""
import gradio as gr

import src.unified as uni
import src.unified_ui as uui


def test_unified_graph_has_all_nodes():
    g = uui._graph
    nodes = set(g.get_graph().nodes.keys())
    # the whole journey wired into one graph
    for n in ("verify", "ask_identity", "load_memory", "supervisor",
              "specialists", "create_memory"):
        assert n in nodes, f"missing node: {n}"


def test_unified_graph_has_both_persistence_layers():
    g = uui._graph
    assert g.checkpointer is not None   # per-thread, for interrupt/resume
    assert g.store is not None          # cross-thread, for memory


def test_route_after_verify_branches_on_customer_id():
    assert uni._route_after_verify({"customer_id": 1}) == "load_memory"
    assert uni._route_after_verify({"customer_id": None}) == "ask_identity"
    assert uni._route_after_verify({}) == "ask_identity"


def test_route_after_supervisor_branches_on_tool_calls():
    class _Msg:
        def __init__(self, tool_calls=None):
            self.tool_calls = tool_calls
    assert uni._route_after_supervisor(
        {"messages": [_Msg(tool_calls=[{"name": "music_specialist"}])]}
    ) == "specialists"
    assert uni._route_after_supervisor(
        {"messages": [_Msg(tool_calls=None)]}
    ) == "create_memory"


def test_fresh_thread_not_paused_and_ready():
    cfg = {"configurable": {"thread_id": "uni-fresh"}}
    assert uui._is_paused(cfg) is False
    assert "Ready" in uui._status("uni-ready")
    assert "Ready" in uui._status(None)


def test_empty_message_is_passthrough_no_invoke():
    hist = [{"role": "user", "content": "hi"}]
    new_hist, box, status = uui.respond("   ", hist, "uni-empty")
    assert new_hist == hist      # unchanged -> no graph.invoke happened
    assert box == ""
    assert isinstance(status, str)


def test_build_unified_ui_returns_blocks():
    assert isinstance(uui.build_unified_ui(), gr.Blocks)
