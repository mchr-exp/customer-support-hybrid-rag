"""Milestone 6 offline wiring tests for the verification UI.

These prove the structure and the pause-detection helpers WITHOUT an API key:
they never call graph.invoke (which would need the LLM). The live pause/resume
path is exercised by the Milestone 3 HITL live test and the manual browser demo.
"""
import gradio as gr

import src.hitl_ui as hui


def test_graph_has_checkpointer_and_interrupt_node():
    g = hui._graph
    assert g.checkpointer is not None               # required for interrupt/resume
    assert "ask_identity" in g.get_graph().nodes      # the node that pauses


def test_fresh_thread_is_not_paused():
    cfg = {"configurable": {"thread_id": "test-fresh-thread"}}
    assert hui._is_paused(cfg) is False               # nothing parked yet


def test_status_ready_on_fresh_thread():
    assert "Ready" in hui._status("test-ready-thread")
    assert "Ready" in hui._status(None)               # pre-load (no thread yet)


def test_empty_message_is_passthrough_no_invoke():
    hist = [{"role": "user", "content": "hi"}]
    new_hist, box, status = hui.respond("   ", hist, "test-empty")
    assert new_hist == hist                           # unchanged -> no graph.invoke
    assert box == ""
    assert isinstance(status, str)


def test_build_hitl_ui_returns_blocks():
    assert isinstance(hui.build_hitl_ui(), gr.Blocks)
