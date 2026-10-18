"""Milestone 6 — offline UI wiring tests (no LLM, no server, no key).

We do NOT test the model's answers here (that needs a key and is non-deterministic);
we test the deterministic bridge logic: the empty-message guard, fresh thread ids,
and that the Blocks app constructs.
"""
import gradio as gr

from src.ui import respond, _new_session, build_ui


def test_empty_message_is_a_noop_and_calls_no_model():
    # A blank/whitespace message must return history unchanged and clear the box,
    # WITHOUT invoking the graph (so this test needs no key and spends nothing).
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    out_history, box = respond("   ", history, "t-test")
    assert out_history == history   # unchanged
    assert box == ""                # textbox cleared


def test_new_session_ids_are_unique():
    ids = {_new_session() for _ in range(100)}
    assert len(ids) == 100          # every session gets its own thread_id


def test_build_ui_constructs_a_blocks_app():
    demo = build_ui()
    assert isinstance(demo, gr.Blocks)
