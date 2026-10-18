"""Milestone 7 -- ONE chat UI over the unified graph.

Wraps build_unified_agent() (verify + interrupt + memory + supervisor routing to
music/invoice) behind a single window, so a single conversation does the whole
journey: identify -> browse the catalog -> look up invoices -> state a preference
-> have it remembered -> recall it next session.

Same bridge shape as the Milestone 6 verification UI: `respond()` uses the graph's
`.next` to decide invoke-fresh vs Command(resume=...), then detects whether this
turn paused (__interrupt__) or produced an answer. The status line is derived
fresh from the checkpoint every turn and also surfaces what memory holds.
"""
from __future__ import annotations

import uuid

import gradio as gr
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.memory import read_preferences
from src.unified import build_unified_agent

# One compiled graph (checkpointer + store) shared by all sessions. State is
# partitioned by thread_id; memory is partitioned by verified customer_id.
_graph = build_unified_agent()


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _is_paused(config) -> bool:
    """Non-empty `.next` means a node is still pending (parked at the interrupt)."""
    return bool(_graph.get_state(config).next)


_READY = (
    "💬 Ready. Tell me your customer id, email, or phone to verify, then ask "
    "about music or invoices."
)


def _status(thread_id: str | None) -> str:
    """Status line derived from the graph's current checkpoint for this thread."""
    if not thread_id:
        return _READY
    snap = _graph.get_state(_config(thread_id))
    if snap.next:  # parked at the identity interrupt
        return "🔒 **Awaiting verification** — reply with a valid customer id, email, or phone."
    cid = (snap.values or {}).get("customer_id")
    if cid:
        prefs = read_preferences(_graph.store, cid)
        tail = f" · remembered: {', '.join(prefs)}" if prefs else ""
        return f"✅ **Verified** — customer {cid}. Ask about music or invoices.{tail}"
    return _READY


def respond(message: str, history: list, thread_id: str):
    """Bridge: decide invoke-vs-resume from the graph's pause state, run it, then
    detect whether it paused again or produced an answer."""
    message = (message or "").strip()
    if not message:
        return history, "", _status(thread_id)

    config = _config(thread_id)

    # (1) DECIDE: parked at the interrupt -> this message resumes it; else fresh.
    if _is_paused(config):
        out = _graph.invoke(Command(resume=message), config)
    else:
        out = _graph.invoke(
            {"messages": [HumanMessage(content=message)], "customer_id": None},
            config,
        )

    # (2) DETECT: a pause surfaces as __interrupt__; otherwise take the answer.
    intr = out.get("__interrupt__")
    reply = intr[0].value if intr else out["messages"][-1].content

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return history, "", _status(thread_id)


def _new_session() -> str:
    """Fresh thread_id per page load and on reset (isolates the checkpointer slot).

    Note: memory is keyed by the VERIFIED customer_id, not the thread_id, so a
    reset starts a clean conversation but the SAME customer's saved preferences
    are recalled once they verify again -- that is the cross-thread memory point.
    """
    return str(uuid.uuid4())


def build_unified_ui() -> gr.Blocks:
    """Build (do not launch) the unified customer-support chat UI."""
    with gr.Blocks(title="Customer Support — Unified Agent") as demo:
        gr.Markdown(
            "## 🎧 Customer Support (unified)\n"
            "One assistant for the **music catalog** and your **invoices**. I verify "
            "your identity once, then remember your music tastes across sessions.\n\n"
            "Try: *\"what rock albums do you have?\"* → I'll ask you to verify → "
            "*\"my customer id is 1\"* → then *\"what's in my last invoice?\"* and "
            "*\"by the way I love jazz\"* (I'll remember it)."
        )
        status = gr.Markdown(_status(None))
        thread_id = gr.State()
        chatbot = gr.Chatbot(height=440, label="Conversation")
        with gr.Row():
            box = gr.Textbox(
                placeholder="Ask about music or invoices…",
                scale=8,
                show_label=False,
                autofocus=True,
            )
            send = gr.Button("Send", variant="primary", scale=1)
        reset = gr.Button("Reset conversation", variant="secondary")

        demo.load(_new_session, outputs=thread_id)
        send.click(respond, [box, chatbot, thread_id], [chatbot, box, status])
        box.submit(respond, [box, chatbot, thread_id], [chatbot, box, status])
        reset.click(
            lambda: ([], _new_session(), _status(None)),
            outputs=[chatbot, thread_id, status],
        )

    return demo
