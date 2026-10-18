"""Milestone 6 -- surface the verification PAUSE in the browser.

Milestone 3 built pause/resume (interrupt() + checkpointer) but it was only driven from
Python. This UI makes the pause VISIBLE in Gradio: ask for an invoice with a bad
id, the graph pauses mid-run, the window shows a "waiting for verification" state,
and your next message RESUMES it -- looping until you verify.

The one new idea: detecting a paused graph FROM OUTSIDE.

    graph.get_state(config).next
        ()                 -> the graph finished this turn; show the answer.
        ('ask_identity',)  -> the graph is PARKED at that node (an interrupt);
                              show "waiting", and treat the next message as a resume.

`.next` is the external view of "did the graph stop mid-run?" We use it twice:
  1. BEFORE invoking, to decide invoke-fresh vs Command(resume=...).
  2. AFTER invoking, to decide "show answer" vs "show waiting".

This wraps the Milestone 3 HITL invoice graph (which already compiles WITH a MemorySaver),
not the Milestone 4 supervisor -- the supervisor has no verify/interrupt node.
"""
from __future__ import annotations

import uuid

import gradio as gr
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.hitl import build_hitl_invoice_agent

# One compiled, checkpointer-backed HITL graph shared by all sessions. State is
# partitioned by thread_id, so each browser session's pause/resume is isolated.
_graph = build_hitl_invoice_agent()


def _is_paused(config) -> bool:
    """True if this thread's graph is parked at an interrupt awaiting resume.

    This is the pause-detection idea: a non-empty `.next` means a node is still
    pending (the graph stopped mid-run at the interrupt); `()` means it finished.
    """
    return bool(_graph.get_state(config).next)


def _status(thread_id: str | None) -> str:
    """Human-readable status line derived from the graph's current checkpoint."""
    if not thread_id:
        return "💬 Ready. Ask about an invoice — I'll verify your identity first."
    config = {"configurable": {"thread_id": thread_id}}
    snap = _graph.get_state(config)
    if snap.next:  # parked at the interrupt
        return "🔒 **Awaiting verification** — reply with a valid customer id, email, or phone."
    cid = (snap.values or {}).get("customer_id")
    if cid:
        return f"✅ **Verified** — customer {cid}. Ask about your invoices."
    return "💬 Ready. Ask about an invoice — I'll verify your identity first."


def respond(message: str, history: list, thread_id: str):
    """Bridge: decide invoke-vs-resume by asking the graph if it's paused, run it,
    then detect whether it paused again or produced an answer."""
    message = (message or "").strip()
    if not message:
        return history, "", _status(thread_id)

    config = {"configurable": {"thread_id": thread_id}}

    # (1) DECIDE: if the graph is parked at the interrupt, this message is the
    # answer to resume with; otherwise it's a fresh question.
    if _is_paused(config):
        out = _graph.invoke(Command(resume=message), config)
    else:
        out = _graph.invoke(
            {"messages": [HumanMessage(content=message)], "customer_id": None},
            config,
        )

    # (2) DETECT: a pause surfaces as __interrupt__ in the output; otherwise the
    # last message is the grounded answer.
    intr = out.get("__interrupt__")
    reply = intr[0].value if intr else out["messages"][-1].content

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return history, "", _status(thread_id)


def _new_session() -> str:
    """Mint a fresh thread_id (per page load and on reset) for session isolation."""
    return str(uuid.uuid4())


def build_hitl_ui() -> gr.Blocks:
    """Build (do not launch) the verification chat UI."""
    with gr.Blocks(title="Invoice Assistant — Identity Verification") as demo:
        gr.Markdown(
            "## 🔐 Invoice Assistant (with identity verification)\n"
            "Ask about your invoices. If I can't verify the customer id/email/phone "
            "in your message, I'll **pause and ask** — then your next reply resumes "
            "the lookup. Try an invalid id first (e.g. *\"invoices for customer 99999\"*), "
            "then reply *\"actually my customer id is 1\"*."
        )
        status = gr.Markdown(_status(None))
        thread_id = gr.State()
        chatbot = gr.Chatbot(height=420, label="Conversation")
        with gr.Row():
            box = gr.Textbox(
                placeholder="Ask about an invoice…",
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
