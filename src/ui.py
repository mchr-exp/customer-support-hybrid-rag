"""Milestone 6 — Gradio chat UI.

Puts a browser chat window in front of the Milestone 4 supervisor. No new LangGraph
concepts: the only idea is SESSION ISOLATION. Each browser session gets its own
`thread_id` (a UUID in `gr.State`), passed in the graph config on every turn, so
two tabs hold independent conversations.

Why this needs a checkpointer: `thread_id` is the key the checkpointer files a
conversation under. We compile the supervisor WITH a MemorySaver here (the Milestone 4
graph is stateless by default), so LangGraph persists each session's history
server-side keyed by its thread_id. That is also why `respond()` sends only the
NEW message each turn — the checkpointer already holds the prior turns.
"""
from __future__ import annotations

import uuid

import gradio as gr
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.supervisor import build_supervisor

# One compiled, checkpointer-backed graph shared by all sessions. The MemorySaver
# partitions state by thread_id, so sharing the graph object does NOT share
# conversations — each thread_id is its own isolated slot.
_graph = build_supervisor(checkpointer=MemorySaver())


def respond(message: str, history: list, thread_id: str):
    """Bridge: Gradio text -> graph.invoke -> assistant text.

    `history` is the Chatbot's message list (role/content dicts). We only invoke
    the graph with the NEW message because the checkpointer, keyed by thread_id,
    already remembers the earlier turns of this session.
    """
    message = (message or "").strip()
    if not message:
        return history, ""
    config = {"configurable": {"thread_id": thread_id}}
    out = _graph.invoke(
        {"messages": [HumanMessage(content=message)], "customer_id": None},
        config,
    )
    reply = out["messages"][-1].content
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return history, ""


def _new_session() -> str:
    """Mint a fresh thread_id (called per page load, and on reset)."""
    return str(uuid.uuid4())


def build_ui() -> gr.Blocks:
    """Build (do not launch) the Gradio Blocks app."""
    with gr.Blocks(title="Customer Support — Music & Invoices") as demo:
        gr.Markdown(
            "## 🎧 Customer Support\n"
            "Ask about the **music catalog** (albums, artists, genres) or your "
            "**invoices** (provide a customer id, e.g. *\"invoices for customer 1\"*). "
            "Off-topic questions are politely refused."
        )
        # Per-session state: each browser session gets its own thread_id, so the
        # checkpointer keeps its conversation separate from every other tab.
        thread_id = gr.State()
        # gradio 6 uses the role/content message format natively (the old
        # tuples format was removed), so no `type=` kwarg is needed.
        chatbot = gr.Chatbot(height=420, label="Conversation")
        with gr.Row():
            box = gr.Textbox(
                placeholder="Ask about music or invoices…",
                scale=8,
                show_label=False,
                autofocus=True,
            )
            send = gr.Button("Send", variant="primary", scale=1)
        reset = gr.Button("Reset conversation", variant="secondary")

        # Fresh thread_id on every page load (this is what makes two tabs independent).
        demo.load(_new_session, outputs=thread_id)

        # Send on button click AND on Enter.
        send.click(respond, [box, chatbot, thread_id], [chatbot, box])
        box.submit(respond, [box, chatbot, thread_id], [chatbot, box])

        # Reset clears the window AND mints a NEW thread_id, so the checkpointer
        # starts a clean slot rather than reusing the old conversation.
        reset.click(lambda: ([], _new_session()), outputs=[chatbot, thread_id])

    return demo
