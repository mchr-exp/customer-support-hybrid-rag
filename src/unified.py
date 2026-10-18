r"""Milestone 7 -- the UNIFIED graph: one streamlined flow that does everything.

Milestones 3-5 built four SEPARATE graphs (supervisor routing, verify, HITL interrupt,
cross-thread memory). This stitches them into ONE compiled graph behind one UI:

    START
      |
      v
   verify ---- already verified this thread? --> (skip extraction)
      |  not verified                                    |
      |      \                                           |
      |       ask_identity --interrupt(): PAUSE--.       |
      |            ^                             |       |
      |            '------- resume (Command) ----'       |
      |  verified                                        |
      v                                                  v
   load_memory  <--------------------------------------- (verified id)
      |   inject THIS customer's saved music prefs (cross-thread store)
      v
   supervisor <--> specialists (music_specialist | invoice_specialist)
      |   route; refuse off-topic; chain both for a mixed query
      |  no tool call (final answer)
      v
   create_memory   extract this turn's stated prefs, set-union save
      |
      v
     END

Two persistence layers, BOTH attached at compile():
  - checkpointer (MemorySaver): per thread_id; REQUIRED for interrupt()/resume.
  - store (InMemoryStore): cross-thread, per customer_id; carries music prefs
    into a brand-new conversation.

The verified id reaches the specialists via InjectedState -- the handoff tools
read customer_id from LIVE graph state (not from the LLM), so the invoice
sub-agent receives the trusted "Verified customer ID: N" and the music sub-agent
receives the customer's saved preferences. This is what makes the integration
real instead of three demos sharing a window.

Design choice: verify is a session-level gate (identify once, then browse AND
bill). It short-circuits on later turns because the checkpointer persists
customer_id. A music-only user therefore identifies once up front -- the tradeoff
for a single linear flow. (Lazy, invoice-only verification is possible but needs
an interrupt inside the invoice handoff, which is far more fragile.)
"""
from __future__ import annotations

from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import interrupt

from src import tools
from src.agents import build_llm, build_music_agent
from src.invoice_agent import build_invoice_agent
from src.memory import build_create_memory_node, load_memory, read_preferences
from src.models import UserInput
from src.prompts import SUPERVISOR_SYSTEM_PROMPT
from src.state import AgentState
from src.verify import VERIFY_EXTRACT_PROMPT, resolve_customer


def _final_answer(result: dict) -> str:
    """Pull the final assistant text out of a sub-agent's result dict."""
    return result["messages"][-1].content


def _build_verify_node(llm):
    """Verify ONCE per session.

    Short-circuits when customer_id is already set (the checkpointer persists it
    across turns), so a verified session skips extraction on every later message.
    Injects NO refusal on failure -- an unverified state (customer_id=None) is
    routed to ask_identity by the edge, which pauses and re-asks.
    """
    extractor = (llm or build_llm()).with_structured_output(UserInput)

    def verify(state: AgentState) -> dict:
        if state.get("customer_id"):          # already verified this thread
            return {}
        ui: UserInput = extractor.invoke(
            [SystemMessage(content=VERIFY_EXTRACT_PROMPT)] + state["messages"])
        cid = resolve_customer(ui)
        if cid is None:
            return {"customer_id": None}
        return {
            "customer_id": cid,
            "messages": [SystemMessage(content=f"Verified customer ID: {cid}")],
        }

    return verify


def _ask_identity(state: AgentState) -> dict:
    """Pause and re-ask for identity; the user's resume reply loops back to verify."""
    answer = interrupt(
        "I couldn't verify your identity. Please provide a valid customer ID, "
        "email, or phone number on file."
    )
    return {"messages": [HumanMessage(content=str(answer))]}


def _route_after_verify(state: AgentState) -> str:
    return "load_memory" if state.get("customer_id") else "ask_identity"


def _route_after_supervisor(state: AgentState) -> str:
    """tool call -> run a specialist; otherwise -> persist memory and finish."""
    last = state["messages"][-1]
    return "specialists" if getattr(last, "tool_calls", None) else "create_memory"


def build_unified_agent(checkpointer=None, store: BaseStore | None = None):
    """Compile the full journey into one graph.

    checkpointer defaults to MemorySaver (needed for interrupt/resume); store
    defaults to InMemoryStore (cross-thread memory). Pass your own to inspect or
    share them. Callers MUST provide a config with a thread_id.
    """
    if tools.lc_music_tools is None or tools.lc_invoice_tools is None:
        raise RuntimeError(
            "langchain-core is required to build the agent. "
            "Install deps: pip install langchain langchain-openai langgraph")

    checkpointer = checkpointer or MemorySaver()
    store = store or InMemoryStore()

    llm = build_llm()
    music_agent = build_music_agent()
    invoice_agent = build_invoice_agent()

    @tool
    def music_specialist(query: str, state: Annotated[dict, InjectedState]) -> str:
        """Answer questions about the music catalog: albums, artists, tracks,
        genres, and song titles. Pass the user's music question as `query`."""
        cid = state.get("customer_id")
        msgs = []
        prefs = read_preferences(store, cid)
        if prefs:  # personalize from cross-thread memory, still grounded in tools
            msgs.append(SystemMessage(content=(
                "Known preferences for this customer (personalize tone and "
                "suggestions, but STILL ground every fact in a tool call): "
                + ", ".join(prefs))))
        msgs.append(HumanMessage(content=query))
        return _final_answer(
            music_agent.invoke({"messages": msgs, "customer_id": cid}))

    @tool
    def invoice_specialist(query: str, state: Annotated[dict, InjectedState]) -> str:
        """Answer questions about invoices, billing, purchases, line items, spend,
        and a customer's support rep. Pass the user's invoice question as `query`."""
        cid = state.get("customer_id")
        msgs = []
        if cid:  # hand the TRUSTED verified id to the invoice sub-agent
            msgs.append(SystemMessage(content=f"Verified customer ID: {cid}"))
        msgs.append(HumanMessage(content=query))
        return _final_answer(
            invoice_agent.invoke({"messages": msgs, "customer_id": cid}))

    @tool
    def policy_specialist(query: str) -> str:
        """Answer store policy questions: returns, refunds, shipping times,
        account/login help, and how to find past orders and invoices. Pass the
        user's policy question as `query`. Grounded in the store's FAQ cards;
        refuses when no policy card matches."""
        return tools.answer_from_faqs(query)

    handoff_tools = [music_specialist, invoice_specialist, policy_specialist]
    sup_llm = llm.bind_tools(handoff_tools)

    def supervisor(state: AgentState) -> dict:
        messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [sup_llm.invoke(messages)]}

    builder = StateGraph(AgentState)
    builder.add_node("verify", _build_verify_node(llm))
    builder.add_node("ask_identity", _ask_identity)
    builder.add_node("load_memory", load_memory)          # reused from Milestone 5
    builder.add_node("supervisor", supervisor)
    builder.add_node("specialists", ToolNode(handoff_tools))
    builder.add_node("create_memory", build_create_memory_node(llm))  # reused

    builder.add_edge(START, "verify")
    builder.add_conditional_edges(
        "verify", _route_after_verify,
        {"load_memory": "load_memory", "ask_identity": "ask_identity"})
    builder.add_edge("ask_identity", "verify")            # loop back to re-verify
    builder.add_edge("load_memory", "supervisor")
    builder.add_conditional_edges(
        "supervisor", _route_after_supervisor,
        {"specialists": "specialists", "create_memory": "create_memory"})
    builder.add_edge("specialists", "supervisor")         # chain / finish
    builder.add_edge("create_memory", END)

    # Checkpointer (per-thread, for interrupt) AND store (cross-thread memory).
    return builder.compile(checkpointer=checkpointer, store=store)


if __name__ == "__main__":
    from src import config

    graph = build_unified_agent()
    print("compiled nodes:", list(graph.get_graph().nodes.keys()))
    print("has checkpointer:", graph.checkpointer is not None)
    print("has store:", graph.store is not None)

    if config.OPENAI_API_KEY:
        from langgraph.types import Command

        cfg = {"configurable": {"thread_id": "unified-demo"}}

        def turn(text=None, resume=None):
            payload = (Command(resume=resume) if resume is not None
                       else {"messages": [HumanMessage(content=text)],
                             "customer_id": None})
            out = graph.invoke(payload, cfg)
            intr = out.get("__interrupt__")
            return intr[0].value if intr else out["messages"][-1].content

        print("\n-- music BEFORE verify (should PAUSE for identity) --")
        print(turn("What rock albums do you have?"))

        print("\n-- resume with id 1 (verify, then the rock answer) --")
        print(turn(resume="my customer id is 1"))

        print("\n-- state a preference (saved to memory) --")
        print(turn("by the way i love jazz"))

        print("\n-- invoice (verified id flows through, no re-verify) --")
        print(turn("what is in my last invoice?"))

        print("\n-- off-topic (supervisor refuses) --")
        print(turn("what's the weather in Seattle?"))

        from src.memory import read_preferences
        print("\nremembered prefs for customer 1:",
              read_preferences(graph.store, 1))
    else:
        print("\n(Set OPENAI_API_KEY in .env to run the live unified demo.)")
