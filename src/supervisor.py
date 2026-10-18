"""Milestone 4 — hand-built supervisor (router) over the specialist agents.

Key idea: the supervisor is itself a ReAct agent — the SAME StateGraph loop you
built in this milestone — but its "tools" are the specialist agents instead of
SQL functions. It decides which specialist handles a query, chains both for a
mixed query, and refuses an off-topic question WITHOUT invoking any specialist
(the refusal happens at the router, so no sub-agent work is spent).

No `langgraph-supervisor` dependency: that package requires the langchain-core
1.x line, which conflicts with this project's pinned langchain-openai 0.3.x. The
hand-built graph is exactly what that library does internally (handoff tools),
and keeps the dependency set stable.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from src.state import AgentState
from src import tools
from src.agents import build_llm, build_music_agent
from src.invoice_agent import build_invoice_agent
from src.prompts import SUPERVISOR_SYSTEM_PROMPT


def _final_answer(result: dict) -> str:
    """Pull the final assistant text out of a sub-agent's result dict."""
    return result["messages"][-1].content


def build_supervisor(checkpointer=None):
    """Compile and return the hand-built supervisor graph.

    Pass a `checkpointer` (e.g. MemorySaver) to make the graph stateful per
    `thread_id` — the UI (Milestone 6) uses this so each browser session gets its own
    isolated conversation. Default `None` = stateless (unchanged for tests)."""
    # Build each specialist ONCE and close over it (don't rebuild per handoff).
    music_agent = build_music_agent()
    invoice_agent = build_invoice_agent()

    @tool
    def music_specialist(query: str) -> str:
        """Answer questions about the music catalog: albums, artists, tracks,
        genres, and song titles. Pass the user's music question as `query`."""
        out = music_agent.invoke(
            {"messages": [HumanMessage(content=query)], "customer_id": None}
        )
        return _final_answer(out)

    @tool
    def invoice_specialist(query: str) -> str:
        """Answer questions about invoices, billing, purchases, line items, spend,
        and a customer's support rep. Pass the user's invoice question as `query`."""
        out = invoice_agent.invoke(
            {"messages": [HumanMessage(content=query)], "customer_id": None}
        )
        return _final_answer(out)

    @tool
    def policy_specialist(query: str) -> str:
        """Answer store policy questions: returns, refunds, shipping times,
        account/login help, and how to find past orders and invoices. Pass the
        user's policy question as `query`. Grounded in the store's FAQ cards;
        refuses when no policy card matches."""
        return tools.answer_from_faqs(query)

    handoff_tools = [music_specialist, invoice_specialist, policy_specialist]
    llm = build_llm().bind_tools(handoff_tools)

    def supervisor(state: AgentState) -> dict:
        # Prepend the routing instruction; the LLM either calls a specialist
        # (handoff) or replies with no tool call (direct refusal / final relay).
        messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm.invoke(messages)]}

    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("tools", ToolNode(handoff_tools))
    builder.add_edge(START, "supervisor")
    # Conditional edge: tool_call -> "tools" (handoff); no tool_call -> END.
    builder.add_conditional_edges("supervisor", tools_condition)
    builder.add_edge("tools", "supervisor")  # loop back so the supervisor can
    #                                           chain a second specialist or finish
    return builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    g = build_supervisor()
    print("compiled supervisor nodes:", list(g.get_graph().nodes.keys()))
    for q in [
        "What rock albums do you have?",
        "What invoices does customer 1 have?",
        "What is the weather in Seattle?",
    ]:
        out = g.invoke({"messages": [HumanMessage(content=q)], "customer_id": None})
        print(f"\nQ: {q}\nA: {out['messages'][-1].content}")
