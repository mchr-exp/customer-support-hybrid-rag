"""Milestone 4 (sub-agents) -- the invoice sub-agent, hand-built as a StateGraph.

Same ReAct shape as the music agent (src/agents.py): an assistant node (LLM +
bound invoice tools) <-> a tool node, joined by a conditional edge that loops
while the LLM keeps asking for tools and ends when it writes a final answer.

We build it by hand (rather than the prebuilt create_react_agent, which is
deprecated in LangGraph v1.0) so the wiring is explicit and the two agents share
one pattern. The invoice prompt also carries the shared 6 grounding rules and
the Day-8 rule: read the verified customer ID from a system message, never from
the user's text.

Note: this is intentionally the same wiring as build_music_agent -- in
production you'd factor a shared `_build_react_graph(tools, prompt)` helper; here
the parallel is kept visible on purpose so the pattern is obvious.
"""
from __future__ import annotations

from langchain_core.messages import SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src import tools
from src.agents import build_llm            # reuse the same env-driven LLM builder
from src.prompts import INVOICE_SYSTEM_PROMPT
from src.state import AgentState


def build_invoice_agent():
    """Compile and return a hand-built ReAct graph over the 4 invoice tools."""
    if tools.lc_invoice_tools is None:       # langchain not importable
        raise RuntimeError(
            "langchain-core is required to build the agent. "
            "Install deps: pip install langchain langchain-openai langgraph"
        )

    invoice_tools = tools.lc_invoice_tools
    llm_with_tools = build_llm().bind_tools(invoice_tools)

    def assistant(state: AgentState) -> dict:
        # Prepend the invoice system prompt; state already carries the verified-id
        # system message (Milestone 3 verified id) + the user message.
        messages = [SystemMessage(content=INVOICE_SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(invoice_tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder.compile()


if __name__ == "__main__":
    from src import config

    graph = build_invoice_agent()
    print("compiled nodes:", list(graph.get_graph().nodes.keys()))

    if config.OPENAI_API_KEY:
        from langchain_core.messages import HumanMessage

        # Milestone 3 injects the verified id automatically; here we seed it by hand.
        out = graph.invoke({"messages": [
            SystemMessage(content="Verified customer ID: 1"),
            HumanMessage(content="What is in my last invoice?"),
        ], "customer_id": 1})
        print("\nAnswer:\n", out["messages"][-1].content)
    else:
        print("\n(Set OPENAI_API_KEY in .env to run the live query.)")
