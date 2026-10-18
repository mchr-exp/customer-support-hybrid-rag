"""Milestone 4 (sub-agents) -- a hand-built ReAct music agent (LangGraph StateGraph).

ReAct = Reason -> Act -> Observe, run in a loop:

        START
          |
          v
    +-----------+      has tool_call?        +-----------+
    | assistant | --------- yes ------------>|   tools   |
    |  (LLM +   |<------------------------- |  (runs the |
    |  tools)   |     (loop: LLM now sees    |   tool,    |
    +-----------+      the tool result)      | appends it)|
          |                                  +-----------+
          |  no tool_call (final prose)
          v
         END

Two prebuilt primitives keep the wiring small:
  - ToolNode(tools):     a node that reads the last message's tool_call, runs
                         the matching Python tool, and appends its JSON output.
  - tools_condition:     the conditional-edge router -- returns "tools" if the
                         LLM asked for a tool, else END.

The anti-hallucination contract lives in two places: temperature=0 (no creative
drift) and the system prompt (answer ONLY from tool output). The shared grounding
prompt carries the full 6 grounding rules.
"""
from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src import config, tools
from src.prompts import MUSIC_SYSTEM_PROMPT   # role + the 6 shared grounding rules
from src.state import AgentState


def build_llm() -> ChatOpenAI:
    """Create the chat model from env-driven config (temperature=0 for grounding).

    Public so the invoice agent can reuse the exact same env-driven LLM.
    """
    kwargs = {"model": config.MODEL_NAME, "temperature": config.TEMPERATURE}
    # Support an OpenAI-compatible endpoint if one is configured.
    if config.OPENAI_API_BASE:
        kwargs["base_url"] = config.OPENAI_API_BASE
    return ChatOpenAI(**kwargs)


def build_music_agent():
    """Compile and return the hand-built ReAct graph for the music catalog."""
    if tools.lc_music_tools is None:  # langchain not importable
        raise RuntimeError(
            "langchain-core is required to build the agent. "
            "Install deps: pip install langchain langchain-openai langgraph"
        )

    music_tools = tools.lc_music_tools          # the @tool-wrapped music tools
    llm_with_tools = build_llm().bind_tools(music_tools)   # attach tool schemas

    # --- the assistant node: LLM reasons, maybe emits a tool_call ---
    def assistant(state: AgentState) -> dict:
        # Prepend the system prompt, then the running conversation.
        messages = [SystemMessage(content=MUSIC_SYSTEM_PROMPT)] + state["messages"]
        reply = llm_with_tools.invoke(messages)
        # add_messages reducer appends this to state["messages"].
        return {"messages": [reply]}

    # --- wire the graph ---
    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(music_tools))   # same list bound to the LLM

    builder.add_edge(START, "assistant")               # start -> think
    builder.add_conditional_edges("assistant", tools_condition)  # think -> tools or END
    builder.add_edge("tools", "assistant")             # observe -> think again (loop)

    return builder.compile()


if __name__ == "__main__":
    # Stage A (no key): prove the graph compiles and is wired correctly.
    graph = build_music_agent()
    print("compiled nodes:", list(graph.get_graph().nodes.keys()))

    # Stage B (needs OPENAI_API_KEY in .env): run a real grounded query.
    if config.OPENAI_API_KEY:
        from langchain_core.messages import HumanMessage

        out = graph.invoke(
            {"messages": [HumanMessage(content="What rock albums do you have?")],
             "customer_id": None}
        )
        print("\nAnswer:\n", out["messages"][-1].content)
    else:
        print("\n(Set OPENAI_API_KEY in .env to run Stage B -- the live query.)")
