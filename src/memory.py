"""Milestone 5 -- preference memory (M-1, FR-5).

Remember a customer's stated music tastes ACROSS conversations, so the catalog
agent can personalize -- without ever letting memory become a hallucination
source (preferences only bias tone/suggestions; every fact still comes from a
tool call).

Two nodes bracket the agent:

    START -> load_memory -> assistant <-> tools
                                 | (final answer, no tool call)
                                 v
                           create_memory -> END

  - load_memory (START): read this customer's stored preferences and inject them
    as a SystemMessage so the agent knows their taste.
  - create_memory (after the response): extract any preference STATED this turn
    and MERGE it into the stored profile by set union.

Store vs checkpointer (the key distinction from Milestone 3's HITL graph):
  - a CHECKPOINTER (the MemorySaver used for interrupt/resume) saves ONE
    conversation's state, keyed by thread_id -- per-thread, so a resume finds the
    paused run.
  - a STORE (here, InMemoryStore) is CROSS-thread, keyed by customer_id -- it
    deliberately survives into a brand-new conversation. That is why memory uses a
    store, not a checkpointer.
  The store is injected into any node that declares a `store: BaseStore` parameter;
  it is attached to the graph at compile(store=...).

Three rules this file enforces (spec M-1):
  1. preference, not question  -- 'I love jazz' saves; 'do you have jazz?' does not.
  2. set-union merge           -- new prefs are added, old ones are NEVER removed.
  3. skip write on empty       -- an empty extraction leaves history untouched.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from src import tools
from src.agents import build_llm
from src.models import PreferenceExtraction
from src.prompts import MEMORY_EXTRACT_PROMPT, MUSIC_SYSTEM_PROMPT
from src.state import AgentState

NAMESPACE_PREFIX = "memory_profile"
PROFILE_KEY = "profile"


def profile_namespace(customer_id) -> tuple[str, str]:
    """Per-customer namespace: ("memory_profile", "<id>").

    Anonymous (unverified) sessions share a single "anonymous" bucket.
    """
    who = str(customer_id) if customer_id is not None else "anonymous"
    return (NAMESPACE_PREFIX, who)


def read_preferences(store: BaseStore, customer_id) -> list[str]:
    """Return the stored preference list for a customer ([] if none)."""
    item = store.get(profile_namespace(customer_id), PROFILE_KEY)
    if item is None:
        return []
    return list(item.value.get("music_preferences", []))


def apply_preferences(store: BaseStore, customer_id, new_prefs) -> list[str]:
    """Set-union merge `new_prefs` into the stored profile; return the result.

    This is the DETERMINISTIC core of create_memory, factored out so it can be
    unit-tested with no LLM. It enforces rules 2 and 3:
      - rule 3: an empty (or whitespace-only) extraction writes nothing -- history
        is never erased.
      - rule 2: otherwise the new tags are unioned with the existing ones (added,
        never removed), normalized to lowercase and de-duplicated.
    """
    cleaned = sorted({p.strip().lower() for p in new_prefs if p and p.strip()})
    existing = read_preferences(store, customer_id)
    if not cleaned:
        return existing  # rule 3: nothing stated -> never touch history
    merged = sorted(set(existing) | set(cleaned))  # rule 2: union, never remove
    store.put(profile_namespace(customer_id), PROFILE_KEY,
              {"music_preferences": merged})
    return merged


def load_memory(state: AgentState, *, store: BaseStore) -> dict:
    """START node: inject the customer's known preferences as a SystemMessage.

    No preferences -> inject nothing (return {}), so a first-time customer's graph
    is unchanged.
    """
    prefs = read_preferences(store, state.get("customer_id"))
    if not prefs:
        return {}
    note = (
        "Known preferences for this customer (personalize your tone and "
        "suggestions, but STILL ground every fact in a tool call): "
        + ", ".join(prefs)
    )
    return {"messages": [SystemMessage(content=note)]}


def build_create_memory_node(llm=None):
    """After-response node: extract this turn's stated preferences and merge them.

    The LLM call is the ONLY non-deterministic part; the write itself goes through
    apply_preferences (rules 2 and 3).
    """
    extractor = (llm or build_llm()).with_structured_output(PreferenceExtraction)

    def create_memory(state: AgentState, *, store: BaseStore) -> dict:
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if last_human is None:
            return {}
        result: PreferenceExtraction = extractor.invoke(
            [SystemMessage(content=MEMORY_EXTRACT_PROMPT), last_human]
        )
        apply_preferences(store, state.get("customer_id"), result.preferences)
        return {}

    return create_memory


def _route_after_assistant(state: AgentState) -> str:
    """tool call -> run tools; otherwise -> persist memory and finish."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "create_memory"


def build_memory_music_agent(store: BaseStore | None = None):
    """START -> load_memory -> assistant <-> tools -> create_memory -> END.

    Compiled WITH a store (created if not supplied). Read graph.store to inspect
    what was remembered. No thread_id/checkpointer needed: memory is cross-thread
    by design -- that is the point, it survives a brand-new conversation.
    """
    if tools.lc_music_tools is None:
        raise RuntimeError(
            "langchain-core is required to build the agent. "
            "Install deps: pip install langchain langchain-openai langgraph")

    store = store or InMemoryStore()
    music_tools = tools.lc_music_tools
    llm = build_llm()
    llm_with_tools = llm.bind_tools(music_tools)

    def assistant(state: AgentState) -> dict:
        messages = [SystemMessage(content=MUSIC_SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    builder = StateGraph(AgentState)
    builder.add_node("load_memory", load_memory)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(music_tools))
    builder.add_node("create_memory", build_create_memory_node(llm))

    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "assistant")
    builder.add_conditional_edges(
        "assistant", _route_after_assistant,
        {"tools": "tools", "create_memory": "create_memory"},
    )
    builder.add_edge("tools", "assistant")
    builder.add_edge("create_memory", END)

    # A store (not a checkpointer) carries cross-conversation memory.
    return builder.compile(store=store)


if __name__ == "__main__":
    from src import config

    graph = build_memory_music_agent()
    print("compiled nodes:", list(graph.get_graph().nodes.keys()))
    print("has store:", graph.store is not None)

    if config.OPENAI_API_KEY:
        cid = 1

        def say(text: str) -> None:
            graph.invoke(
                {"messages": [HumanMessage(content=text)], "customer_id": cid})

        print("\n-- 'I love jazz' (a preference -> SAVED) --")
        say("I love jazz, by the way.")
        print("stored:", read_preferences(graph.store, cid))

        print("\n-- 'do you have any rock albums?' (a question -> NOT saved) --")
        say("Do you have any rock albums?")
        print("stored:", read_preferences(graph.store, cid))

        print("\n-- 'I'm also really into metal' (new pref -> UNION, jazz survives) --")
        say("I'm also really into metal.")
        print("stored:", read_preferences(graph.store, cid))
    else:
        print("\n(Set OPENAI_API_KEY in .env to run the live memory demo.)")
