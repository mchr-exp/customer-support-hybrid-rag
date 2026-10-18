"""Milestone 5 -- preference memory tests.

The offline tests pin the DETERMINISTIC core with a real InMemoryStore and NO LLM:
namespace scoping, set-union merge, skip-on-empty, per-customer isolation, and the
load_memory injection. One gated live test proves the preference-vs-question call
end to end.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.store.memory import InMemoryStore

from src import config
from src.memory import (
    apply_preferences,
    build_memory_music_agent,
    load_memory,
    profile_namespace,
    read_preferences,
)


def test_namespace_scopes_by_customer():
    assert profile_namespace(1) == ("memory_profile", "1")
    assert profile_namespace(None) == ("memory_profile", "anonymous")
    assert profile_namespace(1) != profile_namespace(2)


def test_merge_is_set_union_and_never_removes():
    store = InMemoryStore()
    apply_preferences(store, 1, ["jazz"])
    apply_preferences(store, 1, ["ac/dc"])  # add a second, keep the first
    assert read_preferences(store, 1) == ["ac/dc", "jazz"]  # sorted union


def test_merge_dedups_and_normalizes_case():
    store = InMemoryStore()
    apply_preferences(store, 1, ["Jazz", "JAZZ", " jazz "])
    assert read_preferences(store, 1) == ["jazz"]


def test_empty_extraction_never_erases_history():
    store = InMemoryStore()
    apply_preferences(store, 1, ["jazz"])
    apply_preferences(store, 1, [])         # rule 3: empty -> no write
    apply_preferences(store, 1, ["   "])    # whitespace-only -> still empty
    assert read_preferences(store, 1) == ["jazz"]


def test_preferences_are_isolated_per_customer():
    store = InMemoryStore()
    apply_preferences(store, 1, ["jazz"])
    apply_preferences(store, 2, ["metal"])
    assert read_preferences(store, 1) == ["jazz"]
    assert read_preferences(store, 2) == ["metal"]


def test_load_memory_injects_known_prefs():
    store = InMemoryStore()
    apply_preferences(store, 1, ["jazz"])
    out = load_memory({"messages": [], "customer_id": 1}, store=store)
    msgs = out.get("messages", [])
    assert len(msgs) == 1 and isinstance(msgs[0], SystemMessage)
    assert "jazz" in msgs[0].content.lower()


def test_load_memory_noop_when_no_prefs():
    store = InMemoryStore()
    out = load_memory({"messages": [], "customer_id": 1}, store=store)
    assert out == {}


def test_graph_compiles_with_memory_nodes():
    graph = build_memory_music_agent()
    nodes = set(graph.get_graph().nodes.keys())
    assert {"load_memory", "assistant", "tools", "create_memory"} <= nodes
    assert graph.store is not None


@pytest.mark.live
@pytest.mark.skipif(not config.OPENAI_API_KEY, reason="no OPENAI_API_KEY in .env")
def test_live_preference_saved_question_not_saved_old_survives():
    graph = build_memory_music_agent()
    cid = 1

    def say(text: str) -> None:
        graph.invoke(
            {"messages": [HumanMessage(content=text)], "customer_id": cid})

    say("I love jazz.")
    assert "jazz" in read_preferences(graph.store, cid)  # preference -> saved

    say("Do you have any rock albums?")                  # a question -> no new pref
    after_question = read_preferences(graph.store, cid)
    assert "jazz" in after_question                      # old survives

    say("I'm also really into metal.")                   # new pref -> union
    final = read_preferences(graph.store, cid)
    assert "metal" in final and "jazz" in final          # both present
