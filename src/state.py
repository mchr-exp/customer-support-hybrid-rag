"""Shared agent state -- the 'blackboard' every graph node reads and writes.

LangGraph passes ONE state object between nodes. Each node returns a *partial*
update (e.g. {"messages": [reply]}) and LangGraph merges it in. How it merges
depends on the field's reducer:

  - a plain field (customer_id) is OVERWRITTEN by any update.
  - a field Annotated with a reducer (messages) is MERGED by that reducer.

`add_messages` is the reducer that APPENDS new messages instead of replacing the
list -- that is what lets the conversation accumulate across the ReAct loop
(user question -> LLM tool call -> tool result -> LLM answer) instead of
resetting on every hop.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State carried through the music agent graph."""

    # The running conversation. `add_messages` => appended, not overwritten.
    messages: Annotated[list, add_messages]

    # The verified customer (set in the Milestone 3 verify node). None until then.
    customer_id: int | None
