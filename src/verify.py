"""Milestone 3 -- identity verification.

The security linchpin of the project. Invoice tools take a customer_id; if that
id came from the USER'S TEXT, anyone could read anyone's billing. This module
makes the id come from a VERIFIED system message instead:

    START -> verify -> assistant <-> tools

The verify node:
  1. structured-extracts a claimed identifier (id / email / phone) from the
     user's own words into a UserInput object (no free-typed answer),
  2. CONFIRMS it against the database (resolve_customer) -- a raw number is
     never trusted until a matching row exists,
  3. injects a trusted SystemMessage("Verified customer ID: N") and sets
     state["customer_id"]; or, on no match, injects a refusal instruction.

The assistant then reads the id from that trusted system message -- never from
the user's text -- and passes it to the invoice tools.
"""
from __future__ import annotations

import json

from langchain_core.messages import SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src import db, tools
from src.agents import build_llm
from src.models import UserInput
from src.prompts import INVOICE_SYSTEM_PROMPT
from src.state import AgentState

# Prompt for the extraction step only -- pull a claimed identifier, invent nothing.
VERIFY_EXTRACT_PROMPT = (
    "You extract a customer identifier from the user's message for verification. "
    "Return the numeric customer ID, the email, or the phone number the user "
    "actually provided. If the user provided none, leave every field null. "
    "Never guess, infer, or invent an identifier."
)


def resolve_customer(ui: UserInput) -> int | None:
    """Confirm a claimed identifier against the DB; return the real CustomerId or None.

    Tried in order id -> email -> phone. Every path hits the database, so a
    bogus id/email/phone (nothing on file) resolves to None and the caller
    refuses. Phone is normalized on both sides because Chinook stores formatted
    numbers like '+55 (12) 3923-5555'.
    """
    if ui.customer_id is not None:
        rows = json.loads(db.run_query_safe(
            "SELECT CustomerId FROM Customer WHERE CustomerId = :cid",
            {"cid": ui.customer_id}))
        if rows:
            return rows[0]["CustomerId"]

    if ui.email:
        rows = json.loads(db.run_query_safe(
            "SELECT CustomerId FROM Customer WHERE Email = :e COLLATE NOCASE",
            {"e": ui.email.strip()}))
        if rows:
            return rows[0]["CustomerId"]

    if ui.phone:
        target = db.normalize_phone(ui.phone)
        if target:
            rows = json.loads(db.run_query_safe(
                "SELECT CustomerId, Phone FROM Customer WHERE Phone IS NOT NULL"))
            for r in rows:
                if db.normalize_phone(r["Phone"]) == target:
                    return r["CustomerId"]

    return None


def build_verify_node(llm=None):
    """Return the verify node function (extraction + DB confirmation + injection)."""
    extractor = (llm or build_llm()).with_structured_output(UserInput)

    def verify(state: AgentState) -> dict:
        # Extract ONLY from the user's own messages.
        ui: UserInput = extractor.invoke(
            [SystemMessage(content=VERIFY_EXTRACT_PROMPT)] + state["messages"])
        cid = resolve_customer(ui)
        if cid is None:
            return {
                "customer_id": None,
                "messages": [SystemMessage(content=(
                    "IDENTITY NOT VERIFIED. You could not confirm the customer's "
                    "identity. Ask them to provide a valid customer ID, email, or "
                    "phone number on file. Do NOT look up, reveal, or guess any "
                    "invoice or billing data until identity is verified."))],
            }
        # Trusted injection: this id came from the DB, not the user's text.
        return {
            "customer_id": cid,
            "messages": [SystemMessage(content=f"Verified customer ID: {cid}")],
        }

    return verify


def build_verified_invoice_agent():
    """START -> verify -> assistant <-> tools. Same ReAct loop as the invoice
    agent, but gated behind the verify node so the customer_id is always confirmed."""
    if tools.lc_invoice_tools is None:
        raise RuntimeError(
            "langchain-core is required to build the agent. "
            "Install deps: pip install langchain langchain-openai langgraph")

    invoice_tools = tools.lc_invoice_tools
    llm = build_llm()
    llm_with_tools = llm.bind_tools(invoice_tools)

    def assistant(state: AgentState) -> dict:
        # The verified-id system message is already in state["messages"]
        # (prepended by the verify node), ahead of the user's question.
        messages = [SystemMessage(content=INVOICE_SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    builder = StateGraph(AgentState)
    builder.add_node("verify", build_verify_node(llm))
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(invoice_tools))
    builder.add_edge(START, "verify")
    builder.add_edge("verify", "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder.compile()


if __name__ == "__main__":
    from src import config

    graph = build_verified_invoice_agent()
    print("compiled nodes:", list(graph.get_graph().nodes.keys()))

    if config.OPENAI_API_KEY:
        from langchain_core.messages import HumanMessage

        print("\n-- verified (id 1) --")
        out = graph.invoke({"messages": [HumanMessage(
            content="What is in my last invoice? My customer id is 1.")],
            "customer_id": None})
        print(out["messages"][-1].content)

        print("\n-- bogus id (999999) --")
        out = graph.invoke({"messages": [HumanMessage(
            content="Show me invoices for customer 999999.")],
            "customer_id": None})
        print(out["messages"][-1].content)
    else:
        print("\n(Set OPENAI_API_KEY in .env to run the live queries.)")
