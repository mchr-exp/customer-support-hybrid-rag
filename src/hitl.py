"""Milestone 3 (human-in-the-loop) -- interrupt/resume for failed verification.

The base verify step REFUSED outright when identity could not be confirmed
(inject a refusal, stop). This turns that dead end into a conversation: the graph
PAUSES, asks the customer for a valid identifier, and RESUMES with their next
message -- looping back to re-verify until they confirm.

    START -> verify -> [verified?]
                         yes -> assistant <-> tools -> END
                         no  -> ask_identity --interrupt(): PAUSE--.
                                     ^                             |
                                     '---- resume (Command) -------'
                                     (user's answer -> HumanMessage -> back to verify)

Two new LangGraph pieces:
  - interrupt(value): pauses the graph at this node and hands `value` back to the
    caller of .invoke(). The run stops mid-graph until it is resumed.
  - a checkpointer (MemorySaver): persists graph state per thread_id so the paused
    run can be restored. interrupt() REQUIRES one -- without a checkpointer there is
    nowhere to save the paused state. The thread_id lives in the invoke config:
        {"configurable": {"thread_id": "some-id"}}

Resume by calling the graph again with Command(resume=<user's answer>) and the SAME
thread_id; interrupt() then RETURNS that answer and the node continues.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

from src import tools
from src.agents import build_llm
from src.models import UserInput
from src.prompts import INVOICE_SYSTEM_PROMPT
from src.state import AgentState
from src.verify import VERIFY_EXTRACT_PROMPT, resolve_customer


def build_hitl_verify_node(llm=None):
    """Verify node for the HITL graph: structured-extract an identifier, confirm it
    against the DB, set customer_id.

    Difference from the base verify node: it injects NO refusal on failure. An
    unverified state (customer_id=None) is handled by the routing edge, which sends
    it to ask_identity instead. On success it still injects the trusted
    'Verified customer ID: N' system message (the same security property).
    """
    extractor = (llm or build_llm()).with_structured_output(UserInput)

    def verify(state: AgentState) -> dict:
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


def route_after_verify(state: AgentState) -> str:
    """Conditional edge after verify: verified -> assistant; not verified -> ask_identity."""
    return "assistant" if state.get("customer_id") else "ask_identity"


def ask_identity(state: AgentState) -> dict:
    """Pause the graph and re-ask for identity.

    interrupt() suspends execution HERE and surfaces its argument to the caller.
    The run does not continue until the caller resumes with Command(resume=<answer>);
    interrupt() then RETURNS that answer, which we append as a HumanMessage so the
    verify node (looped back to next) re-extracts from the customer's new reply.
    """
    answer = interrupt(
        "I couldn't verify your identity. Please provide a valid customer ID, "
        "email, or phone number on file."
    )
    return {"messages": [HumanMessage(content=str(answer))]}


def build_hitl_invoice_agent():
    """START -> verify -> {assistant <-> tools | ask_identity -> verify}.

    Compiled WITH a checkpointer so interrupt()/resume works. Callers MUST pass a
    config carrying a thread_id, e.g. {"configurable": {"thread_id": "abc"}}.
    """
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
    builder.add_node("verify", build_hitl_verify_node(llm))
    builder.add_node("ask_identity", ask_identity)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(invoice_tools))

    builder.add_edge(START, "verify")
    builder.add_conditional_edges(
        "verify", route_after_verify,
        {"assistant": "assistant", "ask_identity": "ask_identity"},
    )
    builder.add_edge("ask_identity", "verify")      # loop back to re-verify the retry
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    # A checkpointer is MANDATORY for interrupt()/resume.
    return builder.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    from src import config

    graph = build_hitl_invoice_agent()
    print("compiled nodes:", list(graph.get_graph().nodes.keys()))
    print("has checkpointer:", graph.checkpointer is not None)

    if config.OPENAI_API_KEY:
        from langgraph.types import Command

        cfg = {"configurable": {"thread_id": "day9-demo"}}

        print("\n-- unverified id 99999 (should PAUSE) --")
        out = graph.invoke(
            {"messages": [HumanMessage(content="Show me invoices for customer 99999.")],
             "customer_id": None},
            cfg,
        )
        intr = out.get("__interrupt__")
        print("interrupted:", bool(intr))
        if intr:
            print("re-ask prompt:", intr[0].value)

        print("\n-- RESUME with a valid id (1) --")
        out = graph.invoke(Command(resume="Actually my customer id is 1."), cfg)
        print("customer_id:", out.get("customer_id"))
        print(out["messages"][-1].content)
    else:
        print("\n(Set OPENAI_API_KEY in .env to run the live pause/resume demo.)")
