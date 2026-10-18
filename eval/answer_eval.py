"""Answer-side eval: routing accuracy + no-hallucination (grounded / refusal),
adapted to THIS project's unified LangGraph (src/unified.py).

The reference eval drove a monolithic `SupportAgent` with a structured
`Answer(route, refused, grounded_on)` return. This build has no such class --
routing lives in the LLM supervisor and answers come back as graph message
state. So we read the signals the graph actually produces:

  route    : which specialist TOOL the supervisor called, scanned from the
             message trace -- `music_specialist` / `invoice_specialist`, or
             NONE (the supervisor replied with no tool call => it refused).
  refused  : no specialist tool ran  ==  supervisor refused off-topic.
  grounded : a non-refused answer went through a specialist, which is itself
             bound by the 6 grounding rules + temperature=0 (prompts.py).

Isolation choices (documented, not hidden):
  * customer_id is pre-seeded to 1 so the verify node short-circuits and we
    skip the identity interrupt -- verify/resume is covered by test_unified.
    This eval isolates the SUPERVISOR's routing decision.
  * each case runs on a fresh thread_id so checkpointer state never bleeds.

The policy/FAQ lane: the supervisor now routes returns/refunds/shipping/account
questions to `policy_specialist`, which answers from the FAQ cards (grounded, and
refuses when no FAQ card matches). Those cases carry expected route 'policy'.

  python -m eval.answer_eval        # from project root (needs OPENAI_API_KEY)
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from langchain_core.messages import HumanMessage
from src import config


_ROUTE = {"music_specialist": "music",
          "invoice_specialist": "invoice",
          "policy_specialist": "policy"}
SPECIALIST_TOOLS = set(_ROUTE)


@dataclass
class Case:
    q: str
    route: str          # expected: 'music' | 'invoice' | 'refuse'
    note: str = ""


CASES = [
    Case("what rock albums do you have in store", "music"),
    Case("show me some jazz albums", "music"),
    Case("something upbeat like AC/DC", "music", "exercises semantic_catalog_search"),
    Case("what is in my last invoice", "invoice"),
    Case("how much have I spent in total", "invoice"),
    Case("what is the weather in Seattle today", "refuse", "off-topic"),
    Case("write me a poem about dragons", "refuse", "off-topic"),
    Case("how do I return a CD I bought", "policy"),
    Case("when will I get my refund", "policy"),
]


def _tool_names(messages) -> list[str]:
    """Every tool name requested across the message trace, in order."""
    names = []
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name:
                names.append(name)
    return names


def classify(messages) -> tuple[str, bool]:
    """Map a finished graph message trace to (route, refused).

    route in {'music','invoice','policy','mixed','refuse'}; refused == no
    specialist ran.
    """
    ran = [n for n in _tool_names(messages) if n in SPECIALIST_TOOLS]
    seen = []
    for n in ran:
        if n not in seen:
            seen.append(n)
    if not seen:
        return "refuse", True
    if len(seen) > 1:
        return "mixed", False
    return _ROUTE[seen[0]], False


def main():
    if not config.OPENAI_API_KEY:
        print("answer_eval needs OPENAI_API_KEY (the supervisor is an LLM router).")
        print("Set it in .env and re-run:  python -m eval.answer_eval")
        return

    from src.unified import build_unified_agent
    agent = build_unified_agent()
    print(f"driving: src.unified graph | cases: {len(CASES)}\n")

    rows = []
    for i, c in enumerate(CASES):
        cfg = {"configurable": {"thread_id": f"answer-eval-{i}"}}
        # pre-seed customer_id=1 -> verify short-circuits, no identity interrupt
        out = agent.invoke(
            {"messages": [HumanMessage(content=c.q)], "customer_id": 1}, cfg)
        paused = bool(out.get("__interrupt__"))
        route, refused = classify(out.get("messages", []))
        should_refuse = c.route == "refuse"
        route_ok = (not paused) and (route == c.route)
        refusal_ok = (not paused) and (refused == should_refuse)
        grounded_ok = (not paused) and (should_refuse or route in ("music", "invoice", "policy", "mixed"))
        rows.append((c, route, route_ok, refusal_ok, grounded_ok, paused))

    hdr = f"{'query':<40}{'exp':>8}{'got':>9}{'route':>7}{'refuse':>8}{'ground':>8}"
    print(hdr)
    print("-" * len(hdr))
    def m(b): return "  ok" if b else " XXX"
    agg = [0, 0, 0]
    for c, route, a, b, d, paused in rows:
        got = "PAUSED" if paused else route
        print(f"{c.q[:39]:<40}{c.route:>8}{got:>9}{m(a):>7}{m(b):>8}{m(d):>8}")
        agg[0] += int(a); agg[1] += int(b); agg[2] += int(d)
    n = len(rows)
    print("-" * len(hdr))
    print(f"{'ACCURACY':<40}{'':>8}{'':>9}{agg[0]/n:>7.0%}{agg[1]/n:>8.0%}{agg[2]/n:>8.0%}")


if __name__ == "__main__":
    main()
