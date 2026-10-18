"""Retrieval metrics, hand-implemented for learning transparency.

All take:
  retrieved : list of doc ids, best-first
  gold      : set/list of relevant doc ids
"""
from __future__ import annotations
import math


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """Fraction of gold docs found in the top-k."""
    if not gold:
        return 0.0
    topk = set(retrieved[:k])
    hit = sum(1 for g in gold if g in topk)
    return hit / len(gold)


def reciprocal_rank(retrieved: list[str], gold: list[str]) -> float:
    """1/rank of the FIRST relevant doc (0 if none). Averaged -> MRR."""
    gold_set = set(gold)
    for rank, d in enumerate(retrieved, start=1):
        if d in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """Binary-gain nDCG@k: rewards relevant docs ranked higher."""
    gold_set = set(gold)
    dcg = 0.0
    for i, d in enumerate(retrieved[:k]):
        if d in gold_set:
            dcg += 1.0 / math.log2(i + 2)      # gain 1, discount by position
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def aggregate(rows: list[dict]) -> dict:
    """Mean each metric across queries."""
    if not rows:
        return {}
    keys = [k for k in rows[0] if k != "query"]
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}
