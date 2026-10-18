"""Milestone 8 (hybrid RAG) -- hybrid retriever: dense + sparse(BM25), fused with RRF, optional MMR.

Concepts reinforced here (all hand-implemented, nothing hidden):
  - Dense retrieval  : semantic similarity via embeddings (embeddings.DenseIndex)
  - Sparse retrieval : lexical BM25 over tokenized text (rank_bm25)
  - RRF fusion       : combine two ranked lists by 1/(k+rank), rank-based so the
                       two scorers' incomparable score scales don't matter
  - MMR re-rank      : trade relevance vs diversity to reduce near-duplicate hits
"""
from __future__ import annotations
import re
import numpy as np
from rank_bm25 import BM25Okapi
from .corpus import Doc
from .embeddings import DenseIndex
from . import config

_token = re.compile(r"[a-z0-9]+")


def tok(s: str) -> list[str]:
    return _token.findall(s.lower())


class HybridRetriever:
    def __init__(self, docs: list[Doc]):
        self.docs = docs
        self.texts = [d.text for d in docs]
        self.dense = DenseIndex(self.texts)
        self.bm25 = BM25Okapi([tok(t) for t in self.texts])

    # ---- individual retrievers (return [(idx, score)] sorted desc) ----
    def dense_search(self, q: str, k: int) -> list[tuple[int, float]]:
        return self.dense.search(q, k)

    def sparse_search(self, q: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tok(q))
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]

    # ---- Reciprocal Rank Fusion ----
    @staticmethod
    def rrf(ranked_lists: list[list[int]], k: int = config.RRF_K) -> dict[int, float]:
        """ranked_lists: each is a list of doc indices best-first.
        RRF score(d) = sum over lists of 1/(k + rank_in_that_list)."""
        fused: dict[int, float] = {}
        for lst in ranked_lists:
            for rank, idx in enumerate(lst):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return fused

    def hybrid_search(self, q: str, k: int, pool: int = 20,
                      use_mmr: bool = False, mmr_lambda: float = 0.7):
        """Fuse dense + sparse with RRF, then optionally MMR-diversify."""
        dense_idx = [i for i, _ in self.dense_search(q, pool)]
        sparse_idx = [i for i, _ in self.sparse_search(q, pool)]
        fused = self.rrf([dense_idx, sparse_idx])
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        cand = [i for i, _ in ranked][: max(k, pool)]
        if use_mmr:
            cand = self._mmr(q, cand, k, mmr_lambda)
        else:
            cand = cand[:k]
        return [(i, fused[i]) for i in cand]

    # ---- Maximal Marginal Relevance re-rank ----
    def _mmr(self, q: str, cand_idx: list[int], k: int, lam: float) -> list[int]:
        """Greedy MMR: pick the doc maximizing
        lam*rel(q,d) - (1-lam)*max_sim(d, already_selected).
        Uses dense vectors for both relevance and redundancy."""
        if not cand_idx:
            return []
        M = self.dense.matrix
        qv = self.dense._embed_query(q).ravel()
        rel = {i: float(M[i] @ qv) for i in cand_idx}
        selected: list[int] = []
        pool = list(cand_idx)
        while pool and len(selected) < k:
            best, best_score = None, -1e9
            for i in pool:
                redun = max((float(M[i] @ M[j]) for j in selected), default=0.0)
                score = lam * rel[i] - (1 - lam) * redun
                if score > best_score:
                    best, best_score = i, score
            selected.append(best)
            pool.remove(best)
        return selected

    def doc(self, idx: int) -> Doc:
        return self.docs[idx]
