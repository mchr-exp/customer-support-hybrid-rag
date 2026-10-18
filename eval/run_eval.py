"""Compare DENSE vs SPARSE(BM25) vs HYBRID(RRF) vs HYBRID+MMR retrieval.

Runs fully offline (local embeddings or TF-IDF+SVD fallback). Prints a
per-strategy metric table so the payoff of hybrid retrieval is measurable.

  python -m eval.run_eval          # from project root
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.corpus import build_corpus
from src.hybrid_retriever import HybridRetriever
from eval.dataset import build_eval_set
from eval import metrics as M

K = 5          # cutoff for recall@k / ndcg@k
POOL = 20      # candidate pool each retriever pulls before fusion


def _ids(retr, r):
    return [retr.doc(i).id for i, _ in r]


def run_strategy(retr, name, queries):
    rows = []
    for q in queries:
        if name == "dense":
            r = retr.dense_search(q.query, POOL)
        elif name == "sparse":
            r = retr.sparse_search(q.query, POOL)
        elif name == "hybrid":
            r = retr.hybrid_search(q.query, K, pool=POOL, use_mmr=False)
        elif name == "hybrid+mmr":
            r = retr.hybrid_search(q.query, K, pool=POOL, use_mmr=True)
        else:
            raise ValueError(name)
        ids = _ids(retr, r)
        rows.append({
            "query": q.query,
            "recall@k": M.recall_at_k(ids, q.gold, K),
            "mrr": M.reciprocal_rank(ids, q.gold),
            "ndcg@k": M.ndcg_at_k(ids, q.gold, K),
        })
    return rows


def main():
    print("Building corpus + indexes...")
    docs = build_corpus()
    retr = HybridRetriever(docs)
    print(f"  corpus: {len(docs)} docs | dense backend: {retr.dense.backend}")
    queries = build_eval_set(docs)
    print(f"  eval queries: {len(queries)} | K={K}\n")

    strategies = ["dense", "sparse", "hybrid", "hybrid+mmr"]
    agg = {}
    per_strategy_rows = {}
    for s in strategies:
        rows = run_strategy(retr, s, queries)
        per_strategy_rows[s] = rows
        agg[s] = M.aggregate(rows)

    # summary table
    print("=" * 62)
    print(f"{'strategy':<14}{'recall@'+str(K):>12}{'MRR':>12}{'nDCG@'+str(K):>12}")
    print("-" * 62)
    for s in strategies:
        a = agg[s]
        print(f"{s:<14}{a['recall@k']:>12.3f}{a['mrr']:>12.3f}{a['ndcg@k']:>12.3f}")
    print("=" * 62)

    # per-query MRR, to see WHERE hybrid rescues dense/sparse misses
    print("\nPer-query MRR (0.000 = first relevant doc not in top pool):")
    print(f"{'query':<44}{'dense':>8}{'sparse':>8}{'hybrid':>8}")
    print("-" * 68)
    for i, q in enumerate(queries):
        d = per_strategy_rows['dense'][i]['mrr']
        sp = per_strategy_rows['sparse'][i]['mrr']
        h = per_strategy_rows['hybrid'][i]['mrr']
        flag = "  <-- hybrid rescues" if h > max(d, sp) + 1e-9 else ""
        print(f"{q.query[:43]:<44}{d:>8.3f}{sp:>8.3f}{h:>8.3f}{flag}")


if __name__ == "__main__":
    main()
