"""Milestone 8 (hybrid RAG) -- dense embedding backends.

Preferred: sentence-transformers/all-MiniLM-L6-v2 (real semantic embeddings).
Fallback: TF-IDF + TruncatedSVD (LSA) via scikit-learn -- fully offline, no
model download. Both return L2-normalized dense vectors so cosine similarity
is a plain dot product.

The fallback is still a *dense/semantic-ish* representation (LSA compresses
term co-occurrence into latent topics), which keeps the dense-vs-sparse(BM25)
contrast meaningful even with no network access.
"""
from __future__ import annotations
import numpy as np
from . import config


def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class DenseIndex:
    """Cosine-similarity dense retriever over a fixed corpus."""

    def __init__(self, texts: list[str]):
        self.texts = texts
        self.backend = None
        self._model = None
        self._svd = None
        self._vec = None
        self.matrix = self._fit(texts)

    # ---- fitting ----
    def _fit(self, texts: list[str]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer  # noqa
            self._model = SentenceTransformer(config.DENSE_MODEL)
            self.backend = f"sentence-transformers:{config.DENSE_MODEL}"
            emb = self._model.encode(texts, normalize_embeddings=True,
                                     show_progress_bar=False)
            return np.asarray(emb, dtype=np.float32)
        except Exception as e:  # network/model unavailable -> LSA fallback
            print(f"[embeddings] dense model unavailable ({type(e).__name__}); "
                  f"using TF-IDF+SVD offline fallback")
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            self._vec = TfidfVectorizer(stop_words="english")
            X = self._vec.fit_transform(texts)
            dim = min(config.DENSE_FALLBACK_DIM, min(X.shape) - 1)
            self._svd = TruncatedSVD(n_components=max(2, dim), random_state=0)
            Z = self._svd.fit_transform(X)
            self.backend = f"tfidf+svd:{self._svd.n_components}d"
            return _l2(np.asarray(Z, dtype=np.float32))

    # ---- query ----
    def _embed_query(self, q: str) -> np.ndarray:
        if self._model is not None:
            v = self._model.encode([q], normalize_embeddings=True,
                                    show_progress_bar=False)
            return np.asarray(v, dtype=np.float32)
        Z = self._svd.transform(self._vec.transform([q]))
        return _l2(np.asarray(Z, dtype=np.float32))

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return [(corpus_index, cosine_score)] for the top-k docs."""
        qv = self._embed_query(query)              # (1, d)
        sims = (self.matrix @ qv.T).ravel()        # cosine (all normalized)
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]
