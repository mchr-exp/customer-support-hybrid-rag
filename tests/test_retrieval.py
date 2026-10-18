"""Milestone 8 -- hybrid retrieval tests (offline; no OPENAI_API_KEY needed).

Split by cost:
  * corpus + RRF tests touch only SQLite / pure math -> fast, deterministic.
  * retriever/tool tests build embeddings ONCE via a module-scoped fixture that
    shares the same lazy singleton the tool uses.
"""
import json
import pytest

from src import tools
from src.corpus import build_corpus
from src.hybrid_retriever import HybridRetriever


# ----------------------- corpus (SQLite only, no embeddings) -----------------
def test_corpus_has_three_card_kinds():
    docs = build_corpus()
    kinds = {d.kind for d in docs}
    assert {"album", "genre", "faq"} <= kinds
    assert len(docs) > 50          # Chinook has hundreds of albums


def test_album_card_carries_real_id():
    docs = build_corpus()
    album = next(d for d in docs if d.kind == "album")
    assert album.id.startswith("album:")
    assert isinstance(album.meta["album_id"], int)


def test_genre_card_carries_real_id():
    docs = build_corpus()
    genre = next(d for d in docs if d.kind == "genre")
    assert genre.id.startswith("genre:")
    assert isinstance(genre.meta["genre_id"], int)


# ----------------------- RRF (pure function, deterministic) ------------------
# A=0  B=1  C=2  D=3 ;  dense list [A,B,C]  sparse list [B,D,A]
def test_rrf_prefers_items_in_both_lists():
    fused = HybridRetriever.rrf([[0, 1, 2], [1, 3, 0]], k=60)
    # B is #1 in sparse and #2 in dense -> highest fused score
    assert max(fused, key=fused.get) == 1


def test_rrf_matches_formula():
    fused = HybridRetriever.rrf([[0, 1, 2], [1, 3, 0]], k=60)
    assert fused[0] == pytest.approx(1 / 61 + 1 / 63)   # A: dense#0 + sparse#2
    assert fused[1] == pytest.approx(1 / 62 + 1 / 61)   # B: dense#1 + sparse#0
    assert fused[2] == pytest.approx(1 / 63)            # C: dense#2 only
    assert fused[3] == pytest.approx(1 / 62)            # D: sparse#1 only


# ----------------------- retriever + tool (embeddings built once) ------------
@pytest.fixture(scope="module")
def retr():
    # shares the tool's lazy singleton -> one embedding build for the module
    return tools._get_retriever()


def test_hybrid_search_returns_valid_indices(retr):
    hits = retr.hybrid_search("rock albums", k=5, pool=20, use_mmr=True)
    assert 0 < len(hits) <= 5
    for idx, score in hits:
        assert 0 <= idx < len(retr.docs)


def test_semantic_tool_returns_grounded_json():
    out = json.loads(tools.semantic_catalog_search("jazz albums"))
    assert out["results"]
    for r in out["results"]:                 # every card carries a real id
        assert r["id"]
        assert ("album_id" in r) or ("genre_id" in r) or r["kind"] == "faq"


def test_acdc_vibe_query_surfaces_real_item():
    # Milestone 8 done-when: a vibe query the SQL tools can't express returns a real,
    # relevant catalog item -- AC/DC's own card.
    out = json.loads(tools.semantic_catalog_search("something upbeat like AC/DC"))
    assert "ac/dc" in json.dumps(out).lower()


def test_empty_query_is_handled():
    out = json.loads(tools.semantic_catalog_search("   "))
    assert "message" in out
