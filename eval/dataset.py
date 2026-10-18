"""Labeled eval set: query -> gold-relevant doc ids.

Gold ids are resolved against the live corpus by predicate (title/genre/faq)
so we never hardcode fragile numeric ids. The query mix is deliberate:

  * some queries use SYNONYMS not present in the doc text ("CD" vs "media",
    "heavy metal" vs the literal genre) -> favors DENSE (semantic) retrieval
  * some use exact lexical terms ("Balls to the Wall", "refund") -> favors
    SPARSE (BM25) retrieval

That split is what makes HYBRID (RRF of both) beat either one alone.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from src.corpus import Doc


@dataclass
class EvalQuery:
    query: str
    gold: list[str]        # resolved doc ids
    note: str = ""


def _albums_by_title(docs, *substrs) -> list[str]:
    subs = [s.lower() for s in substrs]
    return [d.id for d in docs if d.kind == "album"
            and any(s in d.meta.get("title", "").lower() for s in subs)]


def _albums_by_artist(docs, artist) -> list[str]:
    a = artist.lower()
    return [d.id for d in docs if d.kind == "album"
            and a in d.meta.get("artist", "").lower()]


def _genre(docs, name) -> list[str]:
    return [d.id for d in docs if d.kind == "genre"
            and d.meta.get("name", "").lower() == name.lower()]


# (query, gold-resolver, note-about-which-retriever-it-favors)
_SPECS: list[tuple[str, Callable[[list[Doc]], list[str]], str]] = [
    ("heavy metal band Accept full album",
     lambda d: _albums_by_artist(d, "Accept"), "dense (synonym: heavy metal)"),
    ("AC/DC album about rocking",
     lambda d: _albums_by_artist(d, "AC/DC"), "mixed"),
    ("Balls to the Wall",
     lambda d: _albums_by_title(d, "Balls to the Wall"), "sparse (exact title)"),
    ("Aerosmith greatest hits collection",
     lambda d: _albums_by_title(d, "Big Ones"), "dense (semantic 'greatest hits')"),
    ("what rock albums do you have in store",
     lambda d: _genre(d, "Rock"), "genre card"),
    ("do you sell any jazz music",
     lambda d: _genre(d, "Jazz"), "dense"),
    ("latin music available in the store",
     lambda d: _genre(d, "Latin"), "mixed"),
    ("how do I return a CD I bought",
     lambda d: ["faq:returns"], "dense (synonym: CD vs media)"),
    ("when will I get my refund",
     lambda d: ["faq:refunds"], "sparse (exact term refund)"),
    ("how long does shipping take",
     lambda d: ["faq:shipping"], "sparse"),
    ("I forgot my password and cannot log in",
     lambda d: ["faq:account"], "dense"),
    ("where can I see my past orders and invoices",
     lambda d: ["faq:invoice"], "mixed"),
]


def build_eval_set(docs: list[Doc]) -> list[EvalQuery]:
    out: list[EvalQuery] = []
    for q, resolver, note in _SPECS:
        gold = resolver(docs)
        assert gold, f"gold empty for query: {q!r}"
        out.append(EvalQuery(query=q, gold=gold, note=note))
    return out
