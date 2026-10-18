"""Milestone 8 (hybrid RAG) -- build the retrieval corpus from the Chinook DB
plus a small support FAQ.

Each document is a small self-contained "card" the retriever can match on:
  - album cards : artist, title, genres, a few sample track names
  - genre cards : what a genre contains + a few representative artists
  - faq cards   : returns / refunds / shipping / account / invoice policy text

GROUNDING: every card carries a stable string id AND real DB ids in `meta`
(album_id / genre_id), so when the retriever surfaces a card the agent still
answers from true catalog rows -- discovery is fuzzy, the ids keep it honest.
"""
from __future__ import annotations
import sqlite3
from dataclasses import dataclass, field
from . import config


@dataclass
class Doc:
    id: str
    text: str
    kind: str                      # "album" | "genre" | "faq"
    meta: dict = field(default_factory=dict)


def _ensure_sqlite() -> None:
    """Materialize the file-backed Chinook SQLite from the SQL script if missing.

    Corpus building reads a read-only *file* database (``config.CHINOOK_DB``). On
    a fresh checkout that file does not exist (it is git-ignored), so we build it
    once from the SQL script -- downloading the script first if needed, the same
    source ``db.py`` loads into its in-memory engine. This keeps ``clone -> run``
    working with no manual data-setup step.
    """
    if config.CHINOOK_DB.exists():
        return
    if not config.CHINOOK_SQL.exists():
        from urllib.request import urlretrieve
        urlretrieve(config.CHINOOK_SQL_URL, config.CHINOOK_SQL)
    sql = config.CHINOOK_SQL.read_text(encoding="utf-8", errors="ignore")
    con = sqlite3.connect(config.CHINOOK_DB)
    try:
        con.executescript(sql)
        con.commit()
    finally:
        con.close()


def _conn() -> sqlite3.Connection:
    # read-only URI: card building never mutates the DB
    _ensure_sqlite()
    c = sqlite3.connect(f"file:{config.CHINOOK_DB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def _album_docs(c: sqlite3.Connection) -> list[Doc]:
    docs: list[Doc] = []
    rows = c.execute(
        """
        SELECT al.AlbumId AS aid, al.Title AS title, ar.Name AS artist
        FROM Album al JOIN Artist ar ON ar.ArtistId = al.ArtistId
        ORDER BY al.AlbumId
        """
    ).fetchall()
    for r in rows:
        aid = r["aid"]
        genres = [g[0] for g in c.execute(
            """SELECT DISTINCT g.Name FROM Track t
               JOIN Genre g ON g.GenreId = t.GenreId
               WHERE t.AlbumId = ? ORDER BY g.Name""", (aid,)).fetchall() if g[0]]
        tracks = [t[0] for t in c.execute(
            "SELECT Name FROM Track WHERE AlbumId = ? LIMIT 5", (aid,)).fetchall()]
        genre_str = ", ".join(genres) if genres else "Unknown"
        text = (
            f"Album: {r['title']} by {r['artist']}. "
            f"Genre(s): {genre_str}. "
            f"Sample tracks: {', '.join(tracks)}."
        )
        docs.append(Doc(id=f"album:{aid}", text=text, kind="album",
                        meta={"album_id": aid, "artist": r["artist"],
                              "title": r["title"], "genres": genres}))
    return docs


def _genre_docs(c: sqlite3.Connection) -> list[Doc]:
    docs: list[Doc] = []
    rows = c.execute("SELECT GenreId AS gid, Name AS name FROM Genre ORDER BY GenreId").fetchall()
    for r in rows:
        gid, name = r["gid"], r["name"]
        artists = [a[0] for a in c.execute(
            """SELECT DISTINCT ar.Name FROM Track t
               JOIN Album al ON al.AlbumId = t.AlbumId
               JOIN Artist ar ON ar.ArtistId = al.ArtistId
               WHERE t.GenreId = ? LIMIT 6""", (gid,)).fetchall()]
        n_tracks = c.execute("SELECT COUNT(*) FROM Track WHERE GenreId = ?", (gid,)).fetchone()[0]
        text = (
            f"Genre: {name}. The store carries {n_tracks} {name} tracks. "
            f"Representative artists include: {', '.join(artists)}."
        )
        docs.append(Doc(id=f"genre:{gid}", text=text, kind="genre",
                        meta={"genre_id": gid, "name": name}))
    return docs


# Hand-written support policy FAQ so the agent can answer non-catalog questions
# and so "no hallucination" (grounded refusal) can be tested.
_FAQ = [
    ("faq:returns",
     "Returns policy: Physical media can be returned within 30 days of purchase "
     "for a full refund if unopened. Opened items are eligible for store credit only."),
    ("faq:refunds",
     "Refunds policy: Refunds are issued to the original payment method within 5-7 "
     "business days after we receive the returned item. Digital purchases are non-refundable."),
    ("faq:shipping",
     "Shipping policy: Standard shipping takes 5-8 business days. Expedited shipping "
     "(2 business days) is available at checkout for an extra fee. We ship to the US and Canada."),
    ("faq:account",
     "Account help: You can reset your password from the login page using 'Forgot password'. "
     "Update your billing address under Account > Profile before placing a new order."),
    ("faq:invoice",
     "Invoices: Every purchase generates an invoice with line items, unit prices, and a total. "
     "You can view past invoices under Account > Order History."),
]


def _faq_docs() -> list[Doc]:
    return [Doc(id=i, text=t, kind="faq", meta={}) for i, t in _FAQ]


def build_corpus() -> list[Doc]:
    with _conn() as c:
        docs = _album_docs(c) + _genre_docs(c) + _faq_docs()
    return docs


if __name__ == "__main__":
    d = build_corpus()
    from collections import Counter
    print(f"{len(d)} docs:", dict(Counter(x.kind for x in d)))
    for x in d[:3] + d[-6:]:
        print(f"  [{x.id}] {x.text[:90]}")
