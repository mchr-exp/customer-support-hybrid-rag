"""The 9 agent tools (Milestone 2).

Design contract (rubric: Tool Design & Data Safety, 20%):
  * agents access data ONLY through these tools -- never raw SQL
  * every query is PARAMETERIZED via db.run_query_safe (no string building)
  * numeric inputs validated with _safe_int (no crashes on "abc")
  * outputs are JSON strings; not-found returns a JSON {"message": ...}
  * genre browsing is DETERMINISTIC and DIVERSE (one track per artist)

Each tool has a plain implementation function (unit-testable, no API key) and
an optional @tool wrapper for the LangGraph agents (activated if langchain-core
is installed).
"""
from __future__ import annotations
import json
from . import db, config


def _safe_int(value, field: str = "value") -> int:
    """Convert to int or raise a clear ValueError (caught by callers)."""
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        raise ValueError(f"Invalid {field}: {value}. Please provide a numeric value.")


def _msg(text: str) -> str:
    """Human-readable not-found/error as VALID JSON (passes json.loads)."""
    return json.dumps({"message": text})


# ============================ MUSIC CATALOG (5) ============================
def get_albums_by_artist(artist: str) -> str:
    """Search albums by artist name (fuzzy LIKE match)."""
    rows = db.run_query_safe(
        """SELECT al.AlbumId, al.Title AS Album, ar.Name AS Artist
           FROM Album al JOIN Artist ar ON ar.ArtistId = al.ArtistId
           WHERE ar.Name LIKE :p ORDER BY ar.Name, al.Title""",
        {"p": f"%{artist}%"},
    )
    return rows if json.loads(rows) else _msg(f"No albums found for artist: {artist}")


def get_tracks_by_artist(artist: str) -> str:
    """Search tracks by artist name: total count + a sample (up to 20)."""
    total = json.loads(db.run_query_safe(
        """SELECT COUNT(*) AS n FROM Track t
           JOIN Album al ON al.AlbumId = t.AlbumId
           JOIN Artist ar ON ar.ArtistId = al.ArtistId
           WHERE ar.Name LIKE :p""", {"p": f"%{artist}%"}))[0]["n"]
    if not total:
        return _msg(f"No tracks found for artist: {artist}")
    sample = json.loads(db.run_query_safe(
        """SELECT t.TrackId, t.Name AS Track, al.Title AS Album, ar.Name AS Artist
           FROM Track t JOIN Album al ON al.AlbumId = t.AlbumId
           JOIN Artist ar ON ar.ArtistId = al.ArtistId
           WHERE ar.Name LIKE :p ORDER BY t.TrackId LIMIT :lim""",
        {"p": f"%{artist}%", "lim": config.TRACK_SAMPLE_LIMIT}))
    return json.dumps({"artist_query": artist, "total": total,
                       "sample_size": len(sample), "sample": sample})


def get_songs_by_genre(genre: str) -> str:
    """Browse songs by genre: ONE representative track per artist (up to 10),
    deterministic via ROW_NUMBER() partitioned by artist, ordered by TrackId."""
    total = json.loads(db.run_query_safe(
        """SELECT COUNT(*) AS n FROM Track t JOIN Genre g ON g.GenreId = t.GenreId
           WHERE g.Name = :g COLLATE NOCASE""", {"g": genre}))[0]["n"]
    if not total:
        return _msg(f"No songs found for genre: {genre}")
    sample = json.loads(db.run_query_safe(
        """WITH ranked AS (
             SELECT ar.ArtistId, ar.Name AS Artist, t.Name AS Track, t.TrackId,
                    ROW_NUMBER() OVER (PARTITION BY ar.ArtistId ORDER BY t.TrackId) AS rn
             FROM Track t
             JOIN Album al ON al.AlbumId = t.AlbumId
             JOIN Artist ar ON ar.ArtistId = al.ArtistId
             JOIN Genre g ON g.GenreId = t.GenreId
             WHERE g.Name = :g COLLATE NOCASE)
           SELECT Artist, Track, TrackId FROM ranked WHERE rn = 1
           ORDER BY Artist, TrackId LIMIT :lim""",
        {"g": genre, "lim": config.GENRE_SAMPLE_ARTISTS}))
    return json.dumps({"genre": genre, "total_tracks": total,
                       "sample_note": f"Showing {len(sample)} tracks (one per artist); "
                                      f"{total} total tracks exist in this genre.",
                       "sample": sample})


def get_songs_by_title(title: str) -> str:
    """Search songs by title (fuzzy LIKE), up to 10 results."""
    rows = db.run_query_safe(
        """SELECT t.TrackId, t.Name AS Track, ar.Name AS Artist, al.Title AS Album
           FROM Track t JOIN Album al ON al.AlbumId = t.AlbumId
           JOIN Artist ar ON ar.ArtistId = al.ArtistId
           WHERE t.Name LIKE :p ORDER BY t.Name LIMIT :lim""",
        {"p": f"%{title}%", "lim": config.TITLE_SEARCH_LIMIT})
    return rows if json.loads(rows) else _msg(f"No songs found matching title: {title}")


def get_track_details(track_id: str) -> str:
    """Full details for a specific track by ID."""
    try:
        tid = _safe_int(track_id, "track ID")
    except ValueError as e:
        return _msg(str(e))
    rows = db.run_query_safe(
        """SELECT t.TrackId, t.Name AS Track, ar.Name AS Artist, al.Title AS Album,
                  g.Name AS Genre, mt.Name AS MediaType,
                  t.Composer, t.Milliseconds, t.Bytes, t.UnitPrice
           FROM Track t
           LEFT JOIN Album al ON al.AlbumId = t.AlbumId
           LEFT JOIN Artist ar ON ar.ArtistId = al.ArtistId
           LEFT JOIN Genre g ON g.GenreId = t.GenreId
           LEFT JOIN MediaType mt ON mt.MediaTypeId = t.MediaTypeId
           WHERE t.TrackId = :tid""", {"tid": tid})
    data = json.loads(rows)
    return rows if data else _msg(f"No track found with ID: {tid}")


# ============================ INVOICE (4) ============================
def get_invoices_by_customer_sorted_by_date(customer_id: str) -> str:
    """All invoices for a customer, most recent first."""
    try:
        cid = _safe_int(customer_id, "customer ID")
    except ValueError as e:
        return _msg(str(e))
    rows = db.run_query_safe(
        """SELECT InvoiceId, InvoiceDate, Total, BillingCity, BillingCountry
           FROM Invoice WHERE CustomerId = :cid
           ORDER BY InvoiceDate DESC, InvoiceId DESC""", {"cid": cid})
    return rows if json.loads(rows) else _msg(f"No invoices found for customer: {cid}")


def get_purchased_tracks_sorted_by_price(customer_id: str) -> str:
    """Every purchased track for a customer, highest unit price first
    (each row is one purchased track, not a whole invoice)."""
    try:
        cid = _safe_int(customer_id, "customer ID")
    except ValueError as e:
        return _msg(str(e))
    rows = db.run_query_safe(
        """SELECT il.InvoiceId, t.Name AS Track, ar.Name AS Artist,
                  il.UnitPrice, il.Quantity
           FROM InvoiceLine il
           JOIN Invoice i ON i.InvoiceId = il.InvoiceId
           JOIN Track t ON t.TrackId = il.TrackId
           JOIN Album al ON al.AlbumId = t.AlbumId
           JOIN Artist ar ON ar.ArtistId = al.ArtistId
           WHERE i.CustomerId = :cid
           ORDER BY il.UnitPrice DESC, t.Name""", {"cid": cid})
    return rows if json.loads(rows) else _msg(f"No purchased tracks found for customer: {cid}")


def get_support_rep_for_invoice(invoice_id: str) -> str:
    """Support representative associated with a specific invoice."""
    try:
        iid = _safe_int(invoice_id, "invoice ID")
    except ValueError as e:
        return _msg(str(e))
    rows = db.run_query_safe(
        """SELECT i.InvoiceId, c.CustomerId, c.FirstName AS CustomerFirst,
                  c.LastName AS CustomerLast, e.EmployeeId AS RepId,
                  e.FirstName AS RepFirst, e.LastName AS RepLast, e.Email AS RepEmail
           FROM Invoice i
           JOIN Customer c ON c.CustomerId = i.CustomerId
           JOIN Employee e ON e.EmployeeId = c.SupportRepId
           WHERE i.InvoiceId = :iid""", {"iid": iid})
    return rows if json.loads(rows) else _msg(f"No invoice found with ID: {iid}")


def get_invoice_line_items(invoice_id: str) -> str:
    """Detailed line items (purchased tracks) for a specific invoice."""
    try:
        iid = _safe_int(invoice_id, "invoice ID")
    except ValueError as e:
        return _msg(str(e))
    rows = db.run_query_safe(
        """SELECT il.InvoiceLineId, t.Name AS Track, ar.Name AS Artist,
                  il.UnitPrice, il.Quantity
           FROM InvoiceLine il
           JOIN Track t ON t.TrackId = il.TrackId
           JOIN Album al ON al.AlbumId = t.AlbumId
           JOIN Artist ar ON ar.ArtistId = al.ArtistId
           WHERE il.InvoiceId = :iid ORDER BY il.InvoiceLineId""", {"iid": iid})
    return rows if json.loads(rows) else _msg(f"No line items found for invoice: {iid}")


# ======================= HYBRID SEMANTIC SEARCH (Milestone 8) =======================
_retriever = None


def _get_retriever():
    """Build the hybrid retriever once (lazily) -- embedding the corpus is the
    expensive step, so we never do it at import time or in the db/tool tests."""
    global _retriever
    if _retriever is None:
        from .corpus import build_corpus
        from .hybrid_retriever import HybridRetriever
        _retriever = HybridRetriever(build_corpus())
    return _retriever


def semantic_catalog_search(query: str) -> str:
    """Natural-language catalog discovery via hybrid retrieval (dense + BM25 +
    RRF + MMR). Use for fuzzy/"vibe" queries the exact SQL tools can't express,
    e.g. "something upbeat like AC/DC" or "music for a rainy evening".

    Returns up to 5 matching catalog cards, each carrying a REAL album_id or
    genre_id so answers stay grounded in true catalog rows (no hallucination).
    """
    q = (query or "").strip()
    if not q:
        return _msg("Please provide a search query.")
    retr = _get_retriever()
    hits = retr.hybrid_search(q, k=5, pool=20, use_mmr=True)
    results = []
    for idx, score in hits:
        d = retr.doc(idx)
        row = {"id": d.id, "kind": d.kind, "text": d.text,
               "score": round(float(score), 4)}
        row.update(d.meta)          # real ids: album_id / genre_id, artist, etc.
        results.append(row)
    if not results:
        return _msg(f"No catalog items found for: {q}")
    return json.dumps({"query": q, "results": results})


def answer_from_faqs(query: str) -> str:
    """Grounded policy/FAQ answer (returns, refunds, shipping, account, invoices).

    Retrieves FAQ cards via the same hybrid retriever and applies a CONFIDENCE
    GATE: if the single best hit over the whole corpus is NOT a FAQ card, the
    question isn't a policy question we have an answer for -> REFUSE rather than
    stretch an unrelated card into an answer (the no-hallucination contract).
    Otherwise returns the matching FAQ card text verbatim -- deterministic and
    grounded, no LLM generation needed.
    """
    q = (query or "").strip()
    if not q:
        return "Please tell me which policy you're asking about."
    retr = _get_retriever()
    hits = retr.hybrid_search(q, k=3, pool=20, use_mmr=True)
    if not hits or retr.doc(hits[0][0]).kind != "faq":
        return ("I don't have information on that. I can help with returns, "
                "refunds, shipping, account/login, or finding your invoices.")
    faqs = [retr.doc(i).text for i, _ in hits if retr.doc(i).kind == "faq"]
    return "Based on our store policies:\n" + "\n".join(f"- {t}" for t in faqs)


# Registry used by tests + agents
MUSIC_TOOLS = [get_albums_by_artist, get_tracks_by_artist, get_songs_by_genre,
               get_songs_by_title, get_track_details, semantic_catalog_search]
INVOICE_TOOLS = [get_invoices_by_customer_sorted_by_date,
                 get_purchased_tracks_sorted_by_price,
                 get_support_rep_for_invoice, get_invoice_line_items]
ALL_TOOLS = MUSIC_TOOLS + INVOICE_TOOLS


# Optional @tool wrappers for the LangGraph agents (no-op if langchain absent)
try:
    from langchain_core.tools import tool as _lc_tool
    lc_music_tools = [_lc_tool(f) for f in MUSIC_TOOLS]
    lc_invoice_tools = [_lc_tool(f) for f in INVOICE_TOOLS]
except Exception:  # pragma: no cover
    lc_music_tools = lc_invoice_tools = None


if __name__ == "__main__":
    print("albums AC/DC:", get_albums_by_artist("AC/DC")[:200])
    print("genre Rock:", get_songs_by_genre("Rock")[:200])
    print("bad track:", get_track_details("abc"))
