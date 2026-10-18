"""Milestone 2 tests: the 9 tools (17+ tests) -- found, not-found, determinism,
diversity, ordering, invalid input, and JSON validity."""
import json
import pytest
from src import tools


def _j(s):
    return json.loads(s)


# ---------------- music: albums by artist ----------------
def test_albums_by_artist_found():
    out = _j(tools.get_albums_by_artist("AC/DC"))
    assert isinstance(out, list) and len(out) >= 1


def test_albums_by_artist_fuzzy():
    # "AC" substring should still match "AC/DC"
    out = _j(tools.get_albums_by_artist("AC"))
    assert any("AC/DC" in a["Artist"] for a in out)


def test_albums_by_artist_not_found():
    out = _j(tools.get_albums_by_artist("NonexistentArtistXYZ"))
    assert "message" in out


# ---------------- music: tracks by artist ----------------
def test_tracks_by_artist_found():
    out = _j(tools.get_tracks_by_artist("AC/DC"))
    assert out["total"] >= 1 and len(out["sample"]) <= 20


def test_tracks_by_artist_not_found():
    assert "message" in _j(tools.get_tracks_by_artist("NoSuchBandZZZ"))


# ---------------- music: songs by genre ----------------
def test_songs_by_genre_found():
    out = _j(tools.get_songs_by_genre("Rock"))
    assert out["total_tracks"] > 0 and len(out["sample"]) >= 1


def test_songs_by_genre_not_found():
    assert "message" in _j(tools.get_songs_by_genre("Polka-ZZZ"))


def test_songs_by_genre_deterministic():
    a = tools.get_songs_by_genre("Rock")
    b = tools.get_songs_by_genre("Rock")
    assert a == b  # identical on repeated calls (no ORDER BY RANDOM)


def test_songs_by_genre_diversity_across_artists():
    out = _j(tools.get_songs_by_genre("Rock"))
    artists = [r["Artist"] for r in out["sample"]]
    assert len(artists) == len(set(artists)) and len(set(artists)) > 1


# ---------------- music: songs by title ----------------
def test_songs_by_title_found():
    out = _j(tools.get_songs_by_title("love"))
    assert isinstance(out, list) and len(out) >= 1 and len(out) <= 10


def test_songs_by_title_not_found():
    assert "message" in _j(tools.get_songs_by_title("zzzznotarealtitlezzz"))


# ---------------- music: track details ----------------
def test_track_details_found():
    out = _j(tools.get_track_details("1"))
    assert out[0]["TrackId"] == 1


def test_track_details_not_found():
    assert "message" in _j(tools.get_track_details("999999"))


def test_track_details_invalid_input():
    out = _j(tools.get_track_details("abc"))
    assert "message" in out and "Invalid" in out["message"]


# ---------------- invoice: invoices by customer ----------------
def test_invoices_by_customer_found():
    out = _j(tools.get_invoices_by_customer_sorted_by_date("1"))
    assert isinstance(out, list) and len(out) >= 1


def test_invoices_date_ordering_desc():
    out = _j(tools.get_invoices_by_customer_sorted_by_date("1"))
    dates = [r["InvoiceDate"] for r in out]
    assert dates == sorted(dates, reverse=True)  # most recent first


def test_invoices_by_customer_not_found():
    assert "message" in _j(tools.get_invoices_by_customer_sorted_by_date("999999"))


def test_invoices_invalid_input():
    out = _j(tools.get_invoices_by_customer_sorted_by_date("notanumber"))
    assert "message" in out and "Invalid" in out["message"]


# ---------------- invoice: purchased tracks by price ----------------
def test_purchased_tracks_found():
    out = _j(tools.get_purchased_tracks_sorted_by_price("1"))
    assert isinstance(out, list) and len(out) >= 1


def test_purchased_tracks_price_ordering_desc():
    out = _j(tools.get_purchased_tracks_sorted_by_price("1"))
    prices = [r["UnitPrice"] for r in out]
    assert prices == sorted(prices, reverse=True)


def test_purchased_tracks_not_found():
    assert "message" in _j(tools.get_purchased_tracks_sorted_by_price("999999"))


# ---------------- invoice: support rep ----------------
def test_support_rep_found():
    out = _j(tools.get_support_rep_for_invoice("1"))
    assert out[0]["RepId"] and out[0]["InvoiceId"] == 1


def test_support_rep_not_found():
    assert "message" in _j(tools.get_support_rep_for_invoice("999999"))


def test_support_rep_invalid_input():
    assert "Invalid" in _j(tools.get_support_rep_for_invoice("xyz"))["message"]


# ---------------- invoice: line items ----------------
def test_invoice_line_items_found():
    out = _j(tools.get_invoice_line_items("1"))
    assert isinstance(out, list) and len(out) >= 1


def test_invoice_line_items_not_found():
    assert "message" in _j(tools.get_invoice_line_items("999999"))


# ---------------- JSON validity across all tools ----------------
@pytest.mark.parametrize("call", [
    lambda: tools.get_albums_by_artist("AC/DC"),
    lambda: tools.get_tracks_by_artist("AC/DC"),
    lambda: tools.get_songs_by_genre("Rock"),
    lambda: tools.get_songs_by_title("love"),
    lambda: tools.get_track_details("1"),
    lambda: tools.get_track_details("abc"),
    lambda: tools.get_invoices_by_customer_sorted_by_date("1"),
    lambda: tools.get_purchased_tracks_sorted_by_price("1"),
    lambda: tools.get_support_rep_for_invoice("1"),
    lambda: tools.get_invoice_line_items("1"),
])
def test_all_tool_outputs_are_valid_json(call):
    json.loads(call())  # raises if any tool returns non-JSON
