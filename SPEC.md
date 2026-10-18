# Project 4 — Product Spec & Requirements
## Multi-Agent Customer Support for a Digital Music Store (with Hybrid Retrieval)

**Status legend:** ✅ Built & tested · ◐ Partial · ⏳ Planned
**Audience:** engineers building/grading the capstone, and reviewers who need the "why."
All terms are defined inline; the doc is readable cold.

---

## 0. Working backwards — the experience we're building

A customer opens a chat with the music store and just *talks*: "I keep forgetting my
password," "what upbeat rock do you have," "what was on my last order." Behind that one
box, a **supervisor** agent routes each message to a **specialist** — a **Music Catalog
Agent** or an **Invoice Agent** — that answers **only** from the real store database. The
customer proves who they are once; after that the system remembers their taste and never
makes up an album, a price, or an invoice line.

The one hard promise: **every fact in an answer traces to a real database row.** Nothing
is invented.

Two things make catalog discovery feel good rather than brittle:
- **Structured lookups** (exact artist/genre/title) come from **SQL tools**.
- **Fuzzy, natural-language discovery** ("something like AC/DC", "chill jazz for studying",
  "heavy metal") comes from **hybrid retrieval** — which is the focus of this spec.

**Terms used throughout**
- **Agent** — an LLM plus a fixed set of tools and a scoped instruction prompt.
- **Supervisor** — a router agent that decides which specialist handles a message.
- **Tool** — a plain function the agent may call (here: a parameterized DB query).
- **Grounding** — requiring the model to answer only from tool output, never memory.
- **Dense retrieval** — ranking documents by *meaning* using embedding vectors (so "CD" ≈
  "media", "heavy metal" ≈ "metal/rock").
- **Sparse retrieval (BM25)** — classic keyword scoring over exact tokens; great on rare
  exact terms, but no stemming (so the token "refund" ≠ "refunds").
- **Hybrid retrieval** — run dense and sparse together and **fuse** their ranked lists.
- **RRF (Reciprocal Rank Fusion)** — fuse two ranked lists by summing `1/(k+rank)`;
  rank-based, so the two scorers' incompatible score scales don't matter.
- **MMR (Maximal Marginal Relevance)** — a re-rank that trades relevance against diversity
  so results aren't near-duplicates.

---

## 1. Personas

- **Maya — returning customer.** Knows what she likes ("jazz, AC/DC") but not exact titles;
  describes music in plain words; wants her past orders fast.
- **Sam — new customer.** Browses by vibe ("something upbeat"), often misspells names.
- **The store (business owner).** Needs answers that are accurate, account-secure, and
  personalized — hallucinated data destroys trust.
- **Devin — engineer/grader.** Needs the system modular, safe (no SQL injection), testable,
  and needs retrieval quality to be *measurable*, not asserted.

---

## 2. User stories & acceptance criteria

IDs: **V** verification · **C** catalog · **H** hybrid retrieval · **I** invoices ·
**M** memory · **G** grounding · **U** UI · **E** eval.

### Identity & security
- **V-1** ⏳ *As Maya, I want to verify with my Customer ID, email, or phone so I can see my
  account.*
  AC: numeric ID, case-insensitive email, and normalized phone all verify; a bad identifier
  triggers a graceful retry (HITL interrupt); the verified ID is injected into state via a
  `SystemMessage` and is **never** parsed from my free text.

### Catalog — structured lookups (SQL)
- **C-1** ✅ *As Sam, I want to find albums by an artist even if I type only part of the
  name.* AC: `get_albums_by_artist("AC")` matches "AC/DC" (fuzzy LIKE); no match returns a
  friendly JSON message.
- **C-2** ✅ *As Maya, I want to browse a genre and see variety, not ten tracks by one band.*
  AC: `get_songs_by_genre("Rock")` returns one representative track per artist (up to 10),
  **identical on every call** (deterministic via `ROW_NUMBER()`), and states the true total.
- **C-3** ✅ *As Sam, I want to search a song by a title fragment and get track details by ID.*

### Catalog — **hybrid retrieval** (the highlight)
- **H-1** ⏳ *As Sam, I want to discover music by describing it in my own words
  ("upbeat 80s rock", "something like AC/DC", "chill jazz for studying") so I don't need to
  know exact artist, genre, or title spellings.*
  AC: a natural-language query returns real catalog items ranked by relevance; the answer
  cites only those real items (grounded); works when the words don't literally appear in the
  catalog text.
- **H-2** ⏳ *As Sam, I want misspellings and synonyms to still find the right thing
  ("heavy metal" → metal/rock, "CD" → media).*
  AC: queries whose intent is lexical (an exact rare token like an artist name) **and**
  queries whose intent is semantic (a paraphrase) both succeed — because dense and sparse
  cover each other's blind spots and are fused.
- **H-3** ⏳ *As Maya, I don't want three near-identical results in a row.*
  AC: an optional MMR re-rank de-duplicates the shortlist while keeping relevance.
- **H-4** ✅ *As Devin, I want retrieval quality to be measurable so I can justify using
  hybrid over a single retriever.* AC: an eval harness reports Recall@k / MRR / nDCG for
  dense-only vs sparse-only vs hybrid vs hybrid+MMR on a labeled query set.

### Invoices (verified only)
- **I-1** ⏳ *As Maya, I want my invoices most-recent-first, my purchases by price, the
  support rep for an invoice, and an invoice's line items.* AC: all four require a verified
  ID; dates sort descending; prices sort descending; results are grounded and JSON-valid.

### Memory
- **M-1** ⏳ *As Maya, I want the store to remember my tastes across chats.* AC: explicit
  preferences ("I love jazz") are saved per-customer and **merged** (set union, never
  removed); questions ("do you have jazz?") are **not** saved; an empty extraction never
  erases history.

### Grounding (anti-hallucination) — applies to every agent
- **G-1** ⏳ *As the store, I want agents to answer only from tool output.* AC: every agent
  prompt carries the six grounding rules (tool-only, exact quoting, honest empty results,
  scope boundaries, no memory answers, truncation transparency); temperature = 0.
- **G-2** ⏳ *As the store, I want off-topic questions declined without wasting an agent.* AC:
  the supervisor rejects "what's the weather" directly.

### UI
- **U-1** ⏳ *As any user, I want a chat with session isolation, graceful verification
  pauses, a reset, and status feedback.* AC: each browser tab gets a unique thread ID; the
  interrupt/resume cycle works; "New conversation" resets state.

---

## 3. Where hybrid retrieval fits (architecture)

```
                         ┌─────────────┐
   user ── message ────▶ │ SUPERVISOR  │  routes: catalog / invoice / mixed / reject
                         └──────┬──────┘
              ┌─────────────────┴───────────────────┐
              ▼                                       ▼
   ┌───────────────────────┐               ┌───────────────────────┐
   │  MUSIC CATALOG AGENT   │               │     INVOICE AGENT      │
   │  tools:                │               │  tools (verified id):  │
   │   • SQL structured     │               │   • invoices by date   │
   │     (C-1/C-2/C-3)      │               │   • purchases by price │
   │   • HYBRID SEARCH ◀────┼── H-1..H-3    │   • support rep        │
   │     (semantic discovery)│              │   • line items         │
   └───────────┬───────────┘               └───────────┬───────────┘
               ▼                                         ▼
        ┌──────────────┐                          ┌──────────────┐
        │ hybrid layer │  dense + BM25 → RRF → MMR │  SQL (params)│
        └──────┬───────┘                          └──────────────┘
               ▼
     candidate catalog cards (each carries a REAL AlbumId/TrackId)
               ▼
     agent answers using ONLY those grounded items  ── no hallucination
```

**Key design decision:** hybrid retrieval is an **additional tool on the Music Catalog
Agent**, not a replacement for SQL. The store's "never make up data" rule is preserved
because retrieval only *selects which real catalog items* are relevant — every candidate
card is built from actual DB rows and carries the real IDs, so the agent's facts still come
from the database, not the model.

- **SQL tools** answer *structured* asks ("albums by AC/DC", "rock genre sample").
- **Hybrid search** answers *open-ended* asks ("something upbeat like AC/DC") that SQL `LIKE`
  can't, because `LIKE` is substring-only — it can't match paraphrase, synonym, or intent.

---

## 4. How the hybrid layer works (mechanism)

1. **Corpus.** Build one compact "card" per album and per genre (artist, title, genres,
   sample tracks) plus support-policy FAQ cards. Each card stores its real DB id.
2. **Two retrievers, run in parallel:**
   - *Dense* — embed the query and every card; rank by cosine similarity (semantic).
   - *Sparse* — BM25 over tokenized card text (exact lexical).
3. **Fuse with RRF** — for each card, `score = Σ 1/(RRF_K + rank_in_list)` across both
   lists. Rank-based fusion avoids having to normalize two incompatible score scales.
4. **Optional MMR re-rank** — greedily pick items maximizing
   `λ·relevance − (1−λ)·max_similarity_to_already_picked` to kill near-duplicates.
5. **Return top-k grounded cards** to the agent, which answers under the grounding rules.

**Why both, not one** (measured, see §6):
- BM25 alone misses morphology/synonyms — the token "refund" ≠ "refunds", so a refund query
  can score **0** on BM25 while dense nails it.
- Dense alone can miss an exact rare token — e.g. an artist name ("Accept") when the card's
  genre text says "Rock", not "heavy metal"; BM25 catches it.
- Fusing covers both; MMR keeps the shortlist diverse.

**Config knobs:** `RRF_K` (fusion flatness), retrieval `pool` size, `k` (results returned),
`use_mmr`, `mmr_lambda`. Dense backend is `all-MiniLM-L6-v2` with an offline TF-IDF+SVD
fallback so retrieval runs without network or API keys.

---

## 5. Functional requirements

- **FR-1** Supervisor routes catalog / invoice / mixed correctly and rejects off-topic. ⏳
- **FR-2** Music agent exposes 5 SQL tools **and** 1 hybrid-search tool. ✅ SQL / ⏳ tool-wiring
- **FR-3** Invoice agent exposes 4 tools, all requiring a verified customer id. ✅ tools / ⏳ gating
- **FR-4** Verification supports id/email/phone with normalization + HITL interrupt. ⏳
- **FR-5** Per-customer preference memory with set-union merge. ⏳
- **FR-6** Every agent prompt embeds the 6 grounding rules; temperature 0. ⏳
- **FR-7** Hybrid retrieval fuses dense+BM25 via RRF, optional MMR, returns grounded cards. ✅
- **FR-8** All DB access parameterized; numeric inputs validated. ✅

## 6. Non-functional requirements & acceptance

- **NFR-1 Data safety.** No string-built SQL anywhere; injection payloads treated as
  literals. ✅ (`test_parameterized_blocks_injection`)
- **NFR-2 Determinism.** Genre sampling identical across calls. ✅
- **NFR-3 Retrieval quality (hybrid).** On the labeled eval set, hybrid must **match or beat
  the better single retriever on Recall@k**, and hybrid+MMR must not regress MRR.
  ◐ current (local MiniLM):

  | strategy | Recall@5 | MRR | nDCG@5 |
  |---|---|---|---|
  | dense | 0.917 | 0.917 | 0.917 |
  | sparse (BM25) | 0.625 | 0.495 | 0.510 |
  | hybrid (RRF) | 0.917 | 0.787 | 0.818 |
  | hybrid + MMR | 0.917 | 0.917 | 0.917 |

  Reading: this corpus is semantically easy, so dense dominates; hybrid matches its recall
  and is robust across query *types*. Honest caveat: naive RRF can slightly dilute a
  dominant retriever (hybrid MRR 0.787 < dense) — motivating a re-ranker / weighted fusion
  (see §8).
- **NFR-4 No hallucination.** Grounded/refusal eval passes: grounded answers cite ≥1 source;
  off-topic queries refuse. ✅ (100% on the current answer-eval)
- **NFR-5 Testing.** ≥28 automated tests, all passing, keyless. ✅ (49 passing)
- **NFR-6 Deployment.** `python app.py` starts the app; Dockerfile on `python:3.12-slim`;
  env-driven config. ⏳ (config ✅)

## 7. Eval plan (proves the hybrid claim)

- **Retrieval eval** (`eval/run_eval.py`): labeled query→gold-doc set; reports Recall@k /
  MRR / nDCG for dense / sparse / hybrid / hybrid+MMR, plus a per-query table showing exactly
  where one retriever rescues another. The query mix is deliberately split into
  synonym/morphology cases (favor dense) and exact-rare-token cases (favor sparse).
- **Answer eval** (`eval/answer_eval.py`): routing accuracy, grounded-refusal correctness,
  citation presence, and a faithfulness check (gold facts appear in cited context).

## 8. Out of scope / future

- Cross-encoder **re-ranker** on the fused shortlist (expected to close the hybrid-vs-dense
  MRR gap in NFR-3).
- **Weighted RRF** (`w_dense`, `w_sparse`) with a weight sweep vs MRR.
- Persistent memory backend (spec allows in-memory for the capstone).
- RAGAS/DeepEval faithfulness + context precision/recall on top of the hand-rolled metrics.

## 9. Rubric traceability

| Rubric dimension (weight) | Stories/Reqs | Status |
|---|---|---|
| Multi-Agent Architecture (20%) | FR-1, G-2 | ⏳ |
| Tool Design & Data Safety (20%) | C-1..3, I-1, FR-8, NFR-1/2 | ✅ |
| Verification & HITL (15%) | V-1, FR-4 | ⏳ |
| Memory (10%) | M-1, FR-5 | ⏳ |
| Prompt Engineering / grounding (15%) | G-1/2, FR-6 | ⏳ |
| Testing (10%) | NFR-5 | ✅ |
| Code Quality & Deployment (10%) | NFR-6 | ◐ |
| *Expansion:* Hybrid retrieval + eval | H-1..4, FR-7, NFR-3/4, §7 | ✅ core / ⏳ tool-wiring |
