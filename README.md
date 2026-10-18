# Multi-Agent AI Customer Support (Digital Music Store)

A multi-agent customer-support assistant over the **Chinook** digital music store. A
routing **supervisor** delegates each question to a specialist — **music catalog**,
**invoices**, or **store policy** — and every answer is grounded in real database rows or
retrieved policy cards, never invented. On top of the core assistant it adds **hybrid
retrieval** (dense + sparse) and an **evaluation harness** that measures retrieval and
routing quality.

Built with Python 3.12 · LangGraph · LangChain + langchain-openai · SQLAlchemy (in-memory
SQLite) · Pydantic v2 · Gradio · sentence-transformers · pytest.

---

## What it does

- **Routes** a question to the right specialist (music / invoice / policy), chains more than
  one for a mixed question, and **refuses** anything off-topic at the router — before any
  specialist runs.
- **Grounds every answer**: specialists answer only from tool output (SQL rows or retrieved
  FAQ cards) under a 6-rule no-hallucination contract at `temperature=0`.
- **Verifies identity** once per session, with a human-in-the-loop **pause/resume** when the
  customer can't be identified.
- **Remembers preferences** across conversations (per-customer, cross-thread) to personalize
  tone — without letting memory become a source of facts.
- **Finds things by meaning**: a "vibe" query like *"something upbeat like AC/DC"* is served
  by hybrid retrieval (dense embeddings + BM25, fused with RRF, diversified with MMR), with
  each result carrying a real catalog id so the answer stays grounded.

---

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# the full assistant (needs an OpenAI key — see below)
python app_unified.py          # then open http://127.0.0.1:7860
```

Create a `.env` from the template and add your key (only needed for the LLM-driven agents
and UI; the database, tools, retrieval, and retrieval-eval all run with **no key**):

```bash
cp .env.example .env
# edit .env:  OPENAI_API_KEY=sk-...
```

The Chinook database is **bootstrapped automatically on first run** (the SQL script is
downloaded and a local SQLite file is built once) — there is no manual data step.

---

## The unified assistant

`app_unified.py` is the main entry point. It runs one LangGraph that stitches every
capability into a single conversation:

```
   START
     |
     v
  verify  --- already verified this session? --- no --> ask_identity --(interrupt: PAUSE)
     |                                                        ^               |
     | yes                                                    '--- resume ----'
     v
  load_memory      inject this customer's saved preferences
     |
     v
  supervisor  <-->  specialists:   music | invoice | policy
     |   route; chain for mixed; refuse off-topic
     | (final answer, no tool call)
     v
  create_memory    save any preference stated this turn
     |
     v
    END
```

Two persistence layers are attached at once: a **checkpointer** (per-session state, which
powers the verification pause/resume) and a **store** (cross-session, per-customer, which
powers preference memory).

Try it in one conversation:

| Say | What happens |
|-----|--------------|
| `what rock albums do you have?` | pauses for identity (🔒) — the verify gate |
| `my customer id is 1` | verifies, then returns real rock albums (music specialist) |
| `something upbeat like AC/DC` | hybrid semantic search returns grounded, genre-diverse picks |
| `by the way i love jazz` | preference saved to cross-session memory |
| `what's in my last invoice?` | no re-verify; invoice specialist answers from SQL |
| `how do I return a CD?` | policy specialist answers from the FAQ cards |
| `what's the weather in Seattle?` | refused — off-topic |

---

## Milestones

| # | Milestone | What it adds |
|---|-----------|--------------|
| 1 | Database foundation | in-memory Chinook via SQLAlchemy; parameterized `run_query_safe` |
| 2 | Agent tools | 9 tools (fuzzy search, deterministic genre sampling, JSON outputs, input validation) |
| 3 | Identity verification + human-in-the-loop | Pydantic extraction + DB confirm; interrupt/resume on failure |
| 4 | Sub-agents + supervisor | hand-built ReAct music & invoice agents; routing + off-topic refusal |
| 5 | Preference memory | cross-thread store, per-customer namespace, set-union merge |
| 6 | Gradio UI | browser chat, session isolation, Dockerfile |
| 7 | Unified graph | verify + memory + routing + policy in one flow, one app |
| 8 | Hybrid RAG | dense + BM25 + RRF + MMR; a policy/FAQ lane backed by retrieval |
| 9 | Evaluation | retrieval metrics (Recall@k / MRR / nDCG) + routing/refusal eval |

---

## Layout

```
app_unified.py           # main launcher — the full assistant (M7)
app.py                   # launcher — music + invoice supervisor only (M6)
app_verify.py            # launcher — identity-verification demo (M6)

src/config.py            # env-driven config (key, model, temperature, retrieval knobs)
src/db.py                # [M1] in-memory Chinook; run_query_safe (parameterized)
src/tools.py             # [M2] the 9 tools + semantic_catalog_search + answer_from_faqs
src/state.py             # AgentState (messages reducer + customer_id)
src/models.py            # [M3] Pydantic shapes (identity, preferences)
src/verify.py            # [M3] identifier extraction + DB-confirmed resolve
src/hitl.py              # [M3] human-in-the-loop verify graph (interrupt/resume)
src/prompts.py           # the 6 grounding rules + routing / specialist / memory prompts
src/agents.py            # [M4] ReAct music agent + shared build_llm
src/invoice_agent.py     # [M4] ReAct invoice agent
src/supervisor.py        # [M4] supervisor: routes music/invoice/policy, refuses off-topic
src/memory.py            # [M5] load/create-memory nodes + set-union preference merge
src/ui.py                # [M6] Gradio supervisor chat (session isolation)
src/hitl_ui.py           # [M6] Gradio verification chat (pause/resume)
src/unified.py           # [M7] the unified graph — everything in one flow
src/unified_ui.py        # [M7] one Gradio UI over the unified graph
src/corpus.py            # [M8] build catalog + genre + FAQ cards from Chinook
src/embeddings.py        # [M8] dense backend: sentence-transformers (+ offline fallback)
src/hybrid_retriever.py  # [M8] dense + BM25 + RRF fusion + MMR re-rank

eval/metrics.py          # [M9] Recall@k, MRR, nDCG (hand-implemented)
eval/dataset.py          # [M9] labeled query set (gold resolved against the live corpus)
eval/run_eval.py         # [M9] dense vs sparse vs hybrid vs hybrid+mmr
eval/answer_eval.py      # [M9] routing + grounded-refusal eval over the unified graph

tests/                   # offline + LLM-gated tests
Dockerfile               # python:3.12-slim, layer-cached deps
```

---

## Tests

```bash
pytest tests/ -q                 # full suite
pytest tests/ -q -k "not live"   # offline only (no API key)
```

**115 tests pass.** Most run with no API key; a handful of LLM-gated tests run with a key
(`-k "live"`). Run `pytest tests/` (not a bare `pytest`) so collection is scoped to the
project's tests.

---

## Evaluation

```bash
python -m eval.run_eval          # retrieval comparison (offline, no key)
python -m eval.answer_eval       # routing + grounded refusal (needs OPENAI_API_KEY)
```

**Retrieval** — dense vs sparse vs hybrid vs hybrid+MMR over a labeled query set, scored on
Recall@5 / MRR / nDCG@5 (local MiniLM embeddings):

```
strategy      recall@5    MRR   nDCG@5
dense            0.917   0.917   0.917
sparse           0.625   0.495   0.510
hybrid           0.917   0.787   0.818
hybrid+mmr       0.917   0.917   0.917
```

An honest read: on this small, mostly-semantic corpus a good embedding model nearly
saturates, so naive RRF fusion can slightly *dilute* a dominant retriever — the MMR re-rank
recovers it. Hybrid pays off most when the two retrievers are of comparable strength and the
query mix has more exact-token/rare-term cases where BM25 shines.

**Answer-level** — routing accuracy + grounded refusal over the unified graph:

```
route 100%   refuse 100%   ground 100%   (9 cases: music / invoice / policy / off-topic)
```

The eval reads the route from the graph's tool trace (which specialist ran; none = refused)
and treats a specialist answer as grounded (specialists are tool-bound at `temperature=0`).

---

## Docker

```bash
docker build -t cs-support .
docker run --rm -p 7860:7860 --env-file .env cs-support   # open http://127.0.0.1:7860
```

The image pins `python:3.12-slim` (Gradio needs the `audioop` stdlib module, removed in 3.13)
and copies `requirements.txt` before the source so the dependency layer caches across code
changes. The API key is passed at run time via `--env-file .env` — never baked into a layer.

---

## Design notes

- **Data safety** — every query is parameterized; a `DROP TABLE` payload is treated as a
  literal (`test_parameterized_blocks_injection`).
- **Deterministic, diverse genre browsing** — `ROW_NUMBER() OVER (PARTITION BY ArtistId)`
  returns one track per artist, identical on repeated calls.
- **No-hallucination contract** — specialists answer only from tool output; the policy lane
  refuses when no FAQ card matches; the supervisor refuses anything off-topic.
- **Grounded retrieval** — every retrieved card carries a real `album_id` / `genre_id`, so
  semantic discovery still resolves to true catalog rows.
