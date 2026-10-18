from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # no-op if .env absent

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# Chinook SQL script (spec: load the .sql into in-memory SQLite; cache locally)
CHINOOK_SQL = DATA / "Chinook_Sqlite.sql"
CHINOOK_SQL_URL = (
    "https://raw.githubusercontent.com/lerocha/chinook-database/master/"
    "ChinookDatabase/DataSources/Chinook_Sqlite.sql"
)

# LLM (only needed to run the agents/UI, not the tools/db/tests)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")           # OpenAI-compatible endpoint
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))           # 0 => minimize hallucination
PORT = int(os.getenv("PORT", "7860"))

# Tool sampling caps (spec-defined)
GENRE_SAMPLE_ARTISTS = 10     # one representative track per artist, up to 10
TRACK_SAMPLE_LIMIT = 20       # tracks-by-artist sample size
TITLE_SEARCH_LIMIT = 10

# Hybrid retrieval (Milestone 8) -----------------------------------------
# A file-backed copy of the Chinook DB used to build the RAG "cards" corpus
# (separate from the in-memory StaticPool the agents query via db.py).
CHINOOK_DB = DATA / "Chinook_Sqlite.sqlite"
# Dense embedding model; falls back to offline TF-IDF+SVD if it can't load.
DENSE_MODEL = os.getenv("DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DENSE_FALLBACK_DIM = int(os.getenv("DENSE_FALLBACK_DIM", "128"))  # LSA latent dims
# Reciprocal Rank Fusion constant: larger k => flatter rank weighting.
RRF_K = int(os.getenv("RRF_K", "60"))