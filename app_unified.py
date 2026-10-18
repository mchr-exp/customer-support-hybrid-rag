"""Milestone 7 launcher -- the ONE app that does everything.

    python app_unified.py   ->   http://127.0.0.1:7860

Verify + interrupt + cross-thread memory + supervisor routing (music & invoice)
in a single conversation. Needs OPENAI_API_KEY in .env to answer.
"""
from src.unified_ui import build_unified_ui

if __name__ == "__main__":
    build_unified_ui().launch()
