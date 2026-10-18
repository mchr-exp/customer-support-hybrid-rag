"""Entry point for the identity-verification chat demo (Milestone 6).

Run:  python app_verify.py
Then open the printed local URL (default http://127.0.0.1:7860).
Try an invalid customer id to see the graph PAUSE in the browser, then reply with
a valid one to RESUME the lookup.

(The main catalog + invoice chat is `python app.py` -- the Milestone 6 supervisor UI.)
"""
from src.hitl_ui import build_hitl_ui

if __name__ == "__main__":
    build_hitl_ui().launch()
