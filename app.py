"""Entry point for the Gradio chat UI.

Run:  python app.py
Then open the printed local URL (default http://127.0.0.1:7860) in your browser.
Open it in two tabs to see that each session is independent (session-isolation goal, Milestone 6).
"""
from src.ui import build_ui

if __name__ == "__main__":
    build_ui().launch()
