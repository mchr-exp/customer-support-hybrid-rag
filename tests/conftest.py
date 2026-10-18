"""Session-scoped DB init so all tests share one in-memory Chinook load."""
import os, sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import db


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: test makes a real LLM API call (needs OPENAI_API_KEY)"
    )


@pytest.fixture(scope="session", autouse=True)
def _init_db():
    db.get_engine()   # load Chinook once for the whole test session
    yield
