import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXAMPLE_ROOT.parents[1]
for path in (REPO_ROOT, EXAMPLE_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

os.environ.setdefault("MANDOL_CHAT_MOCK_LLM", "true")
os.environ.setdefault("MANDOL_CHAT_REAL_EMBEDDING", "false")
os.environ.setdefault("MANDOL_CHAT_DATA_DIR", str(EXAMPLE_ROOT / "mandol_chat" / "data"))

from mandol_chat.main import app  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        test_client.post("/api/reset")
        yield test_client
        test_client.post("/api/reset")

