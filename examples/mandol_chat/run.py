"""Local launcher for the Mandol chat demo."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn


THIS_FILE = Path(__file__).resolve()
EXAMPLE_ROOT = THIS_FILE.parent
REPO_ROOT = EXAMPLE_ROOT.parents[1]

for path in (REPO_ROOT, EXAMPLE_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from mandol_chat.config import get_config  # noqa: E402


if __name__ == "__main__":
    config = get_config()
    uvicorn.run(
        "mandol_chat.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        app_dir=str(EXAMPLE_ROOT),
    )
