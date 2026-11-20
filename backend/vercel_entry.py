"""Vercel Function entrypoint for the backend service (vercel.json).

The service root is the repo root (the function needs ml/, experiments/ and
the serving gold data alongside backend/), so Vercel imports this file by
path with only the repo root and _vendor/ importable. The app's own
imports are `from app...`, which resolve relative to backend/ - the
directory uvicorn is launched from locally (CLAUDE.md, "Commands"). Put
backend/ on sys.path first, then expose the unchanged FastAPI app.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.main import app  # noqa: E402

__all__ = ["app"]
