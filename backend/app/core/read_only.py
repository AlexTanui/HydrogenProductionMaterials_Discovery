"""Guard for endpoints that write to disk or spawn training processes.

The hosted deployment sets READ_ONLY=1 (see Settings.read_only): there the
filesystem is read-only outside /tmp and no request can outlive the
function's time limit, so training jobs and data uploads only work in a
local run. Failing fast with a 503 that says so beats a 500 halfway
through writing a bronze file or launching a subprocess that gets killed.
"""
from __future__ import annotations

from fastapi import HTTPException

from app.core.config import settings


def require_writable() -> None:
    if settings.read_only:
        raise HTTPException(
            503,
            "This hosted deployment is read-only: training jobs and data uploads are only "
            "available when the backend runs locally (see CLAUDE.md, 'Commands').",
        )
