"""Loads the post-hoc conformal calibration reports (SCRUM-68,
ml/training/calibrate.py) that already exist in reports/calibration/*.json,
and exposes them keyed by checkpoint filename.

Deliberately read-only and cheap: computing a calibration scale factor
needs a full pass over the validation split (the reports on disk took
10+ minutes each to produce), so this module never computes one itself -
it only loads what ml.training.calibrate already wrote. A checkpoint with
no report (every Phase 1 model, and any molecule trained live through
POST /jobs/train) simply has no calibration available; callers treat
that as "nothing to apply", not an error.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from app.core.config import ROOT_DIR

CALIBRATION_DIR = ROOT_DIR / "reports" / "calibration"

# checkpoint filename -> the report's own parsed content, loaded once at
# import time. Keyed by filename (not the full path recorded inside each
# report) so it matches regardless of absolute vs. relative path or which
# machine produced the report.
_CALIBRATIONS: dict[str, dict[str, Any]] = {}


def _load_all() -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    if not CALIBRATION_DIR.exists():
        return table
    for path in sorted(CALIBRATION_DIR.glob("*.json")):
        try:
            with open(path) as f:
                report = json.load(f)
            checkpoint_name = Path(report["checkpoint"]).name
        except (OSError, json.JSONDecodeError, KeyError):
            continue  # a malformed report is skipped, not a startup crash
        table[checkpoint_name] = report
    return table


_CALIBRATIONS.update(_load_all())


def get_calibration(checkpoint_path: str | Path) -> Optional[dict[str, Any]]:
    """The full report for this checkpoint (q_energy, q_force, before/after
    uq_metrics blocks), or None if it was never calibrated."""
    return _CALIBRATIONS.get(Path(checkpoint_path).name)
