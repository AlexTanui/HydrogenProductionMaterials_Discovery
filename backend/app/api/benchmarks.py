from __future__ import annotations

import json

from fastapi import APIRouter

from app.core.config import ROOT_DIR
from app.models.schemas import BenchmarkResult

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])

RESULTS_DIR = ROOT_DIR / "experiments" / "results"


@router.get("", response_model=list[BenchmarkResult])
def list_benchmarks() -> list[BenchmarkResult]:
    if not RESULTS_DIR.exists():
        return []

    results = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with open(path) as f:
            payload = json.load(f)
        results.append(
            BenchmarkResult(phase=payload["phase"], model=payload["model"], **payload["metrics"])
        )
    return results
