from __future__ import annotations

import json

from fastapi import APIRouter

from app.core.config import PHASE_CHECKPOINTS, RESULTS_DIR
from app.models.schemas import BenchmarkResult

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


@router.get("", response_model=list[BenchmarkResult])
def list_benchmarks() -> list[BenchmarkResult]:
    """One row per project phase (Phase 1/2/3) - the same three checkpoints
    /predictions serves (app.core.config.PHASE_CHECKPOINTS) - read from
    each one's own scored results file, never recomputed here.

    Not one row per architecture or per run file: a results file's own
    `phase` field is the model *registry* key (e.g. "blip"), which is the
    same for phase2_blip and phase2_blip_nll even though they are two
    different runs - reading that field directly, as the previous version
    of this endpoint did, would have produced duplicate, ambiguous rows
    (and crashed outright: it also read `payload["model"]`/`payload["metrics"]`,
    which do not exist in the real results schema - `payload["benchmark_row"]`
    does, see ml/training/evaluate.py).

    Skips a phase whose checkpoint has never been trained/scored - an
    absent row, not a crash - so the table degrades gracefully instead of
    failing outright if e.g. Phase 3 hasn't been scored yet.
    """
    results = []
    for phase, checkpoint_path in PHASE_CHECKPOINTS.items():
        results_path = RESULTS_DIR / f"{checkpoint_path.stem}.json"
        if not results_path.exists():
            continue
        with open(results_path) as f:
            payload = json.load(f)
        row = payload["benchmark_row"]
        results.append(
            BenchmarkResult(
                phase=phase,
                model=row["model"],
                energy_mae=row["energy_mae"],
                force_mae=row["force_mae"],
                ece=row["ece"],
                uncertainty_correlation=row["uncertainty_correlation"],
            )
        )
    return results
