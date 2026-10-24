from __future__ import annotations

import json

from fastapi import APIRouter

from app.core.config import PHASE_CHECKPOINTS, RESULTS_DIR
from app.models.schemas import BenchmarkResult
from app.services import calibration as calibration_module

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


def _uq_field(payload: dict, key: str, field: str):
    """One number out of uq_metrics.per_key[0][key][field] - None for a
    deterministic model (`uq_metrics` itself is `None`, not just absent -
    see ml/utils/uq_metrics.py/evaluate.py) or if a results file predates
    some field being added."""
    uq_metrics = payload.get("uq_metrics") or {}
    per_key = uq_metrics.get("per_key") or []
    if not per_key:
        return None
    return (per_key[0].get(key) or {}).get(field)


def _calibrated_field(calibration: dict | None, quantity: str, field: str):
    """SCRUM-68: one number out of a calibration report's "after" block -
    same shape as _uq_field's source (both ultimately built by
    ml.utils.uq_metrics._quantity_block), just a different top-level
    path. None when this checkpoint was never calibrated."""
    if calibration is None:
        return None
    return calibration.get("after", {}).get(quantity, {}).get(field)


@router.get("", response_model=list[BenchmarkResult])
def list_benchmarks() -> list[BenchmarkResult]:
    """One row per project phase - the checkpoints /predictions serves
    (app.core.config.PHASE_CHECKPOINTS, now four: Phase 1, BLIP at K=1,
    BLIP at K=4/NLL, Phase 3) - read from each one's own scored results
    file, never recomputed here. The four summary fields (energy_mae,
    force_mae, ece, uncertainty_correlation) come from `benchmark_row`,
    the same row ml/training/evaluate.py computes; everything else comes
    from that same file's `metrics`/`uq_metrics` blocks directly, for a
    caller that wants more than the four-number summary (see
    BenchmarkResult's own comment for why each one is there).

    Not one row per architecture or per run file: a results file's own
    `phase` field is the model *registry* key (e.g. "blip"), the same for
    every BLIP run regardless of which checkpoint - reading it directly
    would produce duplicate, ambiguous rows. PHASE_CHECKPOINTS's keys are
    the unambiguous labels this endpoint (and /predictions) actually use.

    Skips a phase whose checkpoint has never been trained/scored - an
    absent row, not a crash.
    """
    results = []
    for phase, checkpoint_path in PHASE_CHECKPOINTS.items():
        results_path = RESULTS_DIR / f"{checkpoint_path.stem}.json"
        if not results_path.exists():
            continue
        with open(results_path) as f:
            payload = json.load(f)
        row = payload["benchmark_row"]
        per_key = (payload.get("metrics", {}).get("per_key") or [{}])[0]
        calibration = calibration_module.get_calibration(checkpoint_path)
        results.append(
            BenchmarkResult(
                phase=phase,
                model=row["model"],
                energy_mae=row["energy_mae"],
                force_mae=row["force_mae"],
                ece=row["ece"],
                uncertainty_correlation=row["uncertainty_correlation"],
                energy_rmse=per_key.get("energy_rmse_total"),
                force_rmse=per_key.get("force_rmse_component"),
                force_ece=_uq_field(payload, "force", "ece"),
                force_uncertainty_correlation=_uq_field(payload, "force", "spearman_uncertainty_error"),
                energy_coverage_90=((_uq_field(payload, "energy", "coverage") or {}).get("90%")),
                force_coverage_90=((_uq_field(payload, "force", "coverage") or {}).get("90%")),
                energy_sharpness=_uq_field(payload, "energy", "sharpness_mean_std"),
                force_sharpness=_uq_field(payload, "force", "sharpness_mean_std"),
                energy_nll_gaussian=_uq_field(payload, "energy", "nll_gaussian"),
                force_nll_gaussian=_uq_field(payload, "force", "nll_gaussian"),
                energy_crps=_uq_field(payload, "energy", "crps_gaussian"),
                force_crps=_uq_field(payload, "force", "crps_gaussian"),
                # SCRUM-68: post-hoc conformal calibration, fit offline on
                # the held-out validation split (ml/training/calibrate.py) -
                # None for every row with no report yet (currently: Phase 1,
                # which has no uncertainty to calibrate in the first place).
                energy_ece_calibrated=_calibrated_field(calibration, "energy", "ece"),
                force_ece_calibrated=_calibrated_field(calibration, "force", "ece"),
                energy_coverage_90_calibrated=(_calibrated_field(calibration, "energy", "coverage") or {}).get("90%"),
                force_coverage_90_calibrated=(_calibrated_field(calibration, "force", "coverage") or {}).get("90%"),
                energy_sharpness_calibrated=_calibrated_field(calibration, "energy", "sharpness_mean_std"),
                force_sharpness_calibrated=_calibrated_field(calibration, "force", "sharpness_mean_std"),
            )
        )
    return results
