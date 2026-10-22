from __future__ import annotations

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    molecule: str
    frame_index: int = 0
    # Which trained checkpoint answers this request - "phase1_deterministic"
    # / "phase2_blip" / "phase3_graph_stochastic" (short aliases "phase1"/
    # "phase2"/"phase3" also accepted), default Phase 1. Additive: omitting
    # it reproduces the original contract's behaviour exactly.
    phase: str | None = None


class PredictionResponse(BaseModel):
    molecule: str
    frame_index: int
    predicted_energy: float
    predicted_force_rms: float
    uncertainty: float | None
    phase: str
    # The real DFT value for this exact frame, from the gold file's test
    # split - not predicted. None only if the molecule has no trained
    # checkpoint (the request would have failed before reaching here, so
    # in practice these are always populated by a successful response).
    true_energy: float | None = None
    true_force_rms: float | None = None


class MoleculeOut(BaseModel):
    molecule: str
    num_atoms: int


class BenchmarkResult(BaseModel):
    phase: str
    model: str
    energy_mae: float
    force_mae: float
    # Original four fields (glossary.md SS4) - ece/uncertainty_correlation
    # are energy's, unchanged in meaning. Everything below is additive:
    # the fuller metrics suite ml/utils/uq_metrics.py actually computes,
    # for anyone who wants more than the four-number summary.
    ece: float | None = None
    uncertainty_correlation: float | None = None

    energy_rmse: float | None = None
    force_rmse: float | None = None
    force_ece: float | None = None
    force_uncertainty_correlation: float | None = None
    # Empirical coverage at the 90% nominal level - one number that shows
    # which direction a model is miscalibrated in (< 0.9 = overconfident,
    # > 0.9 = overcautious), which ECE alone collapses away.
    energy_coverage_90: float | None = None
    force_coverage_90: float | None = None
    # Predicted uncertainty's own typical magnitude (mean predicted std) -
    # lets two models' coverage numbers be read against how wide their
    # intervals actually are.
    energy_sharpness: float | None = None
    force_sharpness: float | None = None
    energy_nll_gaussian: float | None = None
    force_nll_gaussian: float | None = None
    energy_crps: float | None = None
    force_crps: float | None = None
