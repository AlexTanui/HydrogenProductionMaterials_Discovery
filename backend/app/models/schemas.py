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
    # SCRUM-68: `uncertainty` scaled by a post-hoc conformal correction
    # fit offline on the held-out validation split - None when this
    # checkpoint was never calibrated (ml/training/calibrate.py hasn't
    # been run against it, e.g. a molecule just trained live). Additive -
    # `uncertainty` keeps its original, uncalibrated meaning.
    uncertainty_calibrated: float | None = None


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

    # SCRUM-68: post-hoc conformal calibration (ml/training/calibrate.py,
    # fit offline on the held-out validation split) - None for a row with
    # no calibration report yet.
    energy_ece_calibrated: float | None = None
    force_ece_calibrated: float | None = None
    energy_coverage_90_calibrated: float | None = None
    force_coverage_90_calibrated: float | None = None
    energy_sharpness_calibrated: float | None = None
    force_sharpness_calibrated: float | None = None


class TrainJobRequest(BaseModel):
    molecule: str
    # Registry phase to train (mpnn/schnet/painn/physnet/dimenet/blip/
    # graph_stochastic) - GET /jobs/phases lists the real set.
    phase: str
    # Optional; must match the molecule's own catalogued theory if given -
    # see app.services.jobs.start_job for why only one theory per molecule
    # is launchable through this endpoint right now.
    theory: str | None = None
    # Overrides the template config's epoch count (normally 200) - for a
    # quick demo run or a fast test, not a locked-protocol result.
    epochs: int | None = None


class TrainJobStatus(BaseModel):
    id: str
    molecule: str
    theory: str
    phase: str
    config_name: str
    started_at: str
    status: str  # "running" | "complete" | "failed"
    epochs_run: int | None = None
    max_epochs: int | None = None
    best_epoch: int | None = None
    best_val_loss: float | None = None
    last_val_loss: float | None = None
    stop_reason: str | None = None
    checkpoint_ready: bool
    log_tail: str


class DataUploadResponse(BaseModel):
    molecule: str
    theory: str
    gold_path: str
    n_train: int
    n_val: int
    n_test: int
    used_literature_split: bool
