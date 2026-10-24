"""Real inference over trained Phase 1-3 checkpoints, loaded once and cached.

Only ethanol (DFT) has trained checkpoints. `MD17_MOLECULES_CATALOG` is
read straight from the real gold files in data/gold/md17/ rather than
hand-maintained: the stub this replaced named two molecules
(naphthalene, salicylic_acid) with no gold file at all, and omitted two
that do have one (azobenzene, paracetamol).

Predicting on any molecule other than ethanol/DFT is refused with a clear
error rather than running an ethanol-trained model on it out-of-distribution
and reporting a plausible-looking number - this project's whole point this
sprint (CLAUDE.md, the BLIP calibration work) is not doing exactly that.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import HTTPException

from app.core.config import DEFAULT_PHASE, GOLD_DIR, PHASE_ALIASES, PHASE_CHECKPOINTS, TRAINED_MOLECULE
from app.services import calibration as calibration_module

from ml.data.datasets import MD17Dataset
from ml.models.registry import load_model

# Matches every model trained so far (ml/data/md17.py / configs): the
# dataset's own neighbour-graph cutoff and RBF width. Each model recomputes
# its own (wider) RBF internally from live positions for the energy pass
# (see e.g. ml/models/mpnn.py's predict_energy_and_forces) - these two
# numbers only decide edge_index's topology, not any model's own features.
CUTOFF_RADIUS = 5.0
DATASET_NUM_RBF = 16


def _scan_gold_catalog() -> dict[str, dict[str, Any]]:
    """{molecule: {num_atoms, theory, gold_path}}, straight from the real
    gold files. When a molecule has more than one theory (only ethanol
    does), the dft one wins - it is the theory every trained checkpoint
    here uses."""
    catalog: dict[str, dict[str, Any]] = {}
    for path in sorted(GOLD_DIR.glob("*.npz")):
        with np.load(path, allow_pickle=True) as data:
            molecule = str(data["molecule"])
            theory = str(data["theory"])
            num_atoms = int(data["z"].shape[0])
        existing = catalog.get(molecule)
        if existing is None or (existing["theory"] != "dft" and theory == "dft"):
            catalog[molecule] = {"num_atoms": num_atoms, "theory": theory, "gold_path": path}
    return catalog


MD17_MOLECULES_CATALOG: dict[str, dict[str, Any]] = _scan_gold_catalog()
# Contract shape (glossary.md SS4): {molecule: num_atoms}. Theory is tracked
# above for inference's own use; the documented /molecules response has no
# theory field.
MD17_MOLECULES: dict[str, int] = {m: info["num_atoms"] for m, info in MD17_MOLECULES_CATALOG.items()}


def refresh_catalog() -> dict[str, dict[str, Any]]:
    """Re-scan data/gold/md17/*.npz and update the catalog in place -
    called after app.services.data_pipeline writes a new gold file, so a
    freshly uploaded molecule shows up in GET /molecules without a server
    restart. Mutates the existing dicts (.clear()/.update()) rather than
    rebinding the module-level names: app.api.molecules and app.api.jobs
    both hold their own reference to these same dict objects from their
    own `from ... import MD17_MOLECULES` - rebinding the name here would
    leave their reference pointing at the stale dict."""
    fresh = _scan_gold_catalog()
    MD17_MOLECULES_CATALOG.clear()
    MD17_MOLECULES_CATALOG.update(fresh)
    MD17_MOLECULES.clear()
    MD17_MOLECULES.update({m: info["num_atoms"] for m, info in MD17_MOLECULES_CATALOG.items()})
    return MD17_MOLECULES_CATALOG


_checkpoint_cache: dict[str, tuple[float, Any]] = {}


def _load_checkpoint(path_str: str):
    """Loaded once per checkpoint path and reused - covers both the fixed
    PHASE_CHECKPOINTS entries (never change once trained) and a training
    job's own checkpoint (app.services.jobs), which train.py overwrites
    in place every time validation improves *while the job is still
    running*. A plain lru_cache would return the first epoch's model
    forever even as a job trains past it - predicting against an
    in-progress job is a real, useful thing to do (see /jobs, /predictions
    with phase="job:<id>"), so the cache is keyed on the checkpoint's own
    mtime: unchanged file, instant cache hit; changed file, one reload.

    Not cached on failure - a missing checkpoint raises every time until
    it exists, rather than caching the error forever for a server that
    keeps running after training finishes."""
    path = Path(path_str)
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"No checkpoint at {path}. Checkpoints are gitignored, not committed - train it "
                f"first, e.g. `python -m ml.training.train experiments/configs/{path.stem}.yaml`."
            ),
        )
    mtime = path.stat().st_mtime
    cached = _checkpoint_cache.get(path_str)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    model, _ = load_model(path)
    _checkpoint_cache[path_str] = (mtime, model)
    return model


def _load_cached(phase: str):
    return _load_checkpoint(str(PHASE_CHECKPOINTS[phase]))


@lru_cache(maxsize=None)
def _test_dataset(gold_path_str: str) -> MD17Dataset:
    return MD17Dataset(gold_path_str, split="test", cutoff_radius=CUTOFF_RADIUS, num_rbf=DATASET_NUM_RBF)


def warm_up() -> list[str]:
    """Load every registered phase's checkpoint now, at server startup,
    rather than on whichever request happens to hit it first - so a
    missing or broken checkpoint is visible in the startup log before a
    live demo, not mid-click. Returns one status line per phase."""
    lines = []
    for phase, path in PHASE_CHECKPOINTS.items():
        try:
            _load_cached(phase)
            lines.append(f"  {phase}: loaded ({path.name})")
        except HTTPException as exc:
            lines.append(f"  {phase}: NOT AVAILABLE - {exc.detail}")
    return lines


class InferenceService:
    def resolve_phase(self, phase: Optional[str]) -> str:
        if phase is None:
            return DEFAULT_PHASE
        resolved = PHASE_ALIASES.get(phase, phase)
        if resolved not in PHASE_CHECKPOINTS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unknown phase {phase!r}; choose one of {sorted(PHASE_CHECKPOINTS)} "
                    f"or the short aliases {sorted(PHASE_ALIASES)}"
                ),
            )
        return resolved

    def predict(self, molecule: str, frame_index: int, phase: Optional[str] = None) -> dict[str, Any]:
        info = MD17_MOLECULES_CATALOG.get(molecule)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Unknown molecule {molecule!r}; see GET /molecules")

        if phase is not None and phase.startswith("job:"):
            # A training job's own checkpoint (app.services.jobs), not one
            # of the fixed ethanol-only phases - lazy import to avoid a
            # circular top-level import (jobs.py imports this module to
            # read MD17_MOLECULES_CATALOG when launching a job).
            from app.services import jobs as jobs_module

            job = jobs_module.get_job(phase[len("job:"):])
            if job.molecule != molecule:
                raise HTTPException(
                    status_code=422,
                    detail=f"job {job.id} was trained on {job.molecule!r}, not {molecule!r}",
                )
            status = jobs_module.job_status(job)
            if not status["checkpoint_ready"]:
                raise HTTPException(
                    status_code=503,
                    detail=f"job {job.id} has no checkpoint yet (status: {status['status']})",
                )
            resolved_phase = phase
            checkpoint_path = job.checkpoint_path
        else:
            resolved_phase = self.resolve_phase(phase)
            if molecule != TRAINED_MOLECULE:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"{molecule!r} has real MD17 data but no trained checkpoint yet for "
                        f"{resolved_phase!r} - only {TRAINED_MOLECULE!r} (DFT) is trained across "
                        "Phase 1-3 so far. Predicting on it with an ethanol-trained model would be a "
                        "silently wrong out-of-distribution guess, not a real result - train it "
                        "first (POST /jobs/train)."
                    ),
                )
            checkpoint_path = PHASE_CHECKPOINTS[resolved_phase]

        dataset = _test_dataset(str(info["gold_path"]))
        if not (0 <= frame_index < len(dataset)):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"frame_index must be within [0, {len(dataset) - 1}] for the held-out test split "
                    f"({len(dataset)} frames); got {frame_index}"
                ),
            )

        data = dataset[frame_index]
        model = _load_checkpoint(str(checkpoint_path))

        if getattr(model, "supports_distribution", False):
            e_mean, e_var, f_mean, f_var = model.predict_distribution(
                data.x, data.pos, data.edge_index, data.edge_attr,
            )
            predicted_energy = float(e_mean.item())
            uncertainty = float(e_var.clamp_min(0).sqrt().item())
            force_rms = float(f_mean.pow(2).sum(dim=-1).mean().sqrt().item())

            # SCRUM-68: a post-hoc conformal scale, fit once offline
            # (ml/training/calibrate.py) on the held-out validation split -
            # None when this checkpoint was never calibrated (e.g. a job
            # trained live on another molecule). Additive: `uncertainty`
            # keeps meaning exactly what it always has (the model's raw,
            # uncalibrated std), `uncertainty_calibrated` is new.
            calibration = calibration_module.get_calibration(checkpoint_path)
            uncertainty_calibrated = uncertainty * calibration["q_energy"] if calibration else None
        else:
            energy, forces = model.predict_energy_and_forces(data.x, data.pos, data.edge_index, data.edge_attr, None)
            predicted_energy = float(energy.item())
            uncertainty = None
            uncertainty_calibrated = None
            force_rms = float(forces.pow(2).sum(dim=-1).mean().sqrt().item())

        return {
            "predicted_energy": predicted_energy,
            "predicted_force_rms": force_rms,
            "uncertainty": uncertainty,
            "uncertainty_calibrated": uncertainty_calibrated,
            "phase": resolved_phase,
            "true_energy": float(data.y.item()),
            "true_force_rms": float(data.force.pow(2).sum(dim=-1).mean().sqrt().item()),
        }


inference_service = InferenceService()
