"""Post-hoc conformal calibration for Phase 2/3's predicted uncertainty.

SCRUM-61 and the Phase 3 first-draft model card both found the same thing
two different ways: neither BLIP's loss-function fix nor Phase 3's
representation-noise mechanism produced calibrated uncertainty, because
nothing in *training* ever directly compares predicted variance against
real error. This module does not retrain anything. It fits one scalar per
quantity (energy, force) on the held-out **validation** split - never
training, never test - that rescales the model's own predicted std so it
matches real error on that held-out data, then checks on the untouched
**test** split whether that rescaling actually closes the gap.

Two papers, read 2026-10-23 (docs/papers/): Kuleshov et al. 2018 SS3.3-3.4
(recalibrating a Bayesian model's mean/variance output post hoc; the
studentized-residual feature |error|/std) and Bao et al. 2025 SS3 (Split
Conformal Regression's finite-sample marginal coverage guarantee). This
combines both - a studentized-residual nonconformity score with conformal
quantile selection - rather than Kuleshov's isotonic regression, because
it is the smallest change that reuses ml.utils.uq_metrics's existing
Normal-distribution machinery (`quantity_block`) completely unchanged:
multiply std by one learned scalar, nothing else about the pipeline moves.

Caveat stated plainly, from Kuleshov et al. 2018 SS5 (Discussion): this
cannot invent information. If a model is genuinely overconfident (BLIP),
the calibrated interval gets wider - that is the fix working, not a side
effect to tune away. Energy and force each get their own scale: they are
different units at different scales, and Phase 3's own numbers show the
miscalibration direction is not even the same for both (SS6,
docs/model_cards/graph_stochastic.md).

Run directly:

    python -m ml.training.calibrate --checkpoint experiments/checkpoints/phase2_blip_nll.pt
    python -m ml.training.calibrate --checkpoint experiments/checkpoints/phase3_graph_stochastic.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from ml.data.datasets import MD17Dataset
from ml.models.registry import load_model
from ml.utils.uq_metrics import quantity_block  # uq_metrics' own scoring, public since SCRUM-68

__all__ = ["conformal_scale", "calibrate_and_report"]

_VAR_EPS = 1e-12
DEFAULT_ALPHA = 0.1  # targets 90%-nominal coverage, matching COVERAGE_LEVELS' own 90% entry


def conformal_scale(error: np.ndarray, std: np.ndarray, alpha: float = DEFAULT_ALPHA) -> float:
    """Split-conformal scale factor (Bao et al. 2025 SS3, combined with
    Kuleshov et al. 2018 SS3.4's studentized-residual feature): the
    `(1-alpha)(1+1/n)`-th empirical quantile of `|error|/std` on a
    held-out calibration set. Multiplying every predicted std by this one
    number gives `error/(q*std)` the right spread to make the nominal
    `1-alpha` interval's empirical coverage match `1-alpha` on that
    calibration data - the guarantee is only as good as how representative
    the calibration set is of what gets scored next (exchangeability).

    Clamped to a quantile level of at most 1.0: for a very small
    calibration set, `(1-alpha)(1+1/n)` can exceed 1, which has no empirical
    quantile - clamping takes the maximum observed score instead of raising.
    """
    if error.shape != std.shape:
        raise ValueError(f"error and std must have the same shape, got {error.shape} vs {std.shape}")
    std = np.clip(std, np.sqrt(_VAR_EPS), None)
    scores = np.abs(error) / std
    n = scores.size
    if n < 1:
        raise ValueError("need at least one calibration point")
    level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n))
    return float(np.quantile(scores, level, method="higher"))


def _collect_errors_and_stds(model, dataset, batch_size: int = 32, seed: int = 0) -> dict[str, np.ndarray]:
    """Batched pass over `dataset` via model.predict_distribution - same
    loop shape as ml.training.evaluate.evaluate_checkpoint's stochastic
    branch, but collecting raw (signed) error / predicted-std arrays
    instead of feeding a UQAccumulator, since conformal_scale needs plain
    arrays, not an evaluate.py-shaped record."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    e_error, e_std, f_error, f_std = [], [], [], []
    torch.manual_seed(seed)
    with torch.no_grad():
        for batch in loader:
            e_mean, e_var, f_mean, f_var = model.predict_distribution(
                batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch,
            )
            e_error.append((e_mean - batch.y.view(-1)).detach().double().numpy())
            e_std.append(e_var.detach().double().clamp_min(_VAR_EPS).sqrt().numpy())
            f_error.append((f_mean - batch.force).detach().double().reshape(-1).numpy())
            f_std.append(f_var.detach().double().clamp_min(_VAR_EPS).sqrt().reshape(-1).numpy())
    return {
        "e_error": np.concatenate(e_error),
        "e_std": np.concatenate(e_std),
        "f_error": np.concatenate(f_error),
        "f_std": np.concatenate(f_std),
    }


def _stride_subset(dataset, size: int | None, seed: int):
    """Same convention as ml.training.train.stride_subset: `size` frames
    spread evenly over the whole split via a fixed stride and a seeded
    offset, not an unrepresentative head-of-file slice. `size=None` or
    `size >= len(dataset)` returns the dataset unchanged."""
    if size is None or size >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    stride = len(dataset) // size
    offset = int(rng.integers(stride))
    positions = offset + stride * np.arange(size)
    return torch.utils.data.Subset(dataset, positions.tolist())


def calibrate_and_report(
    checkpoint_path: str | Path,
    alpha: float = DEFAULT_ALPHA,
    val_subset: int | None = None,
    test_subset: int | None = 5000,
    subset_seed: int = 42,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Fits the conformal scale on the validation split, then reports
    before/after coverage/ECE/sharpness/Spearman on the test split.
    `val_subset`/`test_subset` stride-sample each split down to that many
    frames (same convention as ml.training.train.stride_subset) rather
    than using the whole thing - `test_subset` defaults to 5000 since
    this is meant to run in minutes per checkpoint, not the harness's own
    hours-long full-test-split scoring pass; `val_subset` defaults to
    None (the whole validation split) because the calibration quantile's
    own precision depends on calibration-set size, and the real report
    this module exists to produce should use all of it. Pass a small
    `val_subset` (a few hundred) for a fast correctness check instead,
    e.g. in a test.
    """
    model, checkpoint = load_model(checkpoint_path, map_location="cpu")
    if not getattr(model, "supports_distribution", False):
        raise ValueError(f"{checkpoint_path}: model has no predict_distribution - nothing to calibrate")

    data_cfg = checkpoint["data"]
    gold_path, cutoff, num_rbf = data_cfg["gold_path"], data_cfg["cutoff_radius"], data_cfg["num_rbf"]

    val_ds = _stride_subset(
        MD17Dataset(gold_path, split="val", cutoff_radius=cutoff, num_rbf=num_rbf), val_subset, subset_seed,
    )
    test_ds = _stride_subset(
        MD17Dataset(gold_path, split="test", cutoff_radius=cutoff, num_rbf=num_rbf), test_subset, subset_seed,
    )

    val = _collect_errors_and_stds(model, val_ds, batch_size=batch_size)
    q_energy = conformal_scale(val["e_error"], val["e_std"], alpha)
    q_force = conformal_scale(val["f_error"], val["f_std"], alpha)

    test = _collect_errors_and_stds(model, test_ds, batch_size=batch_size)

    return {
        "checkpoint": str(checkpoint_path),
        "alpha": alpha,
        "n_calibration": int(val["e_error"].size),
        "n_test": int(test["e_error"].size),
        "q_energy": q_energy,
        "q_force": q_force,
        "before": {
            "energy": quantity_block(test["e_error"], test["e_std"]),
            "force": quantity_block(test["f_error"], test["f_std"]),
        },
        "after": {
            "energy": quantity_block(test["e_error"], test["e_std"] * q_energy),
            "force": quantity_block(test["f_error"], test["f_std"] * q_force),
        },
    }


def _print_summary(report: dict[str, Any]) -> None:
    print(f"checkpoint: {report['checkpoint']}")
    print(f"calibration set: {report['n_calibration']} val frames; test subset: {report['n_test']} frames")
    print(f"q_energy = {report['q_energy']:.4f}   q_force = {report['q_force']:.4f}")
    print()
    print(f"{'':10} {'ECE before':>12} {'ECE after':>12} {'cov@90 before':>15} {'cov@90 after':>15} "
          f"{'sharp before':>13} {'sharp after':>13}")
    for quantity in ("energy", "force"):
        b, a = report["before"][quantity], report["after"][quantity]
        print(f"{quantity:10} {b['ece']:>12.4f} {a['ece']:>12.4f} "
              f"{b['coverage']['90%']:>15.4f} {a['coverage']['90%']:>15.4f} "
              f"{b['sharpness_mean_std']:>13.4f} {a['sharpness_mean_std']:>13.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--val-subset", type=int, default=None, help="default: the whole validation split")
    parser.add_argument("--test-subset", type=int, default=5000)
    parser.add_argument("--output", default=None, help="write the full report as JSON to this path")
    args = parser.parse_args()

    result = calibrate_and_report(
        args.checkpoint, alpha=args.alpha, val_subset=args.val_subset, test_subset=args.test_subset,
    )
    _print_summary(result)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nfull report written to {args.output}")
