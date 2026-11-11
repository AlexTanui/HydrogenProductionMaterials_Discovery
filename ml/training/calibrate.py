"""Post-hoc conformal calibration for Phase 2/3's predicted uncertainty.

SCRUM-61 and the Phase 3 first-draft model card both found the same thing
two different ways: neither BLIP's loss-function fix nor Phase 3's
representation-noise mechanism produced calibrated uncertainty, because
nothing in *training* ever directly compares predicted variance against
real error. This module does not retrain anything. It fits a correction
per quantity (energy, force) on the held-out **validation** split - never
training, never test - then checks on the untouched **test** split
whether that correction actually closes the gap. Two methods are
implemented, `--method scalar` (the original) and `--method cqr` (new):

**`scalar`** - Kuleshov et al. 2018 SS3.3-3.4 (recalibrating a Bayesian
model's mean/variance output post hoc; the studentized-residual feature
|error|/std) combined with Bao et al. 2025 SS3 (Split Conformal
Regression's finite-sample marginal coverage guarantee): one learned
scalar multiplies the model's predicted std, reusing
ml.utils.uq_metrics's existing Normal-distribution machinery
(`quantity_block`) completely unchanged.

**`cqr`** - Romano et al. 2019's Conformalized Quantile Regression, as
described in Bao et al. 2025 SS3.2, with one change: there is no separate
quantile-regression model to conformalize, because Phase 2/3 already
produce one for free. Every checkpoint scored here computes its mean/var
by running `num_mc_samples` (20) independent stochastic forward passes
and collapsing them - see `ml.models.blip.BLIP.predict_distribution` /
`ml.models.graph_stochastic.GraphStochastic.predict_distribution`. That
collapse is where the Normal assumption enters; the raw stack
(`predict_samples`, added alongside this) is already an empirical sample
from the model's own predictive distribution, whatever shape it has. CQR
takes the stack's own `alpha/2`/`1-alpha/2` empirical quantiles as a
prediction interval - no Gaussian assumption at all - then applies one
*additive* split-conformal correction `q_hat` (not a multiplicative
rescale) so the interval's marginal coverage is exactly `1-alpha` on the
calibration data, the same finite-sample guarantee `scalar` relies on.

Why this is worth having alongside `scalar`, not instead of it (see the
conversation this followed, 2026-11-04): `scalar` can only fix *width* -
it assumes the error really is Normal(0, std^2) and the only thing wrong
is the value of std. Phase 2/3's two mechanisms inject stochasticity very
differently (BLIP: weight noise; graph_stochastic: representation noise),
and there is no reason their residual distributions should both happen to
be symmetric Gaussian. `cqr` makes no such assumption - if the ensemble is
skewed or heavy-tailed, its empirical quantiles already reflect that,
and only the leftover marginal miscalibration gets the conformal
correction. The cost: with `num_mc_samples=20`, the raw `alpha/2`/
`1-alpha/2` quantiles sit near the 1st/19th order statistic per point -
noisy as a *per-point* estimate. That noise does not break the coverage
guarantee (it only depends on exchangeability between the calibration and
test points, not on any one point's quantile being accurate - Romano
et al. 2019 Thm 1), but it does mean `cqr`'s per-point intervals are a
noisier signal of *local* heteroscedasticity than a larger ensemble would
give; this is a smaller step than Bao's locally-adaptive CQR/CHR/LCP
variants, which still remain open as a follow-up.

Caveat stated plainly, from Kuleshov et al. 2018 SS5 (Discussion): neither
method can invent information. If a model is genuinely overconfident
(BLIP), the calibrated interval gets wider - that is the fix working, not
a side effect to tune away. Energy and force each get their own
correction: they are different units at different scales, and Phase 3's
own numbers show the miscalibration direction is not even the same for
both (SS6, docs/model_cards/graph_stochastic.md).

Run directly:

    python -m ml.training.calibrate --checkpoint experiments/checkpoints/phase2_blip_nll.pt
    python -m ml.training.calibrate --checkpoint experiments/checkpoints/phase3_graph_stochastic.pt --method cqr
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import jarque_bera, kurtosis, norm, skew
from torch_geometric.loader import DataLoader

from ml.data.datasets import MD17Dataset
from ml.models.registry import load_model
from ml.utils.uq_metrics import quantity_block  # uq_metrics' own scoring, public since SCRUM-68

__all__ = [
    "conformal_scale",
    "calibrate_and_report",
    "cqr_bounds",
    "cqr_nonconformity",
    "cqr_calibrate",
    "calibrate_cqr_and_report",
    "shape_diagnostics",
    "shape_diagnostics_by_atom_type",
]

_VAR_EPS = 1e-12
DEFAULT_ALPHA = 0.1  # targets 90%-nominal coverage, matching COVERAGE_LEVELS' own 90% entry

# Mirrors ml.data.md17's _PERIODIC_TABLE (inverted), kept as a local copy
# rather than importing that module's private name - this is only for
# labelling shape_diagnostics_by_atom_type's output by element symbol.
_Z_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 12: "Mg", 15: "P", 16: "S", 17: "Cl"}


def _split_conformal_level(n: int, alpha: float) -> float:
    """The `(1-alpha)(1+1/n)` quantile level Bao et al. 2025 SS3's split
    conformal construction calibrates at, clamped to at most 1.0: for a
    very small calibration set this can exceed 1, which has no empirical
    quantile - clamping takes the maximum observed score instead of
    raising. Shared by both `conformal_scale` and `cqr_calibrate` - same
    finite-sample guarantee, different nonconformity score feeding it.
    """
    if n < 1:
        raise ValueError("need at least one calibration point")
    return min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n))


def conformal_scale(error: np.ndarray, std: np.ndarray, alpha: float = DEFAULT_ALPHA) -> float:
    """Split-conformal scale factor (Bao et al. 2025 SS3, combined with
    Kuleshov et al. 2018 SS3.4's studentized-residual feature): the
    `(1-alpha)(1+1/n)`-th empirical quantile of `|error|/std` on a
    held-out calibration set. Multiplying every predicted std by this one
    number gives `error/(q*std)` the right spread to make the nominal
    `1-alpha` interval's empirical coverage match `1-alpha` on that
    calibration data - the guarantee is only as good as how representative
    the calibration set is of what gets scored next (exchangeability).
    """
    if error.shape != std.shape:
        raise ValueError(f"error and std must have the same shape, got {error.shape} vs {std.shape}")
    std = np.clip(std, np.sqrt(_VAR_EPS), None)
    scores = np.abs(error) / std
    level = _split_conformal_level(scores.size, alpha)
    return float(np.quantile(scores, level, method="higher"))


def cqr_bounds(samples: np.ndarray, alpha: float = DEFAULT_ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """Empirical `(lo, hi)` prediction-interval bounds straight from an MC
    ensemble, no distributional assumption: the `alpha/2` and `1-alpha/2`
    quantiles of `samples` (shape `[n_mc, n_points]`) along the MC axis.
    This is the "quantile regressor" Romano et al. 2019's CQR conformalizes
    - here it is the model's own stochastic forward passes rather than a
    fitted quantile-regression model, per Bao et al. 2025 SS3.2.
    """
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D [n_mc, n_points], got shape {samples.shape}")
    lo = np.quantile(samples, alpha / 2, axis=0)
    hi = np.quantile(samples, 1 - alpha / 2, axis=0)
    return lo, hi


def cqr_nonconformity(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Romano et al. 2019 eq. (2): how far outside `[lo, hi]` the true
    value falls, signed so a comfortably-covered point scores negative
    rather than zero - the split-conformal quantile below needs that
    margin, not just an inside/outside flag.
    """
    return np.maximum(lo - y, y - hi)


def cqr_calibrate(samples: np.ndarray, y: np.ndarray, alpha: float = DEFAULT_ALPHA) -> float:
    """Conformalized Quantile Regression (Romano et al. 2019), scored on a
    held-out calibration set: the empirical `(lo, hi)` bounds from
    `cqr_bounds`, corrected by one additive `q_hat` - the
    `(1-alpha)(1+1/n)`-th empirical quantile of the nonconformity scores -
    so that `[lo - q_hat, hi + q_hat]` achieves exactly `1-alpha` marginal
    coverage on this calibration data. Unlike `conformal_scale`'s
    multiplicative std rescale, `q_hat` is additive and the base interval
    `[lo, hi]` need not be symmetric, so this can correct skew the scalar
    method cannot touch.
    """
    lo, hi = cqr_bounds(samples, alpha)
    scores = cqr_nonconformity(lo, hi, y)
    level = _split_conformal_level(scores.size, alpha)
    return float(np.quantile(scores, level, method="higher"))


def cqr_block(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, q_hat: float = 0.0) -> dict[str, float]:
    """CQR-equivalent of ml.utils.uq_metrics's `quantity_block`, scored
    directly off empirical interval bounds instead of a Normal(mean, var)
    z-score. Reports coverage and mean width at the one alpha level this
    interval was built for - not a full ECE curve, since a single
    conformal pass is calibrated at exactly one nominal level, not every
    level at once (Bao et al. 2025's own evaluation reports coverage and
    width/efficiency at a target level for the same reason).
    """
    lo_c, hi_c = lo - q_hat, hi + q_hat
    covered = (y >= lo_c) & (y <= hi_c)
    return {
        "coverage_90": float(covered.mean()),
        "mean_width": float((hi_c - lo_c).mean()),
    }


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


def shape_diagnostics(samples: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Direct measurement of how non-Gaussian the predictive distribution
    actually is - the thing the CQR-vs-Gaussian width gap
    (docs/reviews/cqr-conformal-calibration-2026-11-04.md) could only
    infer indirectly, by noticing CQR needed a wider interval than a
    Normal fit to the same variance. Two complementary views, both pooled
    across every point so there is enough data for third/fourth-moment
    estimates despite only `num_mc_samples` draws at any one point:

    - `ensemble_skew`/`ensemble_excess_kurtosis`: each point's own MC
      draws, standardized by THAT point's own ensemble mean/std, pooled
      across all points. Dividing out each point's own scale before
      pooling means this cannot be an artifact of pooling points with
      different predicted scales (a scale-mixture of Normals already has
      positive excess kurtosis even when every component is exactly
      Normal) - a nonzero value here is the predictive distribution's own
      shape, not a pooling effect.
    - `residual_skew`/`residual_excess_kurtosis`: the studentized residual
      `(y - ensemble_mean) / ensemble_std`, pooled across points - the
      shape `conformal_scale`/Kuleshov et al. 2018's calibration
      diagnostic implicitly assumes is symmetric Normal. This one CAN be
      inflated by cross-point heteroscedasticity, so a nonzero value here
      does not by itself prove the ensemble's own shape is non-Gaussian -
      `ensemble_*` is the one that isolates that.

    Both are (skewness, excess kurtosis), and **neither excess-kurtosis
    figure is centered at 0 even for a correctly-calibrated Gaussian
    model** at `num_mc_samples=20` - both skews are, but dividing by a
    std estimated from only 20 draws biases kurtosis on its own, for two
    different reasons (verified by simulation; each stable across seeds
    to +-0.01 at n_points=200,000):

    - `ensemble_excess_kurtosis` pools to ~**-0.28**, not 0: standardizing
      a point's 20 draws by their OWN sample std is self-referential (the
      denominator was computed from the same 20 numbers it's dividing),
      which shrinks the pooled tails relative to true population-std
      standardization.
    - `residual_excess_kurtosis` pools to ~**+0.4**, not 0: `error / std`
      with `std` from only 20 draws behaves like a Student-t(19) variable
      even when the model is exactly correctly calibrated - the classic
      reason small-sample studentization has heavier tails than Normal
      (Student-t(df) excess kurtosis is `6/(df-4)`; `df = num_mc_samples
      - 1 = 19` gives 6/15 = 0.4, matching the simulation).

    Compare each figure against its own baseline above, not against 0: a
    value *above* `ensemble_excess_kurtosis`'s -0.28 baseline still means
    heavier-than-Gaussian tails in the predictive distribution itself,
    just by less than the raw number suggests on its own; `residual_*`
    similarly needs compared against +0.4, not 0, before concluding the
    real error is heavier-tailed than the model's own uncertainty
    predicts. Both baselines shrink toward 0 as `num_mc_samples` grows.

    `ensemble_jarque_bera_p` is a formal normality test on the pooled,
    standardized ensemble, included for completeness - its null
    hypothesis is standard Normal, which the -0.28 baseline above means
    is already slightly wrong even for a correctly-calibrated model, and
    at the sample sizes here (`num_mc_samples * n_points`, often in the
    hundreds of thousands) it rejects even practically tiny deviations
    regardless. Read the moment magnitudes against their baselines, not
    this p-value, as the actual evidence.
    """
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D [n_mc, n_points], got shape {samples.shape}")
    if samples.shape[1] != y.shape[0]:
        raise ValueError(f"samples has {samples.shape[1]} points but y has {y.shape[0]}")

    mean = samples.mean(axis=0, keepdims=True)
    std = np.clip(samples.std(axis=0, ddof=1, keepdims=True), np.sqrt(_VAR_EPS), None)
    standardized_ensemble = ((samples - mean) / std).reshape(-1)

    studentized_residual = (mean.reshape(-1) - y) / std.reshape(-1)

    jb_stat, jb_p = jarque_bera(standardized_ensemble)

    return {
        "ensemble_skew": float(skew(standardized_ensemble)),
        "ensemble_excess_kurtosis": float(kurtosis(standardized_ensemble, fisher=True)),
        "ensemble_jarque_bera_stat": float(jb_stat),
        "ensemble_jarque_bera_p": float(jb_p),
        "residual_skew": float(skew(studentized_residual)),
        "residual_excess_kurtosis": float(kurtosis(studentized_residual, fisher=True)),
    }


def shape_diagnostics_by_atom_type(samples: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict[str, dict[str, float]]:
    """`shape_diagnostics`, split by atom type - the direct test of the
    atom-type-heterogeneity hypothesis raised to explain why force's
    pooled `residual_excess_kurtosis` is so much larger than energy's
    (docs/reviews/cqr-conformal-calibration-2026-11-04.md).

    Note what this does and doesn't isolate: `shape_diagnostics` already
    studentizes every POINT by its own ensemble std, so a Gaussian
    residual of the "wrong" width (one atom type systematically
    over/underconfident relative to another, e.g. stiff O-H/C-O bonds vs.
    floppy methyl H) is still a Gaussian - its own excess kurtosis stays
    near the usual +0.4 null baseline. What inflates the POOLED figure is
    the textbook Gaussian scale-mixture effect: mixing two differently-
    scaled Gaussians together has elevated kurtosis only once pooled, not
    within either component. So if cross-atom-type heterogeneity (not
    genuine non-Gaussian shape within any one type) is the real cause of
    force's large pooled `residual_excess_kurtosis`, splitting by atom
    type should make most of that inflation disappear: each per-element
    slice below should land back near the single-distribution
    `shape_diagnostics` +0.4/-0.28 null baselines, even though the pooled
    figure that motivated this function is well above +0.4. `z` must
    carry one atomic number per pooled point, same convention as
    `samples`/`y` (for force, 3 entries per atom - x/y/z - since
    `_collect_samples` flattens N*3 together).
    """
    if z.shape != y.shape:
        raise ValueError(f"z must align with y pointwise, got {z.shape} vs {y.shape}")
    out: dict[str, dict[str, float]] = {}
    for atomic_number in sorted(set(z.tolist())):
        mask = z == atomic_number
        symbol = _Z_TO_SYMBOL.get(int(atomic_number), f"Z{int(atomic_number)}")
        diag = shape_diagnostics(samples[:, mask], y[mask])
        diag["n_points"] = int(mask.sum())
        out[symbol] = diag
    return out


def _collect_samples(model, dataset, batch_size: int = 32, seed: int = 0) -> dict[str, np.ndarray]:
    """Batched pass over `dataset` via model.predict_samples - same loop
    shape as `_collect_errors_and_stds`, but keeping the raw MC stack
    instead of collapsing it to mean/std, since `cqr_bounds`/
    `cqr_calibrate` need the actual ensemble. Concatenated along the
    per-point axis (dim 1) across batches, since `num_mc_samples` (dim 0)
    is fixed by the checkpoint and identical in every batch. `f_z` carries
    one atomic number per force point, in the same N*3 (atom, then x/y/z)
    flattening `f_samples` uses, via `np.repeat` - `MD17Dataset.get`
    (ml/data/datasets.py) stores the same fixed, unreordered `z` vector in
    `batch.x` on every frame, so one dataset's atom ordering never shifts
    across frames (`shape_diagnostics_by_atom_type`'s only precondition).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    e_samples, e_true, f_samples, f_true, f_z = [], [], [], [], []
    torch.manual_seed(seed)
    with torch.no_grad():
        for batch in loader:
            e_stack, f_stack = model.predict_samples(
                batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch,
            )
            e_samples.append(e_stack.detach().double().numpy())  # [n_mc, B]
            e_true.append(batch.y.view(-1).detach().double().numpy())
            f_samples.append(f_stack.detach().double().reshape(f_stack.shape[0], -1).numpy())  # [n_mc, N*3]
            f_true.append(batch.force.detach().double().reshape(-1).numpy())
            f_z.append(np.repeat(batch.x.view(-1).detach().numpy(), 3))  # [N*3], 3 per atom
    return {
        "e_samples": np.concatenate(e_samples, axis=1),
        "e_true": np.concatenate(e_true),
        "f_samples": np.concatenate(f_samples, axis=1),
        "f_true": np.concatenate(f_true),
        "f_z": np.concatenate(f_z),
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


def _load_model_and_splits(
    checkpoint_path: str | Path, val_subset: int | None, test_subset: int | None, subset_seed: int,
):
    """Shared by `calibrate_and_report` and `calibrate_cqr_and_report`:
    loads the checkpoint, checks it actually supports a predictive
    distribution, and stride-samples its val/test gold splits down to
    `val_subset`/`test_subset` frames (same convention as
    ml.training.train.stride_subset)."""
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
    return model, val_ds, test_ds


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
    frames rather than using the whole thing - `test_subset` defaults to
    5000 since this is meant to run in minutes per checkpoint, not the
    harness's own hours-long full-test-split scoring pass; `val_subset`
    defaults to None (the whole validation split) because the calibration
    quantile's own precision depends on calibration-set size, and the
    real report this module exists to produce should use all of it. Pass
    a small `val_subset` (a few hundred) for a fast correctness check
    instead, e.g. in a test.
    """
    model, val_ds, test_ds = _load_model_and_splits(checkpoint_path, val_subset, test_subset, subset_seed)

    val = _collect_errors_and_stds(model, val_ds, batch_size=batch_size)
    q_energy = conformal_scale(val["e_error"], val["e_std"], alpha)
    q_force = conformal_scale(val["f_error"], val["f_std"], alpha)

    # q_* is already the multiplier that makes std*q cover 1-alpha of errors
    # directly (conformal_scale's own empirical quantile) - std*q IS the
    # 1-alpha interval's half-width. But _quantity_block (via its coverage
    # table / ECE) treats whatever std it's given as a genuine Gaussian std,
    # scoring the 90% interval as +-norm.ppf(0.95)*std =~ +-1.645*std. Passing
    # std*q straight into _quantity_block therefore double-applies that
    # 1.645 factor, overshooting every "after" coverage/ECE number by
    # norm.ppf(1-alpha/2) (SCRUM-68, Fazin's calibration review). Dividing
    # q by that same z-value converts it into a real std multiplier first,
    # so _quantity_block's own 1.645 reproduces exactly std*q, not std*q*1.645.
    z = norm.ppf(1 - alpha / 2)
    std_scale_energy = q_energy / z
    std_scale_force = q_force / z

    test = _collect_errors_and_stds(model, test_ds, batch_size=batch_size)

    return {
        "checkpoint": str(checkpoint_path),
        "alpha": alpha,
        "n_calibration": int(val["e_error"].size),
        "n_test": int(test["e_error"].size),
        "q_energy": q_energy,
        "q_force": q_force,
        # The multiplier callers (inference.py, benchmarks.py) should actually
        # apply to a raw std - q_* above is only the empirical fit statistic.
        "std_scale_energy": std_scale_energy,
        "std_scale_force": std_scale_force,
        "before": {
            "energy": quantity_block(test["e_error"], test["e_std"]),
            "force": quantity_block(test["f_error"], test["f_std"]),
        },
        "after": {
            "energy": quantity_block(test["e_error"], test["e_std"] * std_scale_energy),
            "force": quantity_block(test["f_error"], test["f_std"] * std_scale_force),
        },
    }


def calibrate_cqr_and_report(
    checkpoint_path: str | Path,
    alpha: float = DEFAULT_ALPHA,
    val_subset: int | None = None,
    test_subset: int | None = 5000,
    subset_seed: int = 42,
    batch_size: int = 32,
) -> dict[str, Any]:
    """CQR counterpart to `calibrate_and_report`: fits the additive
    `q_hat` correction (`cqr_calibrate`) on the validation split's raw MC
    ensembles, then reports before/after coverage and mean interval width
    on the test split - no Normal-distribution assumption anywhere in
    this path. Same subset/seed conventions as `calibrate_and_report`.
    """
    model, val_ds, test_ds = _load_model_and_splits(checkpoint_path, val_subset, test_subset, subset_seed)

    val = _collect_samples(model, val_ds, batch_size=batch_size)
    q_hat_energy = cqr_calibrate(val["e_samples"], val["e_true"], alpha)
    q_hat_force = cqr_calibrate(val["f_samples"], val["f_true"], alpha)

    test = _collect_samples(model, test_ds, batch_size=batch_size)
    lo_e, hi_e = cqr_bounds(test["e_samples"], alpha)
    lo_f, hi_f = cqr_bounds(test["f_samples"], alpha)

    return {
        "checkpoint": str(checkpoint_path),
        "alpha": alpha,
        "n_calibration": int(val["e_true"].size),
        "n_test": int(test["e_true"].size),
        "q_hat_energy": q_hat_energy,
        "q_hat_force": q_hat_force,
        "before": {
            "energy": cqr_block(lo_e, hi_e, test["e_true"]),
            "force": cqr_block(lo_f, hi_f, test["f_true"]),
        },
        "after": {
            "energy": cqr_block(lo_e, hi_e, test["e_true"], q_hat_energy),
            "force": cqr_block(lo_f, hi_f, test["f_true"], q_hat_force),
        },
        # Property of the raw test-split ensemble, not of either
        # calibration step above - see shape_diagnostics' own docstring.
        "shape": {
            "energy": shape_diagnostics(test["e_samples"], test["e_true"]),
            "force": shape_diagnostics(test["f_samples"], test["f_true"]),
        },
        # Energy has no per-atom breakdown (one value per frame, not per
        # atom) - only force is stratified, by shape_diagnostics_by_atom_type.
        "shape_by_atom_type": {
            "force": shape_diagnostics_by_atom_type(test["f_samples"], test["f_true"], test["f_z"]),
        },
    }


def _print_summary(report: dict[str, Any]) -> None:
    print(f"checkpoint: {report['checkpoint']}")
    print(f"calibration set: {report['n_calibration']} val frames; test subset: {report['n_test']} frames")
    print(f"q_energy = {report['q_energy']:.4f}   q_force = {report['q_force']:.4f}")
    print(f"std_scale_energy = {report['std_scale_energy']:.4f}   "
          f"std_scale_force = {report['std_scale_force']:.4f}  (apply this to a raw std, not q_*)")
    print()
    print(f"{'':10} {'ECE before':>12} {'ECE after':>12} {'cov@90 before':>15} {'cov@90 after':>15} "
          f"{'sharp before':>13} {'sharp after':>13}")
    for quantity in ("energy", "force"):
        b, a = report["before"][quantity], report["after"][quantity]
        print(f"{quantity:10} {b['ece']:>12.4f} {a['ece']:>12.4f} "
              f"{b['coverage']['90%']:>15.4f} {a['coverage']['90%']:>15.4f} "
              f"{b['sharpness_mean_std']:>13.4f} {a['sharpness_mean_std']:>13.4f}")


def _print_cqr_summary(report: dict[str, Any]) -> None:
    print(f"checkpoint: {report['checkpoint']}  (method: cqr)")
    print(f"calibration set: {report['n_calibration']} val frames; test subset: {report['n_test']} frames")
    print(f"q_hat_energy = {report['q_hat_energy']:.4f}   q_hat_force = {report['q_hat_force']:.4f}  "
          f"(additive, kcal/mol and kcal/mol/A respectively)")
    print()
    print(f"{'':10} {'cov@90 before':>15} {'cov@90 after':>15} {'width before':>14} {'width after':>14}")
    for quantity in ("energy", "force"):
        b, a = report["before"][quantity], report["after"][quantity]
        print(f"{quantity:10} {b['coverage_90']:>15.4f} {a['coverage_90']:>15.4f} "
              f"{b['mean_width']:>14.4f} {a['mean_width']:>14.4f}")
    if "shape" in report:
        print()
        print(f"{'':10} {'ens. skew':>10} {'ens. exkurt':>12} {'JB p':>10} {'resid. skew':>12} {'resid. exkurt':>14}")
        for quantity in ("energy", "force"):
            s = report["shape"][quantity]
            print(f"{quantity:10} {s['ensemble_skew']:>10.4f} {s['ensemble_excess_kurtosis']:>12.4f} "
                  f"{s['ensemble_jarque_bera_p']:>10.2e} {s['residual_skew']:>12.4f} "
                  f"{s['residual_excess_kurtosis']:>14.4f}")
    if "shape_by_atom_type" in report:
        print()
        print("force, by atom type:")
        print(f"{'':6} {'n_points':>10} {'ens. skew':>10} {'ens. exkurt':>12} {'resid. skew':>12} {'resid. exkurt':>14}")
        for symbol, s in report["shape_by_atom_type"]["force"].items():
            print(f"{symbol:6} {s['n_points']:>10} {s['ensemble_skew']:>10.4f} {s['ensemble_excess_kurtosis']:>12.4f} "
                  f"{s['residual_skew']:>12.4f} {s['residual_excess_kurtosis']:>14.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", choices=["scalar", "cqr"], default="scalar",
                         help="scalar: Gaussian std rescale (default). cqr: distribution-free "
                              "conformalized quantile regression on the raw MC ensemble.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--val-subset", type=int, default=None, help="default: the whole validation split")
    parser.add_argument("--test-subset", type=int, default=5000)
    parser.add_argument("--output", default=None, help="write the full report as JSON to this path")
    args = parser.parse_args()

    if args.method == "cqr":
        result = calibrate_cqr_and_report(
            args.checkpoint, alpha=args.alpha, val_subset=args.val_subset, test_subset=args.test_subset,
        )
        _print_cqr_summary(result)
    else:
        result = calibrate_and_report(
            args.checkpoint, alpha=args.alpha, val_subset=args.val_subset, test_subset=args.test_subset,
        )
        _print_summary(result)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nfull report written to {args.output}")
