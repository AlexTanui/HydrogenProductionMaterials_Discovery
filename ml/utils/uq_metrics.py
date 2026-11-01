"""Uncertainty-quality metrics for stochastic models (Phase 2 BLIP, Phase 3).

Companion to `ml/utils/metrics.py`: that module streams running sums and
is accuracy-only (energy/force MAE/RMSE). Every metric here - calibration,
coverage, Spearman correlation, a risk-coverage curve - is rank- or
quantile-based and needs the *whole* test split's errors and uncertainties
at once to sort or bin them; there is no streaming formulation. So, unlike
`MetricAccumulator`, `UQAccumulator` holds per-frame (energy) and
per-component (force) arrays in memory rather than running sums. For
MD17-scale test splits (tens of thousands of frames, under 25 atoms each)
that is a few tens of MB - acceptable; it would not be for a dataset with
millions of frames, and this module does not try to be.

**These definitions are ratified as UQ protocol v1.0** (Fazin, from
2026-10-23). Dongxiao owned the definitions - binning strategy,
correlation convention, and what counts as calibrated - and assigned them
to Fazin on 2026-10-23, who adopted the choices already implemented here
as v1.0 rather than changing them. So the numbers any earlier run
recorded stand as scored: the definition did not move when ownership did,
so a re-score is expected to reproduce them (none has been run yet; see
SCRUM-38). A record written before that date
still carries a note calling the definition provisional - that note is
historical, not a live caveat. See SCRUM-39.

Changing any choice below is a v2.0 and invalidates cross-run
comparability, so it needs a re-score of every stochastic run in the same
commit, not a quiet edit.

**Distributional assumption - the central decision that shapes every
metric below.** `ModelOutput` (ml/utils/contract.py) carries a predictive
*mean and variance* (`E_var`/`F_var`), not raw MC samples - that is the
contract `BLIP.predict_distribution` already returns, and extending it to
carry every one of `num_mc_samples` raw draws would be a materially bigger
change (every batch's memory cost multiplies by that count, and the
contract's "one batch of predictions" shape would need a new axis).
Every metric here that needs more than a point estimate therefore assumes
the predictive distribution is **Normal(mean, variance)**, per frame for
energy and per component for force. That is what makes closed-form NLL
and CRPS possible without raw samples. Moving to the empirical MC-sample
distribution instead (no Gaussian assumption) is the bigger change
described above and a v2.0, not an edit to this file.

Six metrics, each reported separately for energy and force:

1. **Coverage** - empirical fraction of ground truth within the nominal
   50/80/90/95% central interval (`COVERAGE_LEVELS`), assuming Normal.
2. **ECE** - mean absolute gap between nominal and empirical coverage
   across those same levels (Kuleshov et al. 2018's regression
   calibration error). Reported alongside coverage, not in place of it.
3. **Spearman uncertainty-error correlation** - rank correlation between
   predictive std and |error|. `None` when every predicted std is equal
   (nothing to rank).
4. **Risk-coverage curve** - sort by predictive std ascending (most
   confident first), report MAE of the retained fraction at coverage
   levels 10%..100%.
5. **Gaussian NLL and CRPS** - closed-form scores of the full predictive
   distribution against the Normal assumption above, not just its mean.
6. **Sharpness** - mean predictive std. Meaningless alone (a model can be
   sharp and wrong); read beside coverage/ECE, never instead of them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

from ml.utils.contract import DatasetKey, ModelOutput, ReferenceData, Units, require_compatible

__all__ = ["UQAccumulator", "COVERAGE_LEVELS", "RISK_COVERAGE_LEVELS"]

SCHEMA_VERSION = 1

# Nominal central-interval levels for coverage/ECE. 50/80/90/95% matches the
# paste's own ask and is the conventional spread for a regression reliability
# check - not so coarse (just 95%) that miscalibration elsewhere is invisible,
# not so fine that each level's empirical estimate is noise-dominated at
# realistic test-set sizes.
COVERAGE_LEVELS: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)

# Risk-coverage retained-fraction grid, most-confident-first.
RISK_COVERAGE_LEVELS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# Variance floor. A fresh StochasticLinear's noise_gate can in principle
# output exactly 0 (e.g. at initialisation-like weights); dividing by zero
# there must not crash a whole evaluation run over one frame's bad std.
_VAR_EPS = 1e-12


def _coverage_table(z: np.ndarray, levels: tuple[float, ...] = COVERAGE_LEVELS) -> dict[str, float]:
    """Empirical fraction of |z| within each nominal level's half-width, for z ~ N(0, 1) if calibrated."""
    table = {}
    for p in levels:
        half_width = float(stats.norm.ppf(0.5 + p / 2.0))
        table[f"{p:.0%}"] = float(np.mean(np.abs(z) <= half_width))
    return table


def _ece(coverage: dict[str, float], levels: tuple[float, ...] = COVERAGE_LEVELS) -> float:
    gaps = [abs(coverage[f"{p:.0%}"] - p) for p in levels]
    return float(np.mean(gaps))


def _spearman(std: np.ndarray, abs_error: np.ndarray) -> Optional[float]:
    """None, not 0.0, when std or |error| has no spread - "uncorrelated" and
    "undefined" are different claims, and a constant array makes scipy warn
    and return nan rather than something to round to 0."""
    if std.size < 2 or float(np.ptp(std)) == 0.0 or float(np.ptp(abs_error)) == 0.0:
        return None
    rho, _ = stats.spearmanr(std, abs_error)
    return None if np.isnan(rho) else float(rho)


def _risk_coverage(std: np.ndarray, abs_error: np.ndarray,
                    levels: tuple[float, ...] = RISK_COVERAGE_LEVELS) -> list[dict]:
    order = np.argsort(std, kind="stable")  # most confident (lowest predicted std) first
    sorted_abs_error = abs_error[order]
    n = sorted_abs_error.size
    curve = []
    for coverage in levels:
        k = max(1, int(round(coverage * n)))
        curve.append({"coverage": float(coverage), "n": int(k), "mae": float(np.mean(sorted_abs_error[:k]))})
    return curve


def _gaussian_nll(error: np.ndarray, std: np.ndarray) -> float:
    """Mean -log N(error; 0, std^2) = 0.5*log(2*pi*std^2) + error^2 / (2*std^2)."""
    variance = std * std
    return float(np.mean(0.5 * np.log(2.0 * np.pi * variance) + (error * error) / (2.0 * variance)))


def _gaussian_crps(error: np.ndarray, std: np.ndarray) -> float:
    """Closed-form CRPS of N(0, std^2) against `error` (Gneiting & Raftery 2007, eq. 20)."""
    z = error / std
    phi = stats.norm.pdf(z)
    cdf = stats.norm.cdf(z)
    crps = std * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    return float(np.mean(crps))


def _quantity_block(error: np.ndarray, std: np.ndarray) -> dict:
    z = error / std
    abs_error = np.abs(error)
    coverage = _coverage_table(z)
    return {
        "n": int(error.size),
        "coverage": coverage,
        "ece": _ece(coverage),
        "spearman_uncertainty_error": _spearman(std, abs_error),
        "risk_coverage": _risk_coverage(std, abs_error),
        "nll_gaussian": _gaussian_nll(error, std),
        "crps_gaussian": _gaussian_crps(error, std),
        "sharpness_mean_std": float(np.mean(std)),
    }


@dataclass
class _UQState:
    key: DatasetKey
    units: Units
    e_error: list = field(default_factory=list)
    e_std: list = field(default_factory=list)
    # Force: one entry per (frame, atom, component) - same flattened
    # "component" convention as ml/utils/metrics.py's force_mae_component,
    # so this reads the same number of force values that metric does.
    f_error: list = field(default_factory=list)
    f_std: list = field(default_factory=list)


class UQAccumulator:
    """Collects per-frame uncertainty + error from every stochastic batch seen.

        acc = UQAccumulator()
        for pred, ref in eval_loader:
            if pred.is_stochastic:
                acc.update(pred, ref)
        result = acc.compute()   # None if update() was never called

    Batches for different (molecule, theory) keys accumulate and report
    separately, same as `MetricAccumulator`. A batch with no variance at
    all (`pred.is_stochastic` False) is the caller's to skip - `update`
    raises rather than silently no-op, so a forgotten `if` is loud.
    """

    def __init__(self) -> None:
        self._states: dict[DatasetKey, _UQState] = {}

    def update(self, pred: ModelOutput, ref: ReferenceData) -> None:
        if not pred.is_stochastic:
            raise ValueError(
                "UQAccumulator.update() called with a deterministic ModelOutput "
                "(E_var and F_var both None) - check pred.is_stochastic before calling"
            )
        require_compatible(pred, ref)

        state = self._states.get(ref.key)
        if state is None:
            state = _UQState(key=ref.key, units=ref.units)
            self._states[ref.key] = state
        elif state.units != ref.units:
            raise ValueError(
                f"{ref.key} was accumulated in {state.units.as_dict()} and this batch is in "
                f"{ref.units.as_dict()}; call .to_project_units() on every batch first"
            )

        if pred.E_var is not None:
            e_err = (pred.E - ref.E).detach().double()
            e_std = pred.E_var.detach().double().clamp_min(_VAR_EPS).sqrt()
            state.e_error.extend(e_err.tolist())
            state.e_std.extend(e_std.tolist())
        if pred.F_var is not None:
            f_err = (pred.F - ref.F).detach().double().reshape(-1)
            f_std = pred.F_var.detach().double().clamp_min(_VAR_EPS).sqrt().reshape(-1)
            state.f_error.extend(f_err.tolist())
            state.f_std.extend(f_std.tolist())

    def __len__(self) -> int:
        return len(self._states)

    def compute(self) -> Optional[dict]:
        """None if no stochastic batch was ever seen - never a fabricated empty block."""
        if not self._states:
            return None
        ordered = sorted(self._states.values(), key=lambda s: s.key)
        return {
            "schema_version": SCHEMA_VERSION,
            "distribution_assumption": (
                "Normal(mean, variance) per frame (energy) / per component (force), "
                "from ModelOutput.E_var/F_var - UQ protocol v1.0, see this module's "
                "docstring (SCRUM-39)"
            ),
            "coverage_levels": list(COVERAGE_LEVELS),
            "risk_coverage_levels": list(RISK_COVERAGE_LEVELS),
            "per_key": [self._record(s) for s in ordered],
        }

    @staticmethod
    def _record(state: _UQState) -> dict:
        record: dict = {**state.key.as_dict(), "units": state.units.as_dict()}
        for quantity, errors, stds in (
            ("energy", state.e_error, state.e_std),
            ("force", state.f_error, state.f_std),
        ):
            if not errors:
                record[quantity] = None
                continue
            record[quantity] = _quantity_block(np.asarray(errors, dtype=np.float64),
                                                np.asarray(stds, dtype=np.float64))
        return record
