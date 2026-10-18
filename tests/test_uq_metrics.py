"""Tests for ml/utils/uq_metrics.py.

Where a value can be hand-computed exactly (coverage, ECE, risk-coverage,
Gaussian NLL/CRPS at z=0), it is, with the derivation in a comment -
same discipline as tests/test_metrics.py. Where a property only holds in
aggregate (calibration over many samples), the test draws a large,
seeded synthetic sample and checks the statistic lands in a documented
tolerance band - that is inherent to what's being tested, not a shortcut.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ml.utils.contract import ModelOutput, ReferenceData, Units
from ml.utils.uq_metrics import UQAccumulator


def build(energy_errors, energy_std, force_errors=None, force_std=None,
          n_atoms=1, molecule="toy", theory="dft", units=None):
    """Reference of zeros plus a prediction equal to the intended error, with
    explicit per-frame (energy) / per-component (force) predictive std."""
    units = units or Units()
    n_frames = len(energy_errors)
    if force_errors is None:
        force_errors = [[0.0, 0.0, 0.0]] * (n_frames * n_atoms)
        force_std = [[1.0, 1.0, 1.0]] * (n_frames * n_atoms)

    ref = ReferenceData(
        E=torch.zeros(n_frames, dtype=torch.float64),
        F=torch.zeros(n_frames * n_atoms, 3, dtype=torch.float64),
        batch=torch.repeat_interleave(torch.arange(n_frames), n_atoms),
        molecule=molecule,
        theory=theory,
        units=units,
    )
    pred = ModelOutput(
        E=torch.tensor(energy_errors, dtype=torch.float64),
        F=torch.tensor(force_errors, dtype=torch.float64),
        E_var=torch.tensor(energy_std, dtype=torch.float64) ** 2,
        F_var=torch.tensor(force_std, dtype=torch.float64) ** 2,
        units=units,
    )
    return pred, ref


def energy_block(*batches):
    acc = UQAccumulator()
    for pred, ref in batches:
        acc.update(pred, ref)
    return acc.compute()["per_key"][0]["energy"]


# --------------------------------------------------------------------------
# 1. Accumulator plumbing
# --------------------------------------------------------------------------

def test_compute_is_none_without_any_update():
    assert UQAccumulator().compute() is None


def test_update_rejects_a_deterministic_output():
    pred, ref = build([1.0], [1.0])
    pred = ModelOutput(E=pred.E, F=pred.F, units=pred.units)  # strip E_var/F_var
    assert not pred.is_stochastic
    with pytest.raises(ValueError, match="deterministic"):
        UQAccumulator().update(pred, ref)


def test_per_key_separates_different_molecules():
    acc = UQAccumulator()
    acc.update(*build([1.0], [1.0], molecule="ethanol"))
    acc.update(*build([1.0], [1.0], molecule="aspirin"))
    result = acc.compute()
    assert {r["molecule"] for r in result["per_key"]} == {"ethanol", "aspirin"}


def test_quantity_is_none_when_that_half_of_the_batch_has_no_variance():
    """A batch with E_var but no F_var (or vice versa) must not fabricate a force block."""
    pred, ref = build([1.0], [1.0])
    pred = ModelOutput(E=pred.E, F=pred.F, E_var=pred.E_var, F_var=None, units=pred.units)
    result = UQAccumulator()
    result.update(pred, ref)
    record = result.compute()["per_key"][0]
    assert record["energy"] is not None
    assert record["force"] is None


# --------------------------------------------------------------------------
# 2. Coverage and ECE - hand-computed
# --------------------------------------------------------------------------
#
# std = 1 for every frame, so z == error. Nominal half-widths (scipy
# norm.ppf(0.5 + p/2)): 50% -> 0.67449, 80% -> 1.28155, 90% -> 1.64485,
# 95% -> 1.95996. Errors chosen to straddle each boundary exactly so the
# empirical count is unambiguous:
#
#   error   |z|<=0.67449  |z|<=1.28155  |z|<=1.64485  |z|<=1.95996
#   0.0          y             y              y             y
#   0.5          y             y              y             y
#   0.7          n             y              y             y
#   1.0          n             y              y             y
#   1.3          n             n              y             y
#   1.7          n             n              n             y
#   2.0          n             n              n             n
#   3.0          n             n              n             n
#   count         2             4              5             6
#   coverage    2/8=.25       4/8=.5        5/8=.625      6/8=.75
#
# ECE = mean(|.25-.5|, |.5-.8|, |.625-.9|, |.75-.95|)
#     = mean(.25, .3, .275, .2) = 1.025/4 = 0.25625

ERRORS = [0.0, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0, 3.0]


def test_coverage_matches_hand_count():
    block = energy_block(build(ERRORS, [1.0] * len(ERRORS)))
    assert block["coverage"] == {"50%": 0.25, "80%": 0.5, "90%": 0.625, "95%": 0.75}


def test_ece_matches_hand_computed_gap():
    block = energy_block(build(ERRORS, [1.0] * len(ERRORS)))
    assert block["ece"] == pytest.approx(0.25625)


def test_coverage_undershoots_nominal_when_overconfident():
    """std stated 5x too small -> every nominal level's empirical coverage must fall below it."""
    rng = np.random.default_rng(0)
    n = 20_000
    true_std = 1.0
    stated_std = true_std / 5.0
    errors = rng.normal(0.0, true_std, size=n)
    block = energy_block(build(errors.tolist(), [stated_std] * n))
    for level, empirical in block["coverage"].items():
        nominal = {"50%": 0.5, "80%": 0.8, "90%": 0.9, "95%": 0.95}[level]
        assert empirical < nominal
    assert block["ece"] > 0.2  # badly miscalibrated, not a borderline case


def test_ece_is_small_for_correctly_calibrated_gaussian_errors():
    rng = np.random.default_rng(1)
    n = 20_000
    std = rng.uniform(0.5, 2.0, size=n)
    errors = rng.normal(0.0, 1.0, size=n) * std  # error ~ N(0, std^2) per frame
    block = energy_block(build(errors.tolist(), std.tolist()))
    assert block["ece"] < 0.02


# --------------------------------------------------------------------------
# 3. Spearman uncertainty-error correlation
# --------------------------------------------------------------------------

def test_spearman_is_none_when_std_has_no_spread():
    block = energy_block(build(ERRORS, [1.0] * len(ERRORS)))
    assert block["spearman_uncertainty_error"] is None


def test_spearman_is_one_when_error_tracks_std_exactly():
    std = [1.0, 2.0, 3.0, 4.0, 5.0]
    block = energy_block(build(std, std))  # |error| == std, both strictly increasing
    assert block["spearman_uncertainty_error"] == pytest.approx(1.0)


def test_spearman_is_negative_one_when_error_runs_opposite_to_std():
    std = [1.0, 2.0, 3.0, 4.0, 5.0]
    errors = [5.0, 4.0, 3.0, 2.0, 1.0]
    block = energy_block(build(errors, std))
    assert block["spearman_uncertainty_error"] == pytest.approx(-1.0)


# --------------------------------------------------------------------------
# 4. Risk-coverage curve
# --------------------------------------------------------------------------

def test_risk_coverage_hand_computed_monotonic_case():
    """std == |error| == [1,2,3,4,5]; sorted by std ascending, MAE at coverage
    c is the mean of the c*n smallest errors - computable exactly."""
    std = [1.0, 2.0, 3.0, 4.0, 5.0]
    block = energy_block(build(std, std))
    curve = {row["coverage"]: row["mae"] for row in block["risk_coverage"]}
    # coverage 0.2 -> k=1 -> mean([1]) = 1.0
    assert curve[0.2] == pytest.approx(1.0)
    # coverage 0.6 -> k=3 -> mean([1,2,3]) = 2.0
    assert curve[0.6] == pytest.approx(2.0)
    # coverage 1.0 -> k=5 -> mean([1,2,3,4,5]) = 3.0
    assert curve[1.0] == pytest.approx(3.0)


def test_risk_coverage_error_is_non_decreasing_as_coverage_grows():
    """Retaining more of a std-sorted, error-correlated set cannot lower the MAE."""
    std = [1.0, 2.0, 3.0, 4.0, 5.0]
    block = energy_block(build(std, std))
    maes = [row["mae"] for row in block["risk_coverage"]]
    assert maes == sorted(maes)


# --------------------------------------------------------------------------
# 5. Gaussian NLL / CRPS - hand-computed at error=0, std=1
# --------------------------------------------------------------------------
#
# NLL = 0.5*log(2*pi*std^2) + error^2/(2*std^2); at error=0, std=1:
#     = 0.5*log(2*pi) = 0.9189385332046727
#
# CRPS(N(0,1), y=0) = std*(z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)), z=0:
#     = 2*phi(0) - 1/sqrt(pi) = 2/sqrt(2*pi) - 1/sqrt(pi) = 0.23369497725510913

def test_gaussian_nll_hand_computed():
    block = energy_block(build([0.0], [1.0]))
    assert block["nll_gaussian"] == pytest.approx(0.9189385332046727)


def test_gaussian_crps_hand_computed():
    block = energy_block(build([0.0], [1.0]))
    assert block["crps_gaussian"] == pytest.approx(0.23369497725510913)


def test_nll_penalises_overconfidence_more_than_a_correct_miss():
    """Same |error|, two stds: the one that claims to be certain (small std)
    must score worse (higher NLL) for missing by the same amount."""
    confident = energy_block(build([2.0], [0.1]))
    honest = energy_block(build([2.0], [2.0]))
    assert confident["nll_gaussian"] > honest["nll_gaussian"]


# --------------------------------------------------------------------------
# 6. Sharpness
# --------------------------------------------------------------------------

def test_sharpness_is_the_mean_predictive_std():
    block = energy_block(build([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]))
    assert block["sharpness_mean_std"] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# 7. Force quantity uses the same machinery, keyed separately from energy
# --------------------------------------------------------------------------

def test_force_block_is_computed_independently_of_energy():
    pred, ref = build(
        [0.0], [1.0],
        force_errors=[[0.0, 0.0, 0.0]], force_std=[[1.0, 1.0, 1.0]],
    )
    acc = UQAccumulator()
    acc.update(pred, ref)
    record = acc.compute()["per_key"][0]
    assert record["force"]["n"] == 3  # one frame, one atom, 3 components
    assert record["force"]["nll_gaussian"] == pytest.approx(0.9189385332046727)
