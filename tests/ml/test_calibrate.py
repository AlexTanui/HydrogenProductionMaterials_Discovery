"""Tests for ml/training/calibrate.py - post-hoc conformal calibration
(SCRUM-65 follow-up to SCRUM-61; cqr_* added as the 2026-11-04
distribution-free follow-up). conformal_scale()/cqr_* are pure numpy and
fully testable without a trained model; calibrate_and_report()/
calibrate_cqr_and_report() are exercised against the real checkpoints in
the slower integration tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.training.calibrate import (
    calibrate_and_report,
    calibrate_cqr_and_report,
    conformal_scale,
    cqr_bounds,
    cqr_calibrate,
    cqr_nonconformity,
    shape_diagnostics,
    shape_diagnostics_by_atom_type,
)

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "gold" / "md17" / "ethanol_dft.npz"
CHECKPOINT = ROOT / "experiments" / "checkpoints" / "phase3_graph_stochastic.pt"


def test_conformal_scale_widens_an_underestimated_std():
    rng = np.random.default_rng(0)
    error = rng.normal(0, 2.0, size=5000)  # true std 2.0
    std = np.ones(5000)  # model claims std 1.0 - overconfident by 2x
    q = conformal_scale(error, std, alpha=0.1)
    assert q > 1.0  # must widen, not shrink
    # q*std should land near the 90th percentile of |N(0,2)| ~= 2*1.6449
    assert abs(q - 2 * 1.6449) < 0.3


def test_conformal_scale_shrinks_an_overestimated_std():
    rng = np.random.default_rng(1)
    error = rng.normal(0, 0.5, size=5000)  # true std 0.5
    std = np.full(5000, 2.0)  # model claims std 2.0 - overcautious by 4x
    q = conformal_scale(error, std, alpha=0.1)
    assert q < 1.0  # must shrink


def test_conformal_scale_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        conformal_scale(np.zeros(10), np.ones(5))


def test_conformal_scale_clamps_quantile_level_for_tiny_calibration_sets():
    # n=1: (1-alpha)(1+1/n) = 0.9*2 = 1.8 > 1 - must clamp, not raise.
    q = conformal_scale(np.array([1.0]), np.array([1.0]), alpha=0.1)
    assert np.isfinite(q) and q > 0


@pytest.mark.skipif(not GOLD.exists(), reason="ethanol_dft gold data not present")
@pytest.mark.skipif(not CHECKPOINT.exists(), reason="phase3_graph_stochastic.pt not trained")
def test_calibrate_and_report_on_a_real_checkpoint():
    """End to end on real data: Phase 3's own energy uncertainty was
    measured over-cautious (coverage@90 ~99.5%, docs/model_cards/
    graph_stochastic.md) - conformal calibration should shrink it
    (q < 1), and ECE should not get worse."""
    report = calibrate_and_report(CHECKPOINT, val_subset=500, test_subset=200)
    assert report["q_energy"] < 1.0
    assert report["after"]["energy"]["ece"] <= report["before"]["energy"]["ece"] + 0.05


def test_cqr_bounds_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        cqr_bounds(np.zeros(10))


def test_cqr_bounds_recovers_known_quantiles():
    # 20 MC samples per point, drawn from a known Normal - the empirical
    # 5th/95th percentile across samples should land near +-1.645*std.
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1.0, size=(20, 5000))
    lo, hi = cqr_bounds(samples, alpha=0.1)
    assert abs(float(lo.mean()) - (-1.6449)) < 0.3
    assert abs(float(hi.mean()) - 1.6449) < 0.3


def test_cqr_nonconformity_is_zero_margin_at_the_boundary():
    lo, hi, y = np.array([-1.0, -1.0]), np.array([1.0, 1.0]), np.array([-1.0, 1.0])
    scores = cqr_nonconformity(lo, hi, y)
    assert np.allclose(scores, 0.0)


def test_cqr_nonconformity_is_positive_outside_the_interval():
    lo, hi, y = np.array([-1.0]), np.array([1.0]), np.array([3.0])
    scores = cqr_nonconformity(lo, hi, y)
    assert scores[0] == pytest.approx(2.0)


def test_cqr_calibrate_widens_an_overconfident_ensemble():
    # Ensemble's own spread (std ~0.5) badly understates the true error
    # spread (std 2.0) - q_hat should be large and positive to compensate,
    # the additive-correction analogue of conformal_scale's q > 1 case.
    rng = np.random.default_rng(2)
    samples = rng.normal(0, 0.5, size=(20, 5000))
    y = rng.normal(0, 2.0, size=5000)
    q_hat = cqr_calibrate(samples, y, alpha=0.1)
    assert q_hat > 1.0


@pytest.mark.skipif(not GOLD.exists(), reason="ethanol_dft gold data not present")
@pytest.mark.skipif(not CHECKPOINT.exists(), reason="phase3_graph_stochastic.pt not trained")
def test_calibrate_cqr_and_report_on_a_real_checkpoint():
    """Same checkpoint as the scalar-method integration test above: the
    CQR path should also land near the 90% coverage target after its own
    (additive) correction, via a completely different, distribution-free
    route - no Normal assumption, no std anywhere in this path."""
    report = calibrate_cqr_and_report(CHECKPOINT, val_subset=500, test_subset=200)
    assert abs(report["after"]["energy"]["coverage_90"] - 0.9) < 0.15
    assert abs(report["after"]["force"]["coverage_90"] - 0.9) < 0.15
    # q_hat can be negative (shrinking an already-overcautious interval,
    # same direction conformal_scale's q < 1 case covers) or positive
    # (widening an overconfident one) - either way the corrected interval
    # must still be a valid, non-negative-width interval.
    assert report["before"]["energy"]["mean_width"] >= 0
    assert report["after"]["energy"]["mean_width"] >= 0
    assert report["after"]["force"]["mean_width"] >= 0
    # shape_diagnostics is attached automatically - just check the keys made it through.
    assert set(report["shape"]["energy"]) == {
        "ensemble_skew", "ensemble_excess_kurtosis", "ensemble_jarque_bera_stat",
        "ensemble_jarque_bera_p", "residual_skew", "residual_excess_kurtosis",
    }
    # ethanol is C2 O1 H6 (9 atoms) - the atom-type breakdown should cover
    # exactly those three elements and partition every force point (one
    # point per atom per xyz component per test frame: n_test * 9 * 3).
    by_type = report["shape_by_atom_type"]["force"]
    assert set(by_type) == {"C", "O", "H"}
    assert sum(s["n_points"] for s in by_type.values()) == report["n_test"] * 9 * 3
    assert by_type["C"]["n_points"] == report["n_test"] * 2 * 3
    assert by_type["O"]["n_points"] == report["n_test"] * 1 * 3
    assert by_type["H"]["n_points"] == report["n_test"] * 6 * 3


def test_shape_diagnostics_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        shape_diagnostics(np.zeros(10), np.zeros(10))


def test_shape_diagnostics_rejects_mismatched_point_count():
    with pytest.raises(ValueError):
        shape_diagnostics(np.zeros((20, 10)), np.zeros(5))


def test_shape_diagnostics_is_near_zero_for_a_true_gaussian_ensemble():
    # Every point's own ensemble and the pooled residual are both exactly
    # Normal here (same distribution, same scale for every point). Both
    # skews land near 0, the usual null case - but neither excess
    # kurtosis does, at num_mc_samples=20: see shape_diagnostics'
    # docstring for why each has its own nonzero, simulation-verified
    # baseline (~-0.28 for the ensemble, ~+0.4 for the residual) even
    # under a perfectly-calibrated model.
    rng = np.random.default_rng(3)
    true_mean, true_std = 0.0, 1.0
    samples = rng.normal(true_mean, true_std, size=(20, 20000))
    y = rng.normal(true_mean, true_std, size=20000)
    diag = shape_diagnostics(samples, y)
    assert abs(diag["ensemble_skew"]) < 0.1
    assert abs(diag["ensemble_excess_kurtosis"] - (-0.28)) < 0.1
    assert abs(diag["residual_skew"]) < 0.1
    assert abs(diag["residual_excess_kurtosis"] - 0.4) < 0.15


def test_shape_diagnostics_detects_a_skewed_ensemble():
    # chi-squared draws are strongly right-skewed (skew = sqrt(8/k) for k
    # dof) - standardizing per point should not remove that asymmetry.
    rng = np.random.default_rng(4)
    samples = rng.chisquare(df=3, size=(20, 20000))
    y = samples.mean(axis=0) + rng.normal(0, 0.01, size=20000)
    diag = shape_diagnostics(samples, y)
    assert diag["ensemble_skew"] > 0.5


def test_shape_diagnostics_detects_a_heavy_tailed_ensemble():
    # Student-t(3) has excess kurtosis of infinity in theory, very large
    # in a finite sample - should read as strongly positive, unlike the
    # Gaussian null case above.
    rng = np.random.default_rng(5)
    samples = rng.standard_t(df=3, size=(20, 20000))
    y = samples.mean(axis=0) + rng.normal(0, 0.01, size=20000)
    diag = shape_diagnostics(samples, y)
    assert diag["ensemble_excess_kurtosis"] > 1.0


def test_shape_diagnostics_by_atom_type_rejects_misaligned_z():
    with pytest.raises(ValueError):
        shape_diagnostics_by_atom_type(np.zeros((20, 10)), np.zeros(10), np.zeros(5))


def test_shape_diagnostics_by_atom_type_recovers_each_slices_own_baseline():
    # shape_diagnostics already studentizes each POINT by its own ensemble
    # std, so two atom types that just happen to have different force
    # scales/calibration quality (e.g. stiff vs. floppy bonds, or one type
    # systematically over/underconfident) do not by themselves make EITHER
    # type's own residual non-Gaussian - a Gaussian of the "wrong" width is
    # still a Gaussian, excess kurtosis near the usual +0.4 null baseline.
    # The inflation is a pooling artifact: mixing two differently-scaled
    # Gaussians (here, type "H" correctly calibrated at std 1, type "C"
    # 5x overconfident - ensemble std still 1 but true residual std 5) is
    # the textbook Gaussian scale-mixture, which has elevated kurtosis only
    # once pooled - not within either component. So the pooled figure
    # should be clearly inflated, while EACH per-element slice should land
    # back near the single-distribution +0.4/-0.28 baselines, same
    # tolerance as test_shape_diagnostics_is_near_zero_for_a_true_gaussian_ensemble
    # - confirming that splitting by atom type explains away the pooled
    # inflation rather than each type being independently non-Gaussian.
    rng = np.random.default_rng(6)
    n = 20000
    samples_lo = rng.normal(0, 1.0, size=(20, n))
    y_lo = rng.normal(0, 1.0, size=n)
    samples_hi = rng.normal(0, 1.0, size=(20, n))
    y_hi = rng.normal(0, 5.0, size=n)  # same ensemble spread, 5x the true error

    samples = np.concatenate([samples_lo, samples_hi], axis=1)
    y = np.concatenate([y_lo, y_hi])
    z = np.concatenate([np.full(n, 1), np.full(n, 6)])  # H, C

    pooled = shape_diagnostics(samples, y)
    assert pooled["residual_excess_kurtosis"] > 1.0  # scale-mixture inflation when pooled

    by_type = shape_diagnostics_by_atom_type(samples, y, z)
    assert set(by_type) == {"H", "C"}
    for symbol in ("H", "C"):
        assert abs(by_type[symbol]["ensemble_excess_kurtosis"] - (-0.28)) < 0.1
        assert abs(by_type[symbol]["residual_excess_kurtosis"] - 0.4) < 0.15  # back at baseline once isolated
        assert by_type[symbol]["n_points"] == n
