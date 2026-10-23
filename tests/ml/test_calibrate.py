"""Tests for ml/training/calibrate.py - post-hoc conformal calibration
(SCRUM-65 follow-up to SCRUM-61). conformal_scale() is pure numpy and
fully testable without a trained model; calibrate_and_report() is
exercised against the real checkpoints in the slower integration test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.training.calibrate import calibrate_and_report, conformal_scale

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
