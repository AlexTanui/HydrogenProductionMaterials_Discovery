"""Tests for ml/utils/metrics.py (SCRUM-37).

Every expected value below is hand-computed and written as a literal, with
its derivation in a comment. Nothing here calls the function under test to
produce an expectation - that would only prove the code agrees with itself.

Fixtures are built so the reference is all zeros and the prediction carries
exactly the intended error, which keeps each expectation readable.

All fixture values are small integers or exact binary fractions, so every
intermediate sum is exact in float64 and equality assertions are bit-exact
rather than approximate. That is deliberate: the batching test below is a
regression test, and an approximate comparison there could hide precisely
the drift it exists to catch.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from ml.data.md17 import MD17Sample
from ml.utils.contract import (
    ContractViolation,
    IncomparableData,
    ModelOutput,
    ReferenceData,
    Units,
    require_comparable,
)
from ml.utils.metrics import (
    MetricAccumulator,
    energy_mae,
    energy_rmse,
    force_mae,
    force_rmse,
)


def build(
    energy_errors,
    force_errors=None,
    n_atoms=1,
    molecule="toy",
    theory="dft",
    units=None,
    stochastic=False,
):
    """Reference of zeros plus a prediction equal to the intended error."""
    units = units or Units()
    n_frames = len(energy_errors)
    if force_errors is None:
        force_errors = [[0.0, 0.0, 0.0]] * (n_frames * n_atoms)
    assert len(force_errors) == n_frames * n_atoms

    ref = ReferenceData(
        E=torch.zeros(n_frames, dtype=torch.float64),
        F=torch.zeros(n_frames * n_atoms, 3, dtype=torch.float64),
        batch=torch.repeat_interleave(torch.arange(n_frames), n_atoms),
        molecule=molecule,
        theory=theory,
        units=units,
    )
    pred_E = torch.tensor(energy_errors, dtype=torch.float64)
    pred_F = torch.tensor(force_errors, dtype=torch.float64)
    pred = ModelOutput(
        E=pred_E,
        F=pred_F,
        E_var=torch.full_like(pred_E, 0.25) if stochastic else None,
        F_var=torch.full_like(pred_F, 0.5) if stochastic else None,
        units=units,
    )
    return pred, ref


def scores(*batches):
    """Accumulate the given (pred, ref) batches and return the single key's record."""
    accumulator = MetricAccumulator()
    for pred, ref in batches:
        accumulator.update(pred, ref)
    return accumulator.compute()["per_key"][0]


# --------------------------------------------------------------------------
# 1. total vs per_atom must differ
# --------------------------------------------------------------------------

def test_energy_total_and_per_atom_differ():
    # 3 atoms per frame, energy errors +2 and -4.
    #   total     MAE  = (2 + 4) / 2                   = 3.0
    #   total     RMSE = sqrt((4 + 16) / 2) = sqrt(10) = 3.1622776601683795
    #   per_atom  MAE  = (2/3 + 4/3) / 2               = 1.0
    #   per_atom  RMSE = sqrt(((2/3)^2 + (4/3)^2) / 2) = 1.0540925533894598
    result = scores(build([2.0, -4.0], n_atoms=3))

    assert result["energy_mae_total"] == 3.0
    assert result["energy_rmse_total"] == 3.1622776601683795
    assert result["energy_mae_per_atom"] == 1.0
    assert result["energy_rmse_per_atom"] == 1.0540925533894598

    assert result["energy_mae_total"] != result["energy_mae_per_atom"]
    assert result["energy_rmse_total"] != result["energy_rmse_per_atom"]


# --------------------------------------------------------------------------
# 2. component vs atom_norm must differ
# --------------------------------------------------------------------------

def test_force_component_and_atom_norm_differ():
    # One frame, two atoms, force errors (3, 4, 0) and (1, 0, 0).
    #   component MAE  = (3+4+0 + 1+0+0) / (2 atoms * 3) = 8/6
    #                                                    = 1.3333333333333333
    #   component RMSE = sqrt((9+16+0 + 1) / 6) = sqrt(26/6)
    #                                                    = 2.0816659994661326
    #   atom_norm MAE  = (|(3,4,0)| + |(1,0,0)|) / 2 = (5 + 1) / 2 = 3.0
    #   atom_norm RMSE = sqrt((25 + 1) / 2) = sqrt(13)   = 3.605551275463989
    result = scores(build([0.0], force_errors=[[3.0, 4.0, 0.0], [1.0, 0.0, 0.0]], n_atoms=2))

    assert result["force_mae_component"] == 1.3333333333333333
    assert result["force_rmse_component"] == 2.0816659994661326
    assert result["force_mae_atom_norm"] == 3.0
    assert result["force_rmse_atom_norm"] == 3.605551275463989

    # The MAE modes are not related by a fixed factor: here the ratio is
    # 3.0 / 1.333... = 2.25, whereas for an isotropic error it is sqrt(3).
    assert result["force_mae_atom_norm"] / result["force_mae_component"] == 2.25

    # The RMSE modes share a sum of squares and differ only in denominator,
    # so this ratio is exactly sqrt(3) for any input. Pinned as a property.
    assert result["force_rmse_atom_norm"] == pytest.approx(
        math.sqrt(3.0) * result["force_rmse_component"], rel=1e-12
    )


# --------------------------------------------------------------------------
# 3. MAE vs RMSE under a single large outlier
# --------------------------------------------------------------------------

def test_mae_and_rmse_differ_with_outlier():
    # Errors 1, 1, 1, 5 - one outlier, which RMSE punishes and MAE does not.
    #   MAE  = (1 + 1 + 1 + 5) / 4              = 2.0
    #   RMSE = sqrt((1 + 1 + 1 + 25) / 4) = sqrt(7) = 2.6457513110645907
    result = scores(build([1.0, 1.0, 1.0, 5.0], n_atoms=2))

    assert result["energy_mae_total"] == 2.0
    assert result["energy_rmse_total"] == 2.6457513110645907
    assert result["energy_rmse_total"] > result["energy_mae_total"]


# --------------------------------------------------------------------------
# 4. THE regression test: unequal batches must match one batch
# --------------------------------------------------------------------------

ERRORS_7 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 8.0]
FORCES_7 = [[c, 0.0, 0.0] for c in ERRORS_7 for _ in range(2)]  # 2 atoms/frame

# 7 frames, 2 atoms each.
#   energy MAE  total = (6*1 + 8) / 7 = 14/7            = 2.0
#   energy RMSE total = sqrt((6*1 + 64) / 7) = sqrt(10) = 3.1622776601683795
#   force  MAE  component = 2*(6*1 + 8) / (14 atoms * 3) = 28/42
#                                                        = 0.6666666666666666
#   force  RMSE component = sqrt(2*(6+64) / 42) = sqrt(140/42)
#                                                        = 1.8257418583505538
#   force  MAE  atom_norm = 28 / 14                      = 2.0
#   force  RMSE atom_norm = sqrt(140 / 14) = sqrt(10)    = 3.1622776601683795
EXPECTED_7 = {
    "energy_mae_total": 2.0,
    "energy_rmse_total": 3.1622776601683795,
    "force_mae_component": 0.6666666666666666,
    "force_rmse_component": 1.8257418583505538,
    "force_mae_atom_norm": 2.0,
    "force_rmse_atom_norm": 3.1622776601683795,
}

# What averaging per-batch values would have produced for the 4+2+1 split:
# batch MAEs are 1.0, 1.0, 8.0, so their unweighted mean is 10/3. The same
# figure comes out of averaging the batch RMSEs. Both are wrong.
NAIVE_MEAN_OF_BATCH_VALUES = 3.3333333333333335


def _slice(frames, n_atoms=2):
    """Build one batch from a subset of ERRORS_7/FORCES_7, by frame index."""
    energy = [ERRORS_7[i] for i in frames]
    force = [FORCES_7[i * n_atoms + a] for i in frames for a in range(n_atoms)]
    return build(energy, force_errors=force, n_atoms=n_atoms)


def test_single_batch_matches_expected():
    result = scores(_slice(range(7)))
    for name, expected in EXPECTED_7.items():
        assert result[name] == expected, name


def test_unequal_batches_match_one_batch_exactly():
    """4 + 2 + 1 - deliberately unequal, ending in a batch of one.

    This is the regression test for averaging per-batch metrics. A harness
    that averaged batch MAEs or batch RMSEs would report 3.333... here for
    both statistics instead of 2.0 and 3.162....
    """
    one = scores(_slice(range(7)))
    split = scores(_slice([0, 1, 2, 3]), _slice([4, 5]), _slice([6]))

    for name in EXPECTED_7:
        assert split[name] == one[name], f"{name} changed when the data was split"
        assert split[name] == EXPECTED_7[name], name

    # And prove the test would catch the bug it exists for.
    assert split["energy_mae_total"] != NAIVE_MEAN_OF_BATCH_VALUES
    assert split["energy_rmse_total"] != NAIVE_MEAN_OF_BATCH_VALUES

    assert split["n_frames"] == 7
    assert split["n_atoms"] == 14


def test_many_uneven_batches_still_match():
    """Same data again as 1+1+1+1+1+1+1, the pathological extreme."""
    one = scores(_slice(range(7)))
    singles = scores(*[_slice([i]) for i in range(7)])
    for name in EXPECTED_7:
        assert singles[name] == one[name], name


# --------------------------------------------------------------------------
# 5. Units must never be compared silently
# --------------------------------------------------------------------------

def test_prediction_and_reference_unit_mismatch_rejected():
    pred, _ = build([1.0], units=Units(energy="kcal/mol"))
    _, ref = build([1.0], units=Units(energy="eV"))
    accumulator = MetricAccumulator()
    with pytest.raises(ContractViolation, match="unit mismatch"):
        accumulator.update(pred, ref)


def test_unit_mismatch_between_batches_rejected():
    """Same molecule and theory, second batch in a different unit."""
    accumulator = MetricAccumulator()
    accumulator.update(*build([1.0], units=Units(energy="kcal/mol")))
    with pytest.raises(IncomparableData, match="to_project_units"):
        accumulator.update(*build([1.0], units=Units(energy="eV")))


def test_unit_conversion_scales_mean_linearly_and_variance_quadratically():
    pred, ref = build([1.0], units=Units(energy="eV"), stochastic=True)
    converted = pred.to_project_units()
    factor = 23.060547830618307  # eV -> kcal/mol
    assert converted.E.item() == pytest.approx(factor, rel=1e-12)
    assert converted.E_var.item() == pytest.approx(0.25 * factor * factor, rel=1e-12)
    assert converted.units.energy == "kcal/mol"
    assert ref.to_project_units().units.is_project_units


# --------------------------------------------------------------------------
# 6. Theory levels must never be pooled
# --------------------------------------------------------------------------

def test_mixing_theories_rejected_by_contract():
    _, dft = build([1.0], molecule="ethanol", theory="dft")
    _, ccsd = build([1.0], molecule="ethanol", theory="ccsd_t")
    with pytest.raises(IncomparableData, match="separate keys"):
        require_comparable(dft, ccsd)


def test_two_theories_report_separately_and_energy_is_not_pooled():
    accumulator = MetricAccumulator()
    accumulator.update(*build([2.0], molecule="ethanol", theory="dft", n_atoms=3))
    accumulator.update(*build([4.0], molecule="ethanol", theory="ccsd_t", n_atoms=3))
    results = accumulator.compute()

    assert len(results["per_key"]) == 2
    by_theory = {r["theory"]: r for r in results["per_key"]}
    assert by_theory["dft"]["energy_mae_total"] == 2.0
    assert by_theory["ccsd_t"]["energy_mae_total"] == 4.0

    overall = results["overall"]
    # Forces pool - same units, local per-atom quantities.
    assert "force_mae_component" in overall
    # Energy does not, and the reason is recorded rather than implied.
    for name in ("energy_mae_total", "energy_rmse_total", "energy_mae_per_atom"):
        assert name not in overall
    assert "theory" in overall["excluded"]["energy"]


def test_one_key_pools_energy():
    accumulator = MetricAccumulator()
    accumulator.update(*build([2.0, -4.0], molecule="ethanol", theory="dft", n_atoms=3))
    overall = accumulator.compute()["overall"]
    assert overall["n_keys"] == 1
    assert overall["energy_mae_total"] == 3.0
    assert overall["excluded"] == {}


# --------------------------------------------------------------------------
# 7. Uncertainty must not disturb accuracy
# --------------------------------------------------------------------------

def test_stochastic_output_scores_identically_to_deterministic():
    deterministic = scores(build([2.0, -4.0], n_atoms=3, stochastic=False))
    stochastic = scores(build([2.0, -4.0], n_atoms=3, stochastic=True))

    for name in (
        "energy_mae_total", "energy_rmse_total",
        "energy_mae_per_atom", "energy_rmse_per_atom",
        "force_mae_component", "force_rmse_component",
        "force_mae_atom_norm", "force_rmse_atom_norm",
    ):
        assert stochastic[name] == deterministic[name], name

    # The variance is recorded but not scored here; ECE and the
    # uncertainty-error correlation are Dongxiao's to define.
    assert stochastic["stochastic"] is True
    assert deterministic["stochastic"] is False


# --------------------------------------------------------------------------
# 8. Convenience functions share the accumulator's code path
# --------------------------------------------------------------------------

def test_convenience_functions_return_hand_computed_values():
    pred, ref = build([2.0, -4.0], force_errors=[[3.0, 4.0, 0.0], [1.0, 0.0, 0.0]], n_atoms=1)
    # Energy, 1 atom per frame, errors +2 and -4: total and per_atom coincide.
    assert energy_mae(pred, ref) == 3.0
    assert energy_mae(pred, ref, normalisation="per_atom") == 3.0
    assert energy_rmse(pred, ref) == 3.1622776601683795
    # Force errors (3,4,0) and (1,0,0), now one atom in each of two frames.
    assert force_mae(pred, ref) == 1.3333333333333333
    assert force_mae(pred, ref, mode="atom_norm") == 3.0
    assert force_rmse(pred, ref) == 2.0816659994661326
    assert force_rmse(pred, ref, mode="atom_norm") == 3.605551275463989


def test_unknown_convention_rejected():
    pred, ref = build([1.0])
    with pytest.raises(ValueError, match="normalisation must be one of"):
        energy_mae(pred, ref, normalisation="per_electron")
    with pytest.raises(ValueError, match="mode must be one of"):
        force_mae(pred, ref, mode="magnitude")


# --------------------------------------------------------------------------
# 9. The output is a run record, not just numbers
# --------------------------------------------------------------------------

def test_compute_records_conventions_and_units():
    result = scores(build([2.0], n_atoms=3))
    assert result["units"] == {
        "energy": "kcal/mol", "length": "Ang", "force": "kcal/mol/Ang",
    }
    fields = result["fields"]
    assert fields["energy_mae_total"]["normalisation"] == "total"
    assert fields["energy_mae_total"]["unit"] == "kcal/mol"
    assert fields["energy_rmse_per_atom"]["normalisation"] == "per_atom"
    assert fields["energy_rmse_per_atom"]["unit"] == "kcal/mol/atom"
    assert fields["force_mae_component"]["mode"] == "component"
    assert fields["force_mae_atom_norm"]["mode"] == "atom_norm"
    assert fields["force_mae_atom_norm"]["unit"] == "kcal/mol/Ang"


def test_compute_output_is_json_serialisable():
    accumulator = MetricAccumulator()
    accumulator.update(*build([2.0], molecule="ethanol", theory="dft", n_atoms=3))
    accumulator.update(*build([1.0], molecule="aspirin", theory="dft", n_atoms=3))
    encoded = json.dumps(accumulator.compute())
    assert isinstance(json.loads(encoded)["per_key"], list)


def test_compute_before_update_is_an_error():
    with pytest.raises(ValueError, match="before any update"):
        MetricAccumulator().compute()


def test_reset_clears_state():
    accumulator = MetricAccumulator()
    accumulator.update(*build([2.0, -4.0], n_atoms=3))
    accumulator.reset()
    assert len(accumulator) == 0
    accumulator.update(*build([1.0, 1.0, 1.0, 5.0], n_atoms=2))
    assert accumulator.compute()["per_key"][0]["energy_mae_total"] == 2.0


# --------------------------------------------------------------------------
# 10. The seam onto Shijin's loader
# --------------------------------------------------------------------------

def test_from_sample_builds_a_frame_major_batch():
    # F[frame, atom] = (frame*10 + atom, 0, 0), so the flattened row order is
    # checkable: selecting frames 0 and 2 must give 0, 1, 20, 21.
    n_configs, n_atoms = 3, 2
    forces = np.zeros((n_configs, n_atoms, 3), dtype=np.float32)
    for frame in range(n_configs):
        for atom in range(n_atoms):
            forces[frame, atom, 0] = frame * 10 + atom
    sample = MD17Sample(
        molecule="toy",
        theory="dft",
        z=np.array([1, 6]),
        R=np.zeros((n_configs, n_atoms, 3), dtype=np.float32),
        E=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        F=forces,
    )

    ref = ReferenceData.from_sample(sample, [0, 2])

    assert ref.n_frames == 2
    assert ref.n_atoms == 4
    assert ref.batch.tolist() == [0, 0, 1, 1]
    assert ref.E.tolist() == [1.0, 3.0]
    assert ref.F[:, 0].tolist() == [0.0, 1.0, 20.0, 21.0]
    assert ref.z.tolist() == [1, 6, 1, 6]
    assert ref.atom_counts.tolist() == [2, 2]
    assert ref.key.molecule == "toy" and ref.key.theory == "dft"
    assert ref.units.force == "kcal/mol/Ang"


def test_from_sample_rejects_out_of_range_frames():
    sample = MD17Sample(
        molecule="toy", theory="dft", z=np.array([1]),
        R=np.zeros((2, 1, 3), dtype=np.float32),
        E=np.zeros(2, dtype=np.float32),
        F=np.zeros((2, 1, 3), dtype=np.float32),
    )
    with pytest.raises(ContractViolation, match="out of range"):
        ReferenceData.from_sample(sample, [0, 5])
