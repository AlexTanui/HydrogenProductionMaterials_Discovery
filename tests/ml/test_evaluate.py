"""Tests for ml/training/evaluate.py (SCRUM-52).

Every model in the Phase 1 bake-off gets its final numbers from that
module, so the two properties tested hardest here are the two that would
corrupt a benchmark table silently:

1. **Model-agnosticism.** The tests below score a deliberately trivial
   model that has nothing to do with PaiNN. If any PaiNN-specific
   assumption ever leaks into `evaluate.py`, these fail.
2. **Streaming accumulation.** Scoring the same data at several batch
   sizes - including ones that leave a short final batch - must give
   bit-identical metrics. Per-batch averaging would not: it is wrong
   whenever batch sizes differ, and always wrong for RMSE, because the
   square root does not commute with the mean.

The fixtures build a miniature gold-format file rather than touching the
real 229 MB ethanol data, so these run in seconds and need no LFS.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from torch import nn

from ml.models.registry import MODEL_REGISTRY, load_model, save_checkpoint
from ml.training.evaluate import evaluate_checkpoint

_Z = [6, 6, 8, 1, 1, 1, 1, 1, 1]

# Deliberately unequal so the final batch is short, which is the case that
# separates streaming accumulation from per-batch averaging.
N_TEST_CONFIGS = 11


class ToyModel(nn.Module):
    """A trivial conservative potential. Shares PaiNN's interface, nothing else.

    The energy is a smooth function of interatomic distances, so forces are
    a genuine `-dE/dR` and the module exercises exactly the same code path
    in `evaluate.py` that a real model does - without a single line of
    PaiNN in it.
    """

    def __init__(self, weight: float = 0.5) -> None:
        super().__init__()
        self.weight = weight
        self.scale = nn.Parameter(torch.tensor(float(weight)))

    def config(self) -> dict:
        return {"weight": self.weight}

    def predict_energy(self, z, pos, edge_index, edge_attr, batch):
        j, i = edge_index[0], edge_index[1]
        distance = torch.sqrt(torch.sum((pos[j] - pos[i]) ** 2, dim=-1) + 1e-8)
        per_edge = self.scale / distance
        num_graphs = int(batch.max()) + 1
        energy = torch.zeros(num_graphs, dtype=pos.dtype, device=pos.device)
        return energy.index_add(0, batch[i], per_edge)

    def predict_energy_and_forces(self, z, pos, edge_index, edge_attr, batch):
        with torch.enable_grad():
            if not pos.requires_grad:
                pos = pos.detach().requires_grad_(True)
            energy = self.predict_energy(z, pos, edge_index, edge_attr, batch)
            (grad,) = torch.autograd.grad(energy.sum(), pos, create_graph=self.training)
        return energy, -grad


@pytest.fixture
def toy_registered():
    """Register ToyModel for the duration of one test, then restore."""
    MODEL_REGISTRY["toy"] = ToyModel
    try:
        yield "toy"
    finally:
        MODEL_REGISTRY.pop("toy", None)


@pytest.fixture
def gold_file(tmp_path):
    """A gold-format .npz with the key layout ml/data/preprocessing.py writes."""
    rng = np.random.default_rng(7)
    n_configs, n_atoms = 30, len(_Z)
    path = tmp_path / "toy_dft.npz"
    np.savez_compressed(
        path,
        z=np.array(_Z, dtype=np.int64),
        R=(rng.normal(size=(n_configs, n_atoms, 3)) * 1.4).astype(np.float32),
        E=rng.normal(size=n_configs).astype(np.float32),
        F=rng.normal(size=(n_configs, n_atoms, 3)).astype(np.float32),
        # Contiguous trajectory blocks, as trajectory_block_split produces.
        train_idx=np.arange(0, 15),
        val_idx=np.arange(15, 30 - N_TEST_CONFIGS),
        test_idx=np.arange(30 - N_TEST_CONFIGS, 30),
        molecule="toy",
        theory="dft",
        used_literature_split=False,
    )
    return path


@pytest.fixture
def checkpoint(tmp_path, gold_file, toy_registered):
    path = tmp_path / "toy.pt"
    save_checkpoint(
        path,
        model=ToyModel(),
        phase=toy_registered,
        data={
            "gold_path": str(gold_file),
            "molecule": "toy",
            "theory": "dft",
            "cutoff_radius": 5.0,
            "num_rbf": 16,
        },
        train={"wall_clock_s": 12.5},
    )
    return path


# --------------------------------------------------------------------------
# 1. Model-agnosticism
# --------------------------------------------------------------------------

def test_scores_a_model_that_is_not_painn(checkpoint):
    """No PaiNN-specific logic: a toy conservative potential scores fine."""
    result = evaluate_checkpoint(checkpoint, batch_size=4)

    assert result["phase"] == "toy"
    assert result["benchmark_row"]["model"] == "ToyModel"
    assert result["data"]["n_configs"] == N_TEST_CONFIGS
    assert result["metrics"]["per_key"][0]["n_frames"] == N_TEST_CONFIGS


def test_unregistered_phase_raises_with_a_useful_message(tmp_path, gold_file):
    from ml.models.registry import UnknownModel

    torch.save(
        {"phase": "not_a_model", "model_config": {}, "state_dict": {}},
        tmp_path / "bad.pt",
    )
    with pytest.raises(UnknownModel, match="registered"):
        evaluate_checkpoint(tmp_path / "bad.pt")


def test_bare_state_dict_is_rejected(tmp_path):
    """A checkpoint with no model config cannot be rebuilt - say so plainly."""
    torch.save({"scale": torch.tensor(1.0)}, tmp_path / "bare.pt")
    with pytest.raises(ValueError, match="missing checkpoint field"):
        evaluate_checkpoint(tmp_path / "bare.pt")


# --------------------------------------------------------------------------
# 2. Streaming accumulation: batch size must not change a single digit
# --------------------------------------------------------------------------

# Separates float64 summation roundoff from a real averaging bug. Changing
# the batch size regroups the terms inside each `torch.sum`, so the running
# totals differ in the last ulp or two - a relative error near 1e-16. A
# mean-of-batch-means is wrong by several percent on an uneven split, which
# is fourteen orders of magnitude larger.
# `test_mean_of_means_would_fail_this_tolerance` pins that gap, so this
# number cannot be quietly loosened until the test stops meaning anything.
BATCH_INVARIANCE_REL_TOL = 1e-12


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 7, N_TEST_CONFIGS])
def test_metrics_are_batch_size_invariant(checkpoint, batch_size):
    """11 test configs split 6 ways, most leaving a short final batch.

    Per-batch averaging would drift here: with batches of 4+4+3 the last
    batch's error would carry the same weight as the full ones. Streaming
    sums with a single final division cannot.
    """
    reference = evaluate_checkpoint(checkpoint, batch_size=N_TEST_CONFIGS)
    result = evaluate_checkpoint(checkpoint, batch_size=batch_size)

    expected = reference["metrics"]["per_key"][0]
    actual = result["metrics"]["per_key"][0]

    compared = 0
    for field in expected:
        if isinstance(expected[field], float):
            assert math.isclose(
                actual[field], expected[field], rel_tol=BATCH_INVARIANCE_REL_TOL
            ), f"{field} moved at batch_size={batch_size}"
            compared += 1
    assert compared == 8, "expected all 8 metric fields to be compared"


def test_mean_of_means_would_fail_this_tolerance(checkpoint, gold_file):
    """Proves `BATCH_INVARIANCE_REL_TOL` is tight enough to catch the real bug.

    Without this, the tolerance above could be loosened until it passed
    anything at all. Here the wrong calculation is performed deliberately -
    one MAE per batch, then an unweighted mean over batches - on the same
    4 + 4 + 3 split, and shown to miss by orders of magnitude more than the
    tolerance permits.
    """
    from torch_geometric.loader import DataLoader

    from ml.data.datasets import MD17Dataset

    model, _ = load_model(checkpoint)
    dataset = MD17Dataset(gold_file, split="test", cutoff_radius=5.0, num_rbf=16)

    # Per-frame absolute energy errors, one batch at a time.
    per_batch_maes = []
    all_errors = []
    for batch in DataLoader(dataset, batch_size=4, shuffle=False):
        energy, _ = model.predict_energy_and_forces(
            batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch
        )
        # Match ml/utils/metrics.py's order exactly: subtract in the model's
        # own dtype, *then* promote to float64. Promoting first instead
        # disagrees at float32 epsilon (~1e-8 relative), which would swamp
        # the 1e-12 tolerance being validated here.
        errors = (energy.detach() - batch.y).double().abs()
        all_errors.append(errors)
        per_batch_maes.append(float(errors.mean()))

    assert [len(e) for e in all_errors] == [4, 4, 3], "the split must be uneven"

    streamed = float(torch.cat(all_errors).mean())
    mean_of_means = sum(per_batch_maes) / len(per_batch_maes)

    relative_error = abs(mean_of_means - streamed) / streamed
    assert relative_error > 1e4 * BATCH_INVARIANCE_REL_TOL, (
        "mean-of-batch-means happens to agree on this fixture, so the "
        "batch-invariance test above proves nothing - change the fixture"
    )

    # And the streaming path in evaluate.py agrees with the honest mean.
    reported = evaluate_checkpoint(checkpoint, batch_size=4)
    assert math.isclose(
        reported["metrics"]["per_key"][0]["energy_mae_total"],
        streamed,
        rel_tol=BATCH_INVARIANCE_REL_TOL,
    )


def test_short_final_batch_is_actually_exercised(checkpoint):
    """Guard the test above: confirm the split really is uneven."""
    result = evaluate_checkpoint(checkpoint, batch_size=4)
    assert N_TEST_CONFIGS % 4 != 0
    assert result["timing"]["n_batches"] == 3  # 4 + 4 + 3


# --------------------------------------------------------------------------
# 3. The test split, and only the test split
# --------------------------------------------------------------------------

def test_reads_the_test_split_by_default(checkpoint, gold_file):
    """Split indices come from the gold file; nothing is re-derived."""
    result = evaluate_checkpoint(checkpoint)
    stored = np.load(gold_file, allow_pickle=True)

    assert result["data"]["split"] == "test"
    assert result["data"]["n_configs"] == len(stored["test_idx"])
    assert result["warnings"] == []


def test_non_test_split_is_flagged_as_unreportable(checkpoint):
    """Allowed for development, but never mistakable for a final number."""
    result = evaluate_checkpoint(checkpoint, split="val")

    assert result["data"]["split"] == "val"
    assert any("not a reportable number" in w for w in result["warnings"])


def test_scoring_a_mismatched_molecule_raises(checkpoint, tmp_path):
    """A checkpoint trained on one molecule must not be scored against another."""
    rng = np.random.default_rng(1)
    other = tmp_path / "other_dft.npz"
    np.savez_compressed(
        other,
        z=np.array(_Z, dtype=np.int64),
        R=(rng.normal(size=(6, len(_Z), 3)) * 1.4).astype(np.float32),
        E=rng.normal(size=6).astype(np.float32),
        F=rng.normal(size=(6, len(_Z), 3)).astype(np.float32),
        train_idx=np.arange(0, 4), val_idx=np.arange(4, 5), test_idx=np.arange(5, 6),
        molecule="somethingelse", theory="dft", used_literature_split=False,
    )
    with pytest.raises(ValueError, match="trained on molecule"):
        evaluate_checkpoint(checkpoint, gold_path=other)


# --------------------------------------------------------------------------
# 4. The output record
# --------------------------------------------------------------------------

def test_result_is_json_serialisable(checkpoint):
    """experiments/results/*.json is read by the backend's /benchmarks route."""
    result = evaluate_checkpoint(checkpoint, batch_size=4)
    round_tripped = json.loads(json.dumps(result))
    assert round_tripped["benchmark_row"]["energy_mae"] == result["benchmark_row"]["energy_mae"]


def test_units_are_reported_alongside_every_number(checkpoint):
    """A bare MAE is not interpretable; the unit must travel with it."""
    row = evaluate_checkpoint(checkpoint)["benchmark_row"]

    assert row["energy_mae_unit"] == "kcal/mol"
    assert row["force_mae_unit"] == "kcal/mol/Ang"
    # And the convention that produced each, since metrics.py computes two of each.
    assert row["energy_mae_field"] == "energy_mae_total"
    assert row["force_mae_field"] == "force_mae_component"


def test_uncertainty_fields_are_null_not_invented(checkpoint):
    """ECE and uncertainty-error correlation are Dongxiao's to define."""
    row = evaluate_checkpoint(checkpoint)["benchmark_row"]

    assert row["ece"] is None
    assert row["uncertainty_correlation"] is None
    assert "Dongxiao" in row["uncertainty_note"]


def test_timing_is_recorded_per_config(checkpoint):
    result = evaluate_checkpoint(checkpoint, batch_size=4)
    timing = result["timing"]

    assert timing["inference_s_per_config"] > 0
    assert timing["model_inference_s"] <= timing["eval_wall_clock_s"]
    # Training wall-clock is measured by the training loop and carried
    # through the checkpoint, not re-derived here.
    assert timing["train_wall_clock_s"] == 12.5


def test_checkpoint_round_trips_through_the_registry(checkpoint):
    """Weights and buffers must survive save -> load unchanged."""
    model, stored = load_model(checkpoint)

    assert stored["phase"] == "toy"
    assert isinstance(model, ToyModel)
    assert not model.training  # load_model returns an eval-mode model
