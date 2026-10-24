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

import hashlib
import json
import math

import numpy as np
import pytest
import torch
from torch import nn

from ml.models.registry import MODEL_REGISTRY, load_model, save_checkpoint
from ml.models.scaling import ScaledEnergyModel
from ml.training.evaluate import evaluate_checkpoint, main

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
        # Identity scaling, so the metrics below are ToyModel's own.
        model=ScaledEnergyModel(ToyModel()),
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
        {"checkpoint_version": 2, "phase": "not_a_model", "model_config": {}, "state_dict": {}},
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
    assert isinstance(model, ScaledEnergyModel) and isinstance(model.model, ToyModel)
    assert not model.training  # load_model returns an eval-mode model


# --------------------------------------------------------------------------
# 5. The checkpoint contract: one energy scale-and-shift, version 2
# --------------------------------------------------------------------------

def test_energy_scaling_is_applied_by_evaluate(tmp_path, gold_file, toy_registered):
    """The wrapper's shift and scale travel in the checkpoint and reach the scored energies.

    Expected numbers are computed by hand from the bare ToyModel, so a
    shift lost on save/load, or applied twice, fails here.
    """
    from torch_geometric.loader import DataLoader

    from ml.data.datasets import MD17Dataset

    shift, scale = -3.25, 1.75
    path = tmp_path / "scaled.pt"
    save_checkpoint(
        path, ScaledEnergyModel(ToyModel(), shift=shift, scale=scale), toy_registered,
        data={"gold_path": str(gold_file), "molecule": "toy", "theory": "dft",
              "cutoff_radius": 5.0, "num_rbf": 16},
    )
    record = evaluate_checkpoint(path, batch_size=4)["metrics"]["per_key"][0]

    bare = ToyModel().eval()
    energy_errors, force_errors = [], []
    for batch in DataLoader(MD17Dataset(gold_file, split="test"), batch_size=4):
        energy, forces = bare.predict_energy_and_forces(
            batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch
        )
        expected_energy = scale * energy.detach() + len(_Z) * shift
        energy_errors.append((expected_energy - batch.y).double().abs())
        force_errors.append((scale * forces.detach() - batch.force).double().abs().reshape(-1))

    assert record["energy_mae_total"] == pytest.approx(float(torch.cat(energy_errors).mean()), rel=1e-6)
    assert record["force_mae_component"] == pytest.approx(float(torch.cat(force_errors).mean()), rel=1e-6)


def test_bare_model_cannot_be_saved(tmp_path, toy_registered):
    """A bare model has no energy scale-and-shift; saving one would score ~97,000 kcal/mol off."""
    with pytest.raises(TypeError, match="ScaledEnergyModel"):
        save_checkpoint(tmp_path / "bare.pt", ToyModel(), toy_registered, data={})


def test_version_1_checkpoint_is_refused(tmp_path, toy_registered):
    """Checkpoints from the superseded per-model trainers carry no version and a different energy convention."""
    torch.save(
        {"phase": toy_registered, "model_config": {"weight": 0.5},
         "state_dict": ToyModel().state_dict()},
        tmp_path / "v1.pt",
    )
    with pytest.raises(ValueError, match="checkpoint version 1"):
        load_model(tmp_path / "v1.pt")


def test_checkpoint_write_is_atomic(checkpoint):
    """save_checkpoint renames a finished temporary file into place; none is left behind."""
    assert checkpoint.exists()
    assert not checkpoint.with_name(checkpoint.name + ".tmp").exists()


# --------------------------------------------------------------------------
# 6. A fixed, recorded seed: stochastic models score reproducibly
# --------------------------------------------------------------------------

class NoisyToyModel(ToyModel):
    """ToyModel plus a fresh random energy offset on every call, like BLIP's weight noise."""

    def predict_energy_and_forces(self, z, pos, edge_index, edge_attr, batch):
        energy, forces = super().predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)
        return energy + torch.randn(energy.shape, dtype=energy.dtype), forces


@pytest.fixture
def noisy_checkpoint(tmp_path, gold_file):
    MODEL_REGISTRY["noisy_toy"] = NoisyToyModel
    try:
        path = tmp_path / "noisy.pt"
        save_checkpoint(
            path, ScaledEnergyModel(NoisyToyModel()), "noisy_toy",
            data={"gold_path": str(gold_file), "molecule": "toy", "theory": "dft",
                  "cutoff_radius": 5.0, "num_rbf": 16},
        )
        yield path
    finally:
        MODEL_REGISTRY.pop("noisy_toy", None)


def _energy_mae(result) -> float:
    return result["metrics"]["per_key"][0]["energy_mae_total"]


def test_stochastic_model_scores_identically_under_the_same_seed(noisy_checkpoint):
    first = evaluate_checkpoint(noisy_checkpoint, batch_size=4)
    second = evaluate_checkpoint(noisy_checkpoint, batch_size=4)
    assert _energy_mae(first) == _energy_mae(second)
    assert first["seed"] == second["seed"] == 0
    # And the seed is what fixes it: another seed draws other noise.
    other = evaluate_checkpoint(noisy_checkpoint, batch_size=4, seed=1)
    assert other["seed"] == 1
    assert _energy_mae(other) != _energy_mae(first)


def test_deterministic_model_is_unaffected_by_the_seed(checkpoint):
    results = [evaluate_checkpoint(checkpoint, batch_size=4, seed=seed) for seed in (0, 1, 12345)]
    assert all(r["metrics"] == results[0]["metrics"] for r in results)


def test_scoring_leaves_the_callers_rng_untouched(noisy_checkpoint):
    torch.manual_seed(99)
    expected = torch.rand(3)
    torch.manual_seed(99)
    evaluate_checkpoint(noisy_checkpoint, batch_size=4)
    assert torch.equal(torch.rand(3), expected)


class UncountedMCToyModel(ToyModel):
    """Scored as an MC distribution, like BLIP, but exposes no `num_mc_samples`."""

    calls: list = []  # the num_mc_samples each predict_distribution call received

    def predict_distribution(self, z, pos, edge_index, edge_attr, batch=None, num_mc_samples=None):
        type(self).calls.append(num_mc_samples)
        draws = [self.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)
                 for _ in range(num_mc_samples or 4)]
        energies = torch.stack([e + torch.randn_like(e) for e, _ in draws])
        forces = torch.stack([f for _, f in draws])
        return energies.mean(0), energies.var(0), forces.mean(0), forces.var(0) + 1e-6


class MCToyModel(UncountedMCToyModel):
    def __init__(self, weight: float = 0.5, num_mc_samples: int = 3) -> None:
        super().__init__(weight)
        self.num_mc_samples = num_mc_samples

    def config(self) -> dict:
        return {**super().config(), "num_mc_samples": self.num_mc_samples}


@pytest.fixture
def mc_checkpoint(tmp_path, gold_file):
    """factory(model) -> a saved checkpoint of that MC toy model, registered for the test."""
    registered = []

    def save(model):
        phase = type(model).__name__.lower()
        MODEL_REGISTRY[phase] = type(model)
        registered.append(phase)
        path = tmp_path / f"{phase}.pt"
        save_checkpoint(path, ScaledEnergyModel(model), phase,
                        data={"gold_path": str(gold_file), "molecule": "toy", "theory": "dft",
                              "cutoff_radius": 5.0, "num_rbf": 16})
        return path

    yield save
    for phase in registered:
        MODEL_REGISTRY.pop(phase, None)


def test_the_mc_sample_count_used_is_recorded(mc_checkpoint):
    """SCRUM-49: every stochastic run in a table must share one eval MC count,
    so the record states the count actually passed to predict_distribution."""
    MCToyModel.calls = []
    result = evaluate_checkpoint(mc_checkpoint(MCToyModel(num_mc_samples=3)), batch_size=4)
    assert result["num_mc_samples"] == 3
    assert MCToyModel.calls and set(MCToyModel.calls) == {3}
    assert result["metrics"]["per_key"][0]["stochastic"] is True
    assert json.loads(json.dumps(result))["num_mc_samples"] == 3


def test_a_deterministic_model_records_no_mc_sample_count(checkpoint):
    assert evaluate_checkpoint(checkpoint, batch_size=4)["num_mc_samples"] is None


def test_a_distribution_model_without_a_count_is_scored_with_a_warning(mc_checkpoint):
    UncountedMCToyModel.calls = []
    result = evaluate_checkpoint(mc_checkpoint(UncountedMCToyModel()), batch_size=4)
    assert result["num_mc_samples"] is None
    assert set(UncountedMCToyModel.calls) == {None}  # the model's own default ran
    assert any("no num_mc_samples" in warning for warning in result["warnings"])


# --------------------------------------------------------------------------
# 7. The scoring-data fingerprint
# --------------------------------------------------------------------------

def test_fingerprint_identifies_the_scored_data(checkpoint, gold_file):
    first = evaluate_checkpoint(checkpoint, batch_size=4)["data"]["fingerprint"]
    second = evaluate_checkpoint(checkpoint, batch_size=4)["data"]["fingerprint"]
    assert first == second
    assert set(first["parts"]) == {"indices", "z", "R", "E", "F"}
    assert first["dtypes"] == {"indices": "int64", "z": "int64", "R": "float32", "E": "float32", "F": "float32"}
    assert first["shapes"]["R"] == [N_TEST_CONFIGS, len(_Z), 3]


def _rewrite_gold(source, destination, **changes):
    with np.load(source, allow_pickle=True) as gold:
        arrays = {key: gold[key] for key in gold.files}
    arrays.update(changes)
    np.savez_compressed(destination, **arrays)
    return destination


def test_same_size_different_data_changes_the_fingerprint(checkpoint, gold_file, tmp_path):
    """Molecule, theory and frame count all match - only the fingerprint can tell these apart."""
    reference = evaluate_checkpoint(checkpoint, batch_size=4)
    with np.load(gold_file) as gold:
        shifted_R = gold["R"].copy()
    shifted_R[-1, 0, 0] += 1e-3  # one coordinate of one test frame
    other = _rewrite_gold(gold_file, tmp_path / "other_dft.npz", R=shifted_R)
    result = evaluate_checkpoint(checkpoint, gold_path=other, batch_size=4)

    assert result["data"]["n_configs"] == reference["data"]["n_configs"]
    assert result["data"]["molecule"] == reference["data"]["molecule"]
    old, new = reference["data"]["fingerprint"], result["data"]["fingerprint"]
    assert new["sha256"] != old["sha256"]
    assert [part for part in old["parts"] if old["parts"][part] != new["parts"][part]] == ["R"]


def test_different_test_indices_change_the_fingerprint(checkpoint, gold_file, tmp_path):
    """Same frame count, different frames: the indices hash catches it."""
    reference = evaluate_checkpoint(checkpoint, batch_size=4)["data"]["fingerprint"]
    with np.load(gold_file) as gold:
        n = len(gold["R"])
    shifted = _rewrite_gold(
        gold_file, tmp_path / "shifted_dft.npz",
        train_idx=np.arange(1, 16), val_idx=np.arange(16, n - N_TEST_CONFIGS + 1),
        test_idx=np.concatenate([[0], np.arange(n - N_TEST_CONFIGS + 1, n)]),
    )
    result = evaluate_checkpoint(checkpoint, gold_path=shifted, batch_size=4)["data"]["fingerprint"]
    assert result["parts"]["indices"] != reference["parts"]["indices"]
    assert result["sha256"] != reference["sha256"]


# --------------------------------------------------------------------------
# 8. scripts/fingerprint_run_records.py: re-score within tolerance, or write nothing
# --------------------------------------------------------------------------

def _load_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "fingerprint_run_records.py"
    spec = importlib.util.spec_from_file_location("fingerprint_run_records", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legacy_run(tmp_path, checkpoint, relative_error=0.0):
    """A run record as it looked before fingerprints: scored, then the fingerprint removed."""
    import shutil

    runs, checkpoints = tmp_path / "runs", tmp_path / "checkpoints"
    (runs / "toy_run").mkdir(parents=True)
    checkpoints.mkdir()
    shutil.copy(checkpoint, checkpoints / "toy_run.pt")
    stored = evaluate_checkpoint(checkpoint, batch_size=4)
    del stored["data"]["fingerprint"]
    stored["metrics"]["per_key"][0]["force_mae_component"] *= 1 + relative_error
    path = runs / "toy_run" / "test_metrics.json"
    path.write_text(json.dumps(stored), encoding="utf-8")
    return runs, checkpoints, path


def test_rescoring_adds_a_fingerprint_and_leaves_stored_metrics_unchanged(tmp_path, checkpoint):
    script = _load_script()
    runs, checkpoints, path = _legacy_run(tmp_path, checkpoint)
    before = json.loads(path.read_text(encoding="utf-8"))

    assert script.main(["--runs-dir", str(runs), "--checkpoints-dir", str(checkpoints), "--write"]) == 0
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["metrics"] == before["metrics"]
    assert after["benchmark_row"] == before["benchmark_row"]
    assert after["data"]["fingerprint"]["sha256"]
    assert after["rescore"]["max_relative_difference"] <= after["rescore"]["tolerance_relative"] == 1e-5
    assert after["rescore"]["stored_metrics_unchanged"] is True


def test_rescoring_outside_tolerance_writes_nothing(tmp_path, checkpoint):
    script = _load_script()
    runs, checkpoints, path = _legacy_run(tmp_path, checkpoint, relative_error=1e-3)
    before = path.read_text(encoding="utf-8")

    assert script.main(["--runs-dir", str(runs), "--checkpoints-dir", str(checkpoints), "--write"]) == 1
    assert path.read_text(encoding="utf-8") == before


def test_rescore_comparison_is_relative():
    script = _load_script()
    stored = {"metrics": {"a": 0.25, "n": 7}, "benchmark_row": {"b": 2.0}}
    fresh = {"metrics": {"a": 0.25 * (1 + 3e-6), "n": 7}, "benchmark_row": {"b": 2.0}}
    differences = script.compare(stored, fresh)
    assert differences["metrics.a"] == pytest.approx(3e-6, rel=1e-6)
    assert differences["metrics.n"] == differences["benchmark_row.b"] == 0.0



# --------------------------------------------------------------------------
# 9. The caller's RNG survives scoring, model reconstruction included
# --------------------------------------------------------------------------

def _schnet_checkpoint(tmp_path, gold_file):
    """A real registry model, whose layers initialise randomly when it is rebuilt."""
    from ml.models.registry import build_model

    torch.manual_seed(0)
    model = ScaledEnergyModel(build_model("schnet", hidden_channels=8, num_layers=1, num_rbf=8))
    path = tmp_path / "schnet.pt"
    save_checkpoint(
        path, model, "schnet",
        data={"gold_path": str(gold_file), "molecule": "toy", "theory": "dft",
              "cutoff_radius": 5.0, "num_rbf": 16},
    )
    return path


def test_scoring_a_randomly_initialised_model_leaves_the_callers_rng_untouched(tmp_path, gold_file):
    from ml.models.registry import build_model

    path = _schnet_checkpoint(tmp_path, gold_file)
    torch.manual_seed(99)
    expected = torch.rand(3)

    # The premise: rebuilding this model draws from the global RNG.
    torch.manual_seed(99)
    build_model("schnet", hidden_channels=8, num_layers=1, num_rbf=8)
    assert not torch.equal(torch.rand(3), expected)

    torch.manual_seed(99)
    evaluate_checkpoint(path, batch_size=4)
    assert torch.equal(torch.rand(3), expected)


def test_rng_fork_covers_every_cuda_device(tmp_path, gold_file, monkeypatch):
    """torch.manual_seed reseeds all CUDA devices, so all of them must be forked, not one."""
    import contextlib

    path = _schnet_checkpoint(tmp_path, gold_file)
    forked = []
    real_fork_rng = torch.random.fork_rng

    @contextlib.contextmanager
    def spy(devices=None, **kwargs):
        forked.append(list(devices))
        with real_fork_rng(devices=[], **kwargs):  # no real GPU here
            yield

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(torch.random, "fork_rng", spy)
    evaluate_checkpoint(path, batch_size=4, device="cpu")
    assert forked == [[0, 1, 2]]


# --------------------------------------------------------------------------
# 10. Opt-in per-frame predictions export (SCRUM-50, SCRUM-63)
# --------------------------------------------------------------------------

_WALL_CLOCK = ("eval_wall_clock_s", "model_inference_s", "inference_s_per_config", "wall_clock_s_per_config")


def _without_wall_clock(result: dict) -> dict:
    """Everything but wall-clock timings, which differ between any two scorings."""
    out = {key: value for key, value in result.items() if key != "predictions"}
    out["timing"] = {k: v for k, v in result["timing"].items() if k not in _WALL_CLOCK}
    return out


def _accumulated_maes(predictions) -> tuple[float, float]:
    """MetricAccumulator's own arithmetic, on the saved arrays alone: the
    float32 difference cast to float64, summed per batch (the batch size is
    in the metadata), batch sums added as Python floats, one division."""
    batch_size = json.loads(str(predictions["metadata"]))["batch_size"]
    n_frames, n_atoms = predictions["F_ref"].shape[:2]
    e_total = f_total = 0.0
    for start in range(0, n_frames, batch_size):
        frames = slice(start, start + batch_size)
        d_e = (torch.from_numpy(predictions["E_pred"][frames]) - torch.from_numpy(predictions["E_ref"][frames])).double()
        d_f = (torch.from_numpy(predictions["F_pred"][frames]) - torch.from_numpy(predictions["F_ref"][frames])).double()
        e_total += float(d_e.abs().sum())
        f_total += float(d_f.reshape(-1, 3).abs().sum())
    return e_total / n_frames, f_total / (3 * n_frames * n_atoms)


@pytest.mark.parametrize("batch_size", [4, N_TEST_CONFIGS])  # three batches (one short), and one
def test_mae_from_the_saved_predictions_equals_the_reported_mae_exactly(checkpoint, gold_file, tmp_path,
                                                                        batch_size):
    out = tmp_path / "predictions.npz"
    result = evaluate_checkpoint(checkpoint, batch_size=batch_size, predictions_out=out)
    record = result["metrics"]["per_key"][0]
    with np.load(out, allow_pickle=False) as predictions, np.load(gold_file, allow_pickle=False) as gold:
        energy_mae, force_mae = _accumulated_maes(predictions)
        assert energy_mae == record["energy_mae_total"]
        assert force_mae == record["force_mae_component"]
        # Any other summation order agrees to float64 rounding - subtracting the
        # stored float32 values first, as the accumulator does.
        d_e = (predictions["E_pred"] - predictions["E_ref"]).astype(np.float64)
        assert np.abs(d_e).mean() == pytest.approx(record["energy_mae_total"], rel=1e-12)
        # The gold indices are the scored frames: the references are the gold file's values at them.
        assert np.array_equal(predictions["gold_index"], gold["test_idx"])
        assert np.array_equal(predictions["E_ref"], gold["E"][predictions["gold_index"]])
        assert np.array_equal(predictions["F_ref"], gold["F"][predictions["gold_index"]])
        assert predictions["F_ref"].shape == (N_TEST_CONFIGS, len(_Z), 3)
        # Deterministic: no variances to save.
        assert "E_var" not in predictions.files and "F_var" not in predictions.files


def test_without_predictions_out_nothing_is_written_and_the_result_is_unchanged(checkpoint, tmp_path,
                                                                                 monkeypatch):
    monkeypatch.chdir(tmp_path)  # a stray relative write would land here
    before = set(tmp_path.rglob("*"))
    plain = evaluate_checkpoint(checkpoint, batch_size=4)
    assert "predictions" not in plain
    assert set(tmp_path.rglob("*")) == before
    # Exporting changes nothing else either: same scores, same record.
    exported = evaluate_checkpoint(checkpoint, batch_size=4, predictions_out=tmp_path / "p.npz")
    assert _without_wall_clock(exported) == _without_wall_clock(plain)


def test_a_stochastic_model_also_saves_its_variances(mc_checkpoint, tmp_path):
    path = mc_checkpoint(MCToyModel(num_mc_samples=3))
    plain = evaluate_checkpoint(path, batch_size=4)
    out = tmp_path / "mc.npz"
    exported = evaluate_checkpoint(path, batch_size=4, predictions_out=out)
    # Collecting draws no random numbers: the scores, UQ ones included, are unchanged.
    assert _without_wall_clock(exported) == _without_wall_clock(plain)
    with np.load(out, allow_pickle=False) as predictions:
        assert predictions["E_var"].shape == predictions["E_ref"].shape == (N_TEST_CONFIGS,)
        assert predictions["F_var"].shape == predictions["F_ref"].shape
        assert json.loads(str(predictions["metadata"]))["num_mc_samples"] == 3
        # They are the variances the UQ metrics were scored on.
        std = np.sqrt(predictions["E_var"].astype(np.float64))
        assert std.mean() == pytest.approx(plain["uq_metrics"]["per_key"][0]["energy"]["sharpness_mean_std"],
                                           rel=1e-12)


def test_the_predictions_file_carries_its_provenance(checkpoint, tmp_path):
    out = tmp_path / "nested" / "preds"  # no .npz suffix: written at exactly this path
    result = evaluate_checkpoint(checkpoint, batch_size=4, seed=5, predictions_out=out)
    assert out.is_file() and not list(out.parent.glob("*.npz")) and not list(out.parent.glob("*.partial"))
    assert result["predictions"] == {"path": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    with np.load(out, allow_pickle=False) as predictions:
        meta = json.loads(str(predictions["metadata"]))
    assert meta["checkpoint"] == str(checkpoint)
    assert meta["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert meta["data_fingerprint"] == result["data"]["fingerprint"]
    assert (meta["split"], meta["seed"], meta["num_mc_samples"], meta["batch_size"]) == ("test", 5, None, 4)
    assert meta["units"] == result["metrics"]["per_key"][0]["units"]


def test_cli_save_predictions(checkpoint, tmp_path):
    results = tmp_path / "result.json"
    assert main(["--checkpoint", str(checkpoint), "--batch-size", "4", "--output", str(results)]) == 0
    assert "predictions" not in json.loads(results.read_text(encoding="utf-8"))
    assert not list(tmp_path.rglob("*.npz.partial")) and not (tmp_path / "p.npz").exists()
    assert main(["--checkpoint", str(checkpoint), "--batch-size", "4", "--output", str(results),
                 "--save-predictions", str(tmp_path / "p.npz")]) == 0
    assert json.loads(results.read_text(encoding="utf-8"))["predictions"]["path"] == str(tmp_path / "p.npz")
    assert (tmp_path / "p.npz").is_file()
