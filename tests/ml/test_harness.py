"""Tests for the shared training harness: one locked protocol for every model.

Covers ml/config.py, ml/training/train.py, ml/models/registry.py and
ml/models/scaling.py. What a pass here guarantees about the benchmark:

1. **One path.** Every config trains through registry -> ScaledEnergyModel ->
   train.py -> save_checkpoint -> evaluate_checkpoint, and is scored at
   ethanol's real absolute energy - not ~97,196 kcal/mol off (shift lost)
   or ~194,000 off (shift applied twice).
2. **One protocol.** The six configs differ only in `model:`. Their `data:`
   and `train:` blocks - gold file, molecule, theory, seed, budget,
   optimiser settings - are byte-identical.
3. **One set of frames.** All six train and select on exactly the same
   frames, spread over their whole blocks, and none is a test frame.

Real-data tests skip if data/gold/md17/ethanol_dft.npz is absent (run
`python -m ml.data.preprocessing`). The training tests use a tiny gold file
built from 80 real ethanol frames, every one from the real train block, so
no val or test frame is ever read.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import ml.training.train as train_module
from ml.config import ExperimentConfig
from ml.models.blip import BLIP
from ml.models.painn import PaiNN
from ml.models.registry import MODEL_REGISTRY, UnknownModel, load_model
from ml.models.scaling import ScaledEnergyModel
from ml.models.schnet import SchNet
from ml.training.evaluate import evaluate_checkpoint
from ml.training.train import build_model, open_split, select_subsets, stride_subset, train

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "gold" / "md17" / "ethanol_dft.npz"
CONFIGS = ROOT / "experiments" / "configs"

# config -> (registry phase, architecture class name)
EXPECTED_MODELS = {
    "phase1_baseline": ("mpnn", "MPNN"),
    "phase1_schnet": ("schnet", "SchNet"),
    "phase1_painn": ("painn", "PaiNN"),
    "phase1_physnet": ("physnet", "PhysNet"),
    "phase1_dimenet": ("dimenet", "GoldDimeNetPlusPlus"),
    "phase2_blip": ("blip", "BLIP"),
}
CONFIG_NAMES = tuple(EXPECTED_MODELS)

# The locked protocol. Changing any of these is a protocol change for every
# model at once, so it has to be made here deliberately, not drift in a config.
LOCKED_DATA = {
    "dataset": "md17",
    "molecule": "ethanol",
    "theory": "dft",
    "gold_path": "data/gold/md17/ethanol_dft.npz",
    "cutoff_radius": 5.0,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "use_literature_split": False,
    "num_rbf": 16,
    "seed": 42,
}
LOCKED_TRAIN = {
    "epochs": 200,
    "batch_size": 32,
    "lr": 0.0005,
    "energy_loss_weight": 1.0,
    "force_loss_weight": 100.0,
    "patience": 20,
    "device": "cpu",
    "train_subset": 10000,
    "train_subset_seed": 42,
    "val_subset": 5000,
}


def _load(name: str) -> ExperimentConfig:
    return ExperimentConfig.from_yaml(CONFIGS / f"{name}.yaml")


def _top_level_block(text: str, key: str) -> str:
    """The raw text of one top-level YAML block, from `key:` to the next top-level key."""
    lines = text.splitlines()
    start = lines.index(f"{key}:")
    end = next(
        (i for i in range(start + 1, len(lines)) if re.match(r"[A-Za-z_]", lines[i])), len(lines)
    )
    return "\n".join(lines[start:end]).rstrip()


# --------------------------------------------------------------------------
# 1. One protocol: the configs differ only in model:
# --------------------------------------------------------------------------

@pytest.mark.parametrize("block", ["data", "train"])
def test_locked_blocks_are_byte_identical_across_configs(block):
    texts = {
        name: _top_level_block((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"), block)
        for name in CONFIG_NAMES
    }
    for name, text in texts.items():
        assert text == texts["phase1_baseline"], (
            f"{name}.yaml's {block}: block differs from phase1_baseline.yaml's - the protocol is "
            "locked; change it in all six configs or none"
        )


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_config_resolves_to_the_locked_protocol(name):
    """Same gold_path, molecule and theory - and every other data/train value - in all six."""
    resolved = _load(name).to_dict()
    assert resolved["data"] == LOCKED_DATA
    assert resolved["train"] == LOCKED_TRAIN


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_config_builds_its_own_registered_model(name):
    cfg = _load(name)
    phase, class_name = EXPECTED_MODELS[name]
    assert cfg.model.phase == phase
    model = build_model(cfg)
    assert type(model).__name__ == class_name
    # Each model's own cutoff matches the graph radius, as train.py requires.
    assert ScaledEnergyModel(model).cutoff_radius == cfg.data.cutoff_radius


def test_the_six_configs_cover_every_registered_model():
    assert {phase for phase, _ in EXPECTED_MODELS.values()} == set(MODEL_REGISTRY)


# --------------------------------------------------------------------------
# 2. One set of frames
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gold_splits() -> dict[str, np.ndarray]:
    if not GOLD.exists():
        pytest.skip(f"{GOLD} not found; run `python -m ml.data.preprocessing`")
    with np.load(GOLD) as gold:
        return {split: gold[f"{split}_idx"] for split in ("train", "val", "test")}


@pytest.fixture(scope="module")
def subsets(gold_splits) -> dict[str, dict]:
    return {name: select_subsets(_load(name)) for name in CONFIG_NAMES}


@pytest.mark.parametrize("split", ["train", "val"])
def test_subsets_are_identical_across_configs(subsets, split):
    reference = subsets["phase1_baseline"][split]["gold_indices"]
    for name, selected in subsets.items():
        np.testing.assert_array_equal(selected[split]["gold_indices"], reference, err_msg=name)


@pytest.mark.parametrize("split, size", [("train", 10000), ("val", 5000)])
def test_subset_is_spread_over_its_whole_block_and_stays_in_it(subsets, gold_splits, split, size):
    info = subsets["phase1_baseline"][split]
    positions, stride = info["positions"], info["stride"]
    n = len(gold_splits[split])
    assert len(np.unique(positions)) == len(positions) == size
    assert np.all(np.diff(positions) == stride), "not an even stride"
    assert positions[0] < stride, "does not start in the first stride window"
    # stride = n // size, so the last n - stride * size frames (4,074 of train,
    # 509 of val) are never drawn - the same as scripts/train_painn.py. The
    # subset still spans over 99% of its block.
    assert positions[-1] >= stride * (size - 1), "does not reach the last stride window"
    assert (positions[-1] - positions[0] + stride) / n > 0.99
    assert np.isin(info["gold_indices"], gold_splits[split]).all()


def test_no_training_or_selection_frame_is_a_test_frame(subsets, gold_splits):
    for name, selected in subsets.items():
        for split in ("train", "val"):
            assert not np.isin(selected[split]["gold_indices"], gold_splits["test"]).any(), (name, split)
    reference = subsets["phase1_baseline"]
    assert not np.isin(reference["train"]["gold_indices"], reference["val"]["gold_indices"]).any()


def test_train_subset_matches_the_provisional_painn_trainer(subsets):
    """Same frames as scripts/train_painn.py: stride 44, offset 3 at seed 42 (review H6)."""
    train_info, val_info = subsets["phase1_baseline"]["train"], subsets["phase1_baseline"]["val"]
    assert (train_info["stride"], train_info["offset"]) == (44, 3)
    assert (train_info["positions"][0], train_info["positions"][-1]) == (3, 439_959)
    assert (val_info["stride"], val_info["offset"]) == (11, 0)


def test_stride_subset_edge_cases():
    positions, stride, offset = stride_subset(10, None, 0)
    assert positions.tolist() == list(range(10)) and (stride, offset) == (1, 0)
    np.testing.assert_array_equal(stride_subset(1000, 7, 42)[0], stride_subset(1000, 7, 42)[0])
    assert stride_subset(1000, 7, 42)[0].max() < 1000
    for bad in (0, 11):
        with pytest.raises(ValueError):
            stride_subset(10, bad, 0)


# --------------------------------------------------------------------------
# 3. Config schema: unknown keys fail loudly
# --------------------------------------------------------------------------

def _yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("text, match", [
    ("name: x\nmodel: {phase: mpnn}\noutput: {}\n", "unknown top-level"),
    ("name: x\nmodel: {phase: mpnn}\ndata: {num_rbfs: 3}\n", r"unknown key\(s\) \['num_rbfs'\]"),
    ("name: x\nmodel: {phase: mpnn}\ntrain: {batchsize: 3}\n", r"unknown key\(s\) \['batchsize'\]"),
    ("name: x\nmodel: {hidden_channels: 3}\n", "model.phase is required"),
    ("name: x\nmodel: {phase: mpnn}\ntrain: {train_subset: 100}\n", "train_subset_seed is required"),
    ("name: x\nmodel: {phase: mpnn}\ntrain: {val_subset: 0, train_subset_seed: 1}\n", "positive integer"),
])
def test_bad_config_fails_on_load(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        ExperimentConfig.from_yaml(_yaml(tmp_path, text))


def test_unknown_model_key_fails_when_built(tmp_path):
    cfg = ExperimentConfig.from_yaml(_yaml(tmp_path, "name: x\nmodel: {phase: schnet, hiden_channels: 8}\n"))
    with pytest.raises(TypeError, match="hiden_channels"):
        build_model(cfg)
    # DimeNet++'s constructor takes **kwargs and checks them against PyG's signature itself.
    dimenet = ("name: x\nmodel: {phase: dimenet, hidden_channels: 16, out_channels: 1, num_blocks: 1, "
               "int_emb_size: 8, basis_emb_size: 4, out_emb_channels: 16, num_spherical: 3, "
               "num_radial: 3, bogus: 1}\n")
    with pytest.raises(TypeError, match="bogus"):
        build_model(ExperimentConfig.from_yaml(_yaml(tmp_path, dimenet)))


def test_unknown_phase_fails_when_built(tmp_path):
    cfg = ExperimentConfig.from_yaml(_yaml(tmp_path, "name: x\nmodel: {phase: not_a_model}\n"))
    with pytest.raises(UnknownModel, match="registered"):
        build_model(cfg)


def test_blip_rejects_unknown_constructor_arguments():
    """BLIP used to swallow unknown kwargs (**_ignored); a typo now fails."""
    with pytest.raises(TypeError):
        BLIP(perturbation_strength=0.1)


# --------------------------------------------------------------------------
# 4. ScaledEnergyModel: E = scale * E_model + n_atoms * shift, F = scale * F_model
# --------------------------------------------------------------------------

class _FixedOutput(nn.Module):
    def __init__(self, energy: torch.Tensor, forces: torch.Tensor) -> None:
        super().__init__()
        self.energy, self.forces = energy, forces

    def config(self) -> dict:
        return {"cutoff": 4.0}

    def predict_energy_and_forces(self, z, pos, edge_index, edge_attr, batch):
        return self.energy, self.forces


def test_wrapper_applies_the_shift_per_atom_and_the_scale_to_both_outputs():
    batch = torch.tensor([0, 0, 0, 1, 1])  # molecules of 3 and 2 atoms
    inner_forces = torch.arange(15.0).view(5, 3)
    model = ScaledEnergyModel(_FixedOutput(torch.tensor([1.0, -2.0]), inner_forces), shift=-10.0, scale=0.5)
    energy, forces = model.predict_energy_and_forces(None, torch.zeros(5, 3), None, None, batch)
    assert energy.tolist() == [0.5 * 1.0 + 3 * -10.0, 0.5 * -2.0 + 2 * -10.0]
    assert torch.equal(forces, 0.5 * inner_forces)


def test_statistics_are_per_atom_float64_mean_and_unbiased_std():
    energies = torch.tensor([-97190.0, -97200.0, -97195.5], dtype=torch.float32)
    shift, scale = ScaledEnergyModel.statistics_from_energies(energies, n_atoms=9)
    reference = energies.double().numpy()
    assert shift == pytest.approx(reference.mean() / 9, abs=1e-12)
    assert scale == pytest.approx(reference.std(ddof=1) / 9, abs=1e-12)


def test_wrapper_rejects_a_non_positive_scale():
    with pytest.raises(ValueError):
        ScaledEnergyModel(_FixedOutput(torch.zeros(1), torch.zeros(1, 3)), scale=0.0)


def test_cutoff_radius_reads_either_constructor_name():
    assert ScaledEnergyModel(PaiNN(hidden_channels=8, num_layers=1, cutoff_radius=4.5)).cutoff_radius == 4.5
    assert ScaledEnergyModel(SchNet(hidden_channels=8, num_layers=1, cutoff=3.5)).cutoff_radius == 3.5


# --------------------------------------------------------------------------
# 5. One path: every config trains, saves via the registry, and is scored
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_gold(tmp_path_factory) -> SimpleNamespace:
    """80 real ethanol frames spread over the real train block, re-split 64/8/8."""
    if not GOLD.exists():
        pytest.skip(f"{GOLD} not found; run `python -m ml.data.preprocessing`")
    path = tmp_path_factory.mktemp("gold") / "ethanol_dft_tiny.npz"
    with np.load(GOLD) as gold:
        train_idx = gold["train_idx"]
        frames = train_idx[np.linspace(0, len(train_idx) - 1, 80).astype(int)]
        held_out = np.concatenate([gold["val_idx"], gold["test_idx"]])
        assert not np.isin(frames, held_out).any()
        np.savez(
            path, z=gold["z"], R=gold["R"][frames], E=gold["E"][frames], F=gold["F"][frames],
            train_idx=np.arange(64), val_idx=np.arange(64, 72), test_idx=np.arange(72, 80),
            molecule=gold["molecule"], theory=gold["theory"], used_literature_split=False,
        )
        train_mean = float(gold["E"][frames[:64]].astype(np.float64).mean())
    return SimpleNamespace(path=path, train_mean=train_mean)


def _tiny_config(name: str, gold_path: Path, **train_overrides) -> ExperimentConfig:
    """The config as written, pointed at the tiny gold file with a budget it can hold.

    64 train frames at batch 32 is two optimiser steps per epoch.
    """
    cfg = _load(name)
    budget = {"epochs": 1, "train_subset": 64, "val_subset": 8, **train_overrides}
    return dataclasses.replace(
        cfg,
        data=dataclasses.replace(cfg.data, gold_path=str(gold_path)),
        train=dataclasses.replace(cfg.train, **budget),
    )


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_config_trains_saves_and_is_scored_at_absolute_energy(name, tiny_gold, tmp_path):
    cfg = _tiny_config(name, tiny_gold.path)
    phase, class_name = EXPECTED_MODELS[name]
    run_dir = tmp_path / "runs" / cfg.name

    record = train(cfg, output_root=tmp_path)

    assert record["status"] == "complete" and record["reportable"] is True
    assert record["epochs_run"] == record["best_epoch"] == 1
    assert len(list(csv.DictReader((run_dir / "history.csv").open(encoding="utf-8")))) == 1

    # Saved through the registry: current format, the right architecture, wrapped.
    model, stored = load_model(tmp_path / "checkpoints" / f"{cfg.name}.pt")
    assert stored["checkpoint_version"] == 2 and stored["phase"] == phase
    assert isinstance(model, ScaledEnergyModel) and type(model.model).__name__ == class_name
    assert stored["train"]["complete"] is True
    # The shift is the training subset's mean energy, per atom.
    assert float(model.shift) * 9 == pytest.approx(tiny_gold.train_mean, abs=1e-2)

    # Scored by evaluate_checkpoint at ethanol's absolute energy. A lost shift
    # is ~97,196 kcal/mol off and a doubled one ~194,000; at the right scale
    # an all-but-untrained model is within a few kcal/mol (std(E) ~4.2).
    result = evaluate_checkpoint(
        tmp_path / "checkpoints" / f"{cfg.name}.pt", split="test", batch_size=cfg.train.batch_size
    )
    row = result["benchmark_row"]
    assert row["model"] == class_name and row["energy_mae_unit"] == "kcal/mol"
    assert 0 < row["energy_mae"] < 100, row["energy_mae"]
    assert result["energy_scaling"]["shift_per_atom"] == pytest.approx(float(model.shift))

    # train.py's own final scoring is that same call, saved in the run record:
    # equal up to multithreaded float32 summation order (~1e-10 relative) -
    # BLIP included, since evaluate_checkpoint scores under a fixed seed.
    saved = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    assert saved["seed"] == result["seed"]
    assert saved["benchmark_row"]["energy_mae"] == pytest.approx(row["energy_mae"], rel=1e-6)
    assert saved["benchmark_row"]["force_mae"] == pytest.approx(row["force_mae"], rel=1e-6)


def test_run_record_carries_the_provenance(tiny_gold, tmp_path):
    cfg = _tiny_config("phase1_schnet", tiny_gold.path)
    train(cfg, output_root=tmp_path, config_path=CONFIGS / "phase1_schnet.yaml")
    run_dir = tmp_path / "runs" / cfg.name
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    for key in ("git_commit", "git_dirty", "config", "config_sha256", "model_config",
                "n_parameters", "subsets", "energy_statistics", "environment", "protocol",
                "epochs_run", "best_epoch", "stop_reason", "wall_clock_s", "test"):
        assert record[key] is not None, key
    assert record["config"]["model"]["phase"] == "schnet"
    assert record["energy_statistics"]["computed_on"] == "train subset"
    assert record["n_parameters"] == sum(p.numel() for p in build_model(cfg).parameters())

    subsets = json.loads((run_dir / "subsets.json").read_text(encoding="utf-8"))
    assert len(subsets["train"]["gold_indices"]) == 64 and len(subsets["val"]["gold_indices"]) == 8
    assert subsets["train"]["gold_indices_sha256"] == record["subsets"]["train"]["gold_indices_sha256"]


def test_training_opens_train_and_val_only(tiny_gold, tmp_path, monkeypatch):
    """The test split is constructed by evaluate.py alone - never on the training path."""
    opened = []
    original = train_module.MD17Dataset

    class Spy(original):
        def __init__(self, *args, split="train", **kwargs):
            opened.append(split)
            super().__init__(*args, split=split, **kwargs)

    monkeypatch.setattr(train_module, "MD17Dataset", Spy)
    record = train(_tiny_config("phase1_schnet", tiny_gold.path), output_root=tmp_path)
    assert "test" in record  # scored - through evaluate.py's own dataset
    assert sorted(opened) == ["train", "val"]
    with pytest.raises(ValueError, match="refusing"):
        open_split(_tiny_config("phase1_schnet", tiny_gold.path), "test")


def test_development_run_is_not_reportable_and_never_scores(tiny_gold, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(train_module, "evaluate_checkpoint", lambda *a, **k: calls.append(a))
    cfg = _tiny_config("phase1_schnet", tiny_gold.path)
    record = train(cfg, output_root=tmp_path, epochs=1, max_batches=1, score_test=False)
    assert record["reportable"] is False
    assert record["overrides"] == {"epochs": 1, "max_batches": 1}
    assert calls == [] and not (tmp_path / "runs" / cfg.name / "test_metrics.json").exists()


def test_a_finished_run_is_not_overwritten(tiny_gold, tmp_path):
    cfg = _tiny_config("phase1_schnet", tiny_gold.path)
    train(cfg, output_root=tmp_path, max_batches=1, score_test=False)
    with pytest.raises(FileExistsError, match="--overwrite"):
        train(cfg, output_root=tmp_path, max_batches=1, score_test=False)
    train(cfg, output_root=tmp_path, max_batches=1, score_test=False, overwrite=True)


def test_model_cutoff_must_match_the_graph(tiny_gold, tmp_path):
    cfg = _tiny_config("phase1_painn", tiny_gold.path)
    cfg = dataclasses.replace(
        cfg, model=dataclasses.replace(cfg.model, kwargs={**cfg.model.kwargs, "cutoff_radius": 4.0})
    )
    with pytest.raises(ValueError, match="cutoff"):
        train(cfg, output_root=tmp_path)
