"""Interface and convention checks for PaiNN against the bake-off (SCRUM-52 review).

`test_painn.py` and `test_painn_physics.py` cover physics. This file
covers the conventions a model has to share with the other four entries
for the comparison to be like-for-like:

- the architecture built from `phase1_painn.yaml` is the one the config
  describes;
- outputs satisfy `ml/utils/contract.py` at training precision (float32)
  on a real `MD17Dataset` batch;
- the per-atom `energy_shift` / `energy_scale` buffers mean exactly what
  their docstrings say, and survive the registry -> `evaluate.py` round
  trip;
- the shared 4-argument `predict_energy` call fails loudly rather than
  silently misreading its arguments;
- the fixed experimental parameters in `phase1_painn.yaml` match the other
  Phase 1 configs.

Shared-harness gaps found during the review (train.py / config.py) are
deliberately *not* encoded here; they are reported separately.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from ml.data.md17 import build_graph
from ml.models.painn import PaiNN

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "gold" / "md17" / "ethanol_dft.npz"
CONFIGS = ROOT / "experiments" / "configs"
OTHER_PHASE1_CONFIGS = ("phase1_baseline", "phase1_schnet", "phase1_physnet")

_Z = [6, 6, 8, 1, 1, 1, 1, 1, 1]
ETHANOL_MEAN = -97195.89  # kcal/mol, train split


def _config(name: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))


def _painn_from_config() -> PaiNN:
    cfg = _config("phase1_painn")["model"]
    torch.manual_seed(0)
    return PaiNN(
        hidden_channels=cfg["hidden_channels"],
        num_layers=cfg["num_layers"],
        num_rbf=cfg["num_rbf"],
        cutoff_radius=cfg["cutoff_radius"],
        max_atomic_number=cfg["max_atomic_number"],
        eps=float(cfg["eps"]),
    )


def _molecule(seed: int = 0, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    pos = torch.randn(len(_Z), 3, generator=generator, dtype=dtype) * 1.4
    edge_index, edge_attr = build_graph(pos.numpy(), cutoff=5.0, num_rbf=16)
    z = torch.tensor(_Z, dtype=torch.int64)
    return z, pos, edge_index, edge_attr.to(dtype), torch.zeros(len(_Z), dtype=torch.int64)


# --------------------------------------------------------------------------
# The config builds the architecture it describes
# --------------------------------------------------------------------------

def test_config_builds_the_described_architecture():
    """3 message+update blocks, F = 128, 20-function Bessel basis, cutoff = data.cutoff_radius."""
    config = _config("phase1_painn")
    model = _painn_from_config()

    assert len(model.message_blocks) == len(model.update_blocks) == config["model"]["num_layers"] == 3
    assert model.embedding.embedding_dim == config["model"]["hidden_channels"] == 128
    assert model.radial_basis.freqs.numel() == config["model"]["num_rbf"] == 20
    for block in model.message_blocks:
        assert block.filter_net.in_features == 20
    assert model.cutoff_radius == config["data"]["cutoff_radius"] == config["model"]["cutoff_radius"]
    # Vector-channel maps act on the channel axis only and carry no bias.
    for block in model.update_blocks:
        assert block.U.bias is None and block.V.bias is None
        assert block.U.weight.shape == block.V.weight.shape == (128, 128)
    # The readout sees the F scalar channels only.
    assert model.readout[0].in_features == 128


# --------------------------------------------------------------------------
# Output contract at training precision, on a real loader batch
# --------------------------------------------------------------------------

def test_real_float32_batch_satisfies_the_metric_contract():
    """Energy (B,), forces (N, 3), float32, finite - accepted by `contract.py` unmodified."""
    if not GOLD.exists():
        pytest.skip(f"{GOLD} not found; run `python -m ml.data.preprocessing`")
    from torch_geometric.loader import DataLoader

    from ml.data.datasets import MD17Dataset
    from ml.utils.contract import ModelOutput, ReferenceData, require_compatible

    dataset = MD17Dataset(GOLD, split="train", cutoff_radius=5.0, num_rbf=16)
    subset = torch.utils.data.Subset(dataset, range(5))
    batch = next(iter(DataLoader(subset, batch_size=5, shuffle=False)))

    model = _painn_from_config()
    model.set_energy_statistics(ETHANOL_MEAN / 9, 4.2 / 9)
    model.eval()
    energy, forces = model.predict_energy_and_forces(
        batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch
    )

    assert energy.shape == (5,) and energy.dtype == torch.float32
    assert forces.shape == (45, 3) and forces.dtype == torch.float32
    require_compatible(
        ModelOutput(E=energy.detach(), F=forces.detach()),
        ReferenceData(
            E=batch.y, F=batch.force, batch=batch.batch,
            molecule=dataset.molecule, theory=dataset.theory, z=batch.x.view(-1),
        ),
    )
    # Absolute energy, in the data's own unit and scale: within a few kcal/mol
    # of the reference before any training, because the shift is per-atom.
    assert (energy - batch.y).abs().max() < 50


def test_output_dtype_follows_the_model():
    model = PaiNN(hidden_channels=16, num_layers=1, num_rbf=8).double().eval()
    energy, forces = model.predict_energy_and_forces(*_molecule())
    assert energy.dtype == forces.dtype == torch.float64


# --------------------------------------------------------------------------
# Energy statistics: what the buffers mean
# --------------------------------------------------------------------------

def _zero_readout(model: PaiNN, bias: float = 0.0) -> None:
    with torch.no_grad():
        model.readout[-1].weight.zero_()
        model.readout[-1].bias.fill_(bias)


def test_energy_shift_is_per_atom_not_per_molecule():
    """With the readout silenced, E = N * shift. Passing a *total* energy mean would be off by (N - 1) x.

    This is the convention a training loop must follow when it calls
    `set_energy_statistics`: `scripts/smoke_train_painn.py` divides by N;
    `ml/training/train.py` currently does not call it at all.
    """
    model = PaiNN(hidden_channels=16, num_layers=1, num_rbf=8).double().eval()
    _zero_readout(model)
    model.set_energy_statistics(shift=-10.0, scale=3.0)
    energy = model.predict_energy(*_molecule())
    assert energy.item() == pytest.approx(len(_Z) * -10.0, abs=1e-12)

    _zero_readout(model, bias=1.0)
    energy = model.predict_energy(*_molecule())
    assert energy.item() == pytest.approx(len(_Z) * (1.0 * 3.0 - 10.0), abs=1e-12)


def test_energy_scale_multiplies_forces_and_shift_does_not_touch_them():
    """F = -scale * d(sum readout)/dR: a training loop that changes `scale` rescales the force target fit."""
    z, pos, edge_index, edge_attr, batch = _molecule()
    model = PaiNN(hidden_channels=16, num_layers=1, num_rbf=8).double().eval()

    model.set_energy_statistics(shift=0.0, scale=1.0)
    _, forces_unit = model.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)
    model.set_energy_statistics(shift=-1234.5, scale=2.5)
    _, forces_scaled = model.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)

    assert torch.allclose(forces_scaled, 2.5 * forces_unit, atol=1e-12, rtol=0)


def test_energy_statistics_survive_registry_and_evaluate(tmp_path):
    """Buffers travel in the checkpoint; `evaluate.py` scores the shifted model, not an unshifted one."""
    from torch_geometric.loader import DataLoader

    from ml.data.datasets import MD17Dataset
    from ml.models.registry import load_model, save_checkpoint
    from ml.training.evaluate import evaluate_checkpoint

    rng = np.random.default_rng(0)
    n_configs = 12
    gold = tmp_path / "toy_dft.npz"
    np.savez_compressed(
        gold,
        z=np.array(_Z, dtype=np.int64),
        R=(rng.normal(size=(n_configs, len(_Z), 3)) * 1.4).astype(np.float32),
        E=(ETHANOL_MEAN + rng.normal(size=n_configs)).astype(np.float32),
        F=rng.normal(size=(n_configs, len(_Z), 3)).astype(np.float32),
        train_idx=np.arange(0, 8), val_idx=np.arange(8, 10), test_idx=np.arange(10, 12),
        molecule="toy", theory="dft", used_literature_split=False,
    )

    torch.manual_seed(0)
    model = PaiNN(hidden_channels=16, num_layers=2, num_rbf=8)
    model.set_energy_statistics(ETHANOL_MEAN / len(_Z), 0.5)
    checkpoint = save_checkpoint(
        tmp_path / "painn.pt", model, "painn",
        data={"gold_path": str(gold), "molecule": "toy", "theory": "dft",
              "cutoff_radius": 5.0, "num_rbf": 16},
    )

    restored, _ = load_model(checkpoint)
    assert torch.equal(restored.energy_shift, model.energy_shift)
    assert torch.equal(restored.energy_scale, model.energy_scale)

    result = evaluate_checkpoint(checkpoint, split="test", batch_size=1)
    record = result["metrics"]["per_key"][0]

    model.eval()
    batch = next(iter(DataLoader(MD17Dataset(gold, split="test"), batch_size=2)))
    energy, forces = model.predict_energy_and_forces(
        batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch
    )
    expected_energy_mae = (energy.double() - batch.y.double()).abs().mean().item()
    expected_force_mae = (forces.double() - batch.force.double()).abs().mean().item()

    assert record["energy_mae_total"] == pytest.approx(expected_energy_mae, abs=1e-3)
    assert record["force_mae_component"] == pytest.approx(expected_force_mae, rel=1e-5)
    # A lost shift would put this near 97,000 kcal/mol.
    assert record["energy_mae_total"] < 100


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------

def test_shared_four_argument_call_fails_loudly():
    """PaiNN's `predict_energy` takes `pos`; the ticket's 4-arg form must raise, not misread.

    Positionally, the 4-arg call would hand `edge_index` in as `pos`. If
    that were ever accepted, the energy would be computed from integer
    atom indices treated as coordinates.
    """
    model = PaiNN(hidden_channels=16, num_layers=1, num_rbf=8).double().eval()
    z, _, edge_index, edge_attr, batch = _molecule()
    with pytest.raises((TypeError, ValueError)):
        model.predict_energy(z, edge_index, edge_attr, batch)
    with pytest.raises((TypeError, ValueError)):
        model.predict_energy(z, edge_index, edge_attr, batch=batch)


# --------------------------------------------------------------------------
# Fixed experimental parameters match the other Phase 1 configs
# --------------------------------------------------------------------------

@pytest.mark.parametrize("other", OTHER_PHASE1_CONFIGS)
def test_data_section_matches_the_other_phase1_configs(other):
    """Every data key the two configs share has the same value (molecule, theory, splits, cutoff, seed)."""
    painn, reference = _config("phase1_painn")["data"], _config(other)["data"]
    shared = set(painn) & set(reference)
    assert {"molecule", "theory", "gold_path", "cutoff_radius", "seed"} <= shared
    assert {key: painn[key] for key in shared} == {key: reference[key] for key in shared}


@pytest.mark.parametrize("other", OTHER_PHASE1_CONFIGS)
def test_loss_weights_match_the_other_phase1_configs(other):
    painn, reference = _config("phase1_painn")["train"], _config(other)["train"]
    assert painn["energy_loss_weight"] == reference["energy_loss_weight"] == 1.0
    assert painn["force_loss_weight"] == reference["force_loss_weight"] == 100.0
