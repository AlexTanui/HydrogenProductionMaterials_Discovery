"""Physics and symmetry tests for every harness model on real ethanol frames.

Written for PaiNN in the SCRUM-52 review; now run on all six models the
shared harness trains - mpnn, schnet, painn, physnet, dimenet and blip -
each built from its own experiment config through ml/models/registry.py and
wrapped in the harness's `ScaledEnergyModel`, exactly as train.py builds
it, so energies sit at ethanol's real ~-97,196 kcal/mol.

`tests/ml/test_painn.py` pins PaiNN's symmetries on random synthetic
molecules. This file asks the same questions of the inputs the models are
actually trained on: real MD17 ethanol geometries from the TRAIN split,
graphs from the production `build_graph`, and batches collated by the real
PyG loader.

Two models need a test-only accommodation, neither of which touches the model:

- **DimeNet++** is built with `output_initializer="glorot_orthogonal"`.
  PyG's default zero-initialises the output layers, so an untrained model
  returns E = 0 and F = 0 exactly and every force check would pass
  vacuously (tests/test_dimenet.py does the same).
- **BLIP** draws fresh weight noise on every forward pass, so the global
  RNG is reseeded before each one and every pass draws the same weights.
  Its batching check is `xfail(strict=True)`: BLIP's noise scale averages
  over the whole batch, so a molecule's prediction depends on its
  batch-mates (docs/reviews/blip-batch-dependence.md). Strict, so the
  marker has to come off once that is fixed.

Three things make a pass here evidence rather than a rubber stamp:

1. **Positive controls.** Every check runs on all six models. A check that
   failed on all of them would point at the check, not at the models.
2. **Negative controls.** Each check is also run against a deliberately
   broken mutant and must *fail* there by at least `MARGIN` times its
   tolerance. A check that cannot fail proves nothing.
3. **The graph is rebuilt from the transformed coordinates** wherever the
   data pipeline would rebuild it - after a rotation, translation,
   permutation, or finite-difference displacement. Reusing the unperturbed
   `edge_attr` would hide exactly the bug check 7 exists for: a model that
   reads the dataset's detached `edge_attr` passes every symmetry test and
   still reports forces that are not the gradient of its energy
   (`test_stale_edge_attr_mutant_passes_every_symmetry_check` shows this).

Everything runs in float64. Each tolerance is defined once below with its
reasoning.

Skipped if `data/gold/md17/ethanol_dft.npz` is absent - run
`python -m ml.data.preprocessing`. Each trained variant (`<phase>_trained`)
runs once `experiments/checkpoints/<config name>.pt` exists in the current
checkpoint format, and is skipped until then.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, NamedTuple

import numpy as np
import pytest
import torch
import yaml
from torch import Tensor, nn

from ml.config import ExperimentConfig
from ml.data.md17 import _gaussian_rbf, build_graph
from ml.models.painn import PaiNN
from ml.models.registry import build_model
from ml.models.scaling import ScaledEnergyModel
from ml.models.schnet import SchNet

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "gold" / "md17" / "ethanol_dft.npz"
CONFIGS = ROOT / "experiments" / "configs"
CHECKPOINTS = ROOT / "experiments" / "checkpoints"
CUTOFF = 5.0

# registry phase -> the experiment config that trains it.
HARNESS_CONFIGS = {
    "mpnn": "phase1_baseline",
    "schnet": "phase1_schnet",
    "painn": "phase1_painn",
    "physnet": "phase1_physnet",
    "dimenet": "phase1_dimenet",
    "blip": "phase2_blip",
}
# Test-only constructor overrides; see the module docstring.
TEST_OVERRIDES = {"dimenet": {"output_initializer": "glorot_orthogonal"}}
STOCHASTIC = {"blip"}
NOISE_SEED = 1234
BATCH_COUPLING_REASON = (
    "BLIP's noise scale averages over the whole batch (blip.py:123), so a molecule's E/F "
    "depend on its batch-mates - docs/reviews/blip-batch-dependence.md"
)
DATASET_NUM_RBF = 16  # MD17Dataset's default edge_attr width

# Positions *within* train_idx: both ends of the block and two interior
# points, so the frames span the trajectory instead of being near-duplicates.
TRAIN_POSITIONS = (0, 148_024, 296_049, 444_073)

# |E| is ~9.7e4 kcal/mol for PaiNN and PhysNet (absolute energy), where one
# float64 ulp is ~1.5e-11. Roundoff through the network stays well below
# 1e-9; the mutants below break symmetries at 1e-3 or more.
TOL_E = 1e-9  # kcal/mol
TOL_F = 1e-9  # kcal/mol/A

# Central differences: truncation is O(h^2 * F''') and roundoff is about
# ulp(E) / h. For the absolute-energy models (|E| ~ 9.7e4) the two balance
# near h = 1e-4 A; a scan over h = 1e-3..1e-5 gave a worst relative error of
# 6.5e-6 (PaiNN) and 1.1e-5 (PhysNet) there. The error is measured relative
# to max |F_fd| because untrained forces are small (~0.1 kcal/mol/A), so an
# absolute tolerance would be loose. The stale-edge_attr mutant scores ~26.
FD_STEP = 1e-4  # A
FD_TOL = 1e-4  # max |F - F_fd| / max |F_fd|, over every force component

# A broken mutant must miss the tolerance by at least this factor, or the
# negative control is not discriminating.
MARGIN = 1e3


# --------------------------------------------------------------------------
# Data: real ethanol frames, TRAIN split only
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ethanol() -> SimpleNamespace:
    if not GOLD.exists():
        pytest.skip(f"{GOLD} not found; run `python -m ml.data.preprocessing`")
    with np.load(GOLD) as data:
        train_idx = data["train_idx"]
        gold_indices = train_idx[list(TRAIN_POSITIONS)]
        coordinates = data["R"]
        train_energy = data["E"][train_idx].astype(np.float64)
        return SimpleNamespace(
            z=torch.as_tensor(data["z"], dtype=torch.int64),
            frames=[torch.as_tensor(coordinates[i], dtype=torch.float64) for i in gold_indices],
            gold_indices=gold_indices,
            held_out=np.concatenate([data["val_idx"], data["test_idx"]]),
            energy_mean=float(train_energy.mean()),
            energy_std=float(train_energy.std()),
        )


@pytest.fixture(scope="module")
def train_subset(ethanol):
    """The same four frames, served by the real `MD17Dataset` (train split)."""
    from ml.data.datasets import MD17Dataset

    dataset = MD17Dataset(GOLD, split="train", cutoff_radius=CUTOFF, num_rbf=DATASET_NUM_RBF)
    for position, frame in zip(TRAIN_POSITIONS, ethanol.frames):
        assert torch.equal(dataset.R[position].double(), frame)
    return torch.utils.data.Subset(dataset, list(TRAIN_POSITIONS))


def test_frames_come_from_the_train_split(ethanol):
    assert not np.isin(ethanol.gold_indices, ethanol.held_out).any()


# --------------------------------------------------------------------------
# Graphs and the shared call
# --------------------------------------------------------------------------

@dataclass
class Graph:
    z: Tensor
    pos: Tensor
    edge_index: Tensor
    edge_attr: Tensor | None
    batch: Tensor


def graph_for(z: Tensor, pos: Tensor, num_rbf: int = DATASET_NUM_RBF) -> Graph:
    """The graph `MD17Dataset` would build for `pos`, with `edge_attr` in float64.

    `build_graph` decides the edge set (production code and cutoff). The
    Gaussian expansion is then redone in float64 from the same coordinates
    with the production `_gaussian_rbf`, so a model that reads `edge_attr`
    sees a value consistent with `pos` to float64 precision - which finite
    differences at h = 1e-5 need. It is detached from `pos`, as the
    dataset's is.
    """
    edge_index, _ = build_graph(pos.detach().numpy(), cutoff=CUTOFF, num_rbf=num_rbf)
    j, i = edge_index
    distance = (pos[j] - pos[i]).detach().norm(dim=-1)
    edge_attr = _gaussian_rbf(distance, num_rbf=num_rbf, cutoff=CUTOFF).double()
    return Graph(z, pos, edge_index, edge_attr, torch.zeros(pos.shape[0], dtype=torch.int64))


def graph_from_loader(batch) -> Graph:
    return Graph(batch.x, batch.pos.double(), batch.edge_index, batch.edge_attr.double(), batch.batch)


def predict(model: nn.Module, graph: Graph) -> tuple[Tensor, Tensor]:
    """(E, F) through the shared entry point, on a fresh copy of `pos`.

    The copy matters: SchNet and MPNN call `pos.requires_grad_(True)` on the
    tensor they are handed, which would leak grad-tracking into the caller's
    tensor and every transform applied to it afterwards.
    """
    energy, forces = model.predict_energy_and_forces(
        graph.z, graph.pos.detach().clone(), graph.edge_index, graph.edge_attr, graph.batch
    )
    return energy.detach(), forces.detach()


# --------------------------------------------------------------------------
# Models under test, each built from its own experiment config
# --------------------------------------------------------------------------

def _model_config(name: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))["model"]


def build_painn(ethanol, **overrides) -> PaiNN:
    cfg = _model_config("phase1_painn")
    params = dict(
        hidden_channels=cfg["hidden_channels"],
        num_layers=cfg["num_layers"],
        num_rbf=cfg["num_rbf"],
        cutoff_radius=cfg["cutoff_radius"],
        max_atomic_number=cfg["max_atomic_number"],
        eps=float(cfg["eps"]),
    )
    params.update(overrides)
    torch.manual_seed(0)
    model = PaiNN(**params)
    # The per-atom statistics scripts/smoke_train_painn.py sets, so energies
    # sit at ethanol's real ~-97,196 kcal/mol and tolerances face real magnitudes.
    n_atoms = ethanol.z.numel()
    model.set_energy_statistics(ethanol.energy_mean / n_atoms, ethanol.energy_std / n_atoms)
    return model.double().eval()


def _bare_schnet() -> SchNet:
    """SchNet without the harness wrapper, for the energy-only mutant below."""
    cfg = _model_config("phase1_schnet")
    torch.manual_seed(0)
    model = SchNet(cfg["hidden_channels"], cfg["num_layers"], cfg["num_rbf"], cutoff=CUTOFF)
    return model.double().eval()


class _ReseededNoise(nn.Module):
    """Reseeds the global RNG before every forward pass of a stochastic model.

    BLIP samples fresh weights on each call. With the seed fixed, every call
    draws the same weights, so its energy is one deterministic function of
    the geometry and each check compares like with like.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def predict_energy_and_forces(self, z, pos, edge_index, edge_attr, batch):
        torch.manual_seed(NOISE_SEED)
        return self.model.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)


def _for_testing(phase: str, model: nn.Module) -> nn.Module:
    model = model.double().eval()
    return _ReseededNoise(model).eval() if phase in STOCHASTIC else model


def _harness_model(phase: str, ethanol) -> nn.Module:
    """Built the way train.py builds it: the registered model from its own
    config, wrapped in ScaledEnergyModel with per-atom train-split statistics."""
    cfg = ExperimentConfig.from_yaml(CONFIGS / f"{HARNESS_CONFIGS[phase]}.yaml")
    torch.manual_seed(0)
    inner = build_model(phase, **{**cfg.model.kwargs, **TEST_OVERRIDES.get(phase, {})})
    n_atoms = ethanol.z.numel()
    wrapped = ScaledEnergyModel(
        inner, shift=ethanol.energy_mean / n_atoms, scale=ethanol.energy_std / n_atoms
    )
    return _for_testing(phase, wrapped)


def _trained_model(phase: str, ethanol) -> nn.Module:
    from ml.models.registry import CHECKPOINT_VERSION, load_model

    path = CHECKPOINTS / f"{HARNESS_CONFIGS[phase]}.pt"
    if not path.exists():
        pytest.skip(f"no trained checkpoint at {path}")
    version = torch.load(path, map_location="cpu", weights_only=False).get("checkpoint_version", 1)
    if version != CHECKPOINT_VERSION:
        pytest.skip(f"{path.name} is checkpoint version {version}, from a superseded trainer")
    model, _ = load_model(path)
    return _for_testing(phase, model)


BUILDERS: dict[str, Callable[[SimpleNamespace], nn.Module]] = {}
for _phase in HARNESS_CONFIGS:
    BUILDERS[_phase] = partial(_harness_model, _phase)
    BUILDERS[f"{_phase}_trained"] = partial(_trained_model, _phase)


@pytest.fixture(scope="module", params=list(BUILDERS))
def model(request, ethanol) -> nn.Module:
    return BUILDERS[request.param](ethanol)


# --------------------------------------------------------------------------
# Negative controls: each breaks exactly one property
# --------------------------------------------------------------------------

class _EnergyFunctionModel(nn.Module):
    """Wraps an arbitrary energy function in the shared interface, forces by autograd."""

    def __init__(self, energy_fn: Callable[..., Tensor], *parts: nn.Module) -> None:
        super().__init__()
        self.parts = nn.ModuleList(parts)
        self.energy_fn = energy_fn

    def predict_energy_and_forces(self, z, pos, edge_index, edge_attr, batch):
        with torch.enable_grad():
            pos = pos.detach().requires_grad_(True)
            energy = self.energy_fn(z, pos, edge_index, edge_attr, batch)
            (grad,) = torch.autograd.grad(energy.sum(), pos, allow_unused=True)
        if grad is None:  # the energy never touched pos
            grad = torch.zeros_like(pos)
        return energy.detach(), -grad


class _DetachedRadialBasis(nn.Module):
    """Hands back the dataset's precomputed `edge_attr` instead of a basis built from `dist`."""

    def __init__(self) -> None:
        super().__init__()
        self.edge_attr: Tensor | None = None

    def forward(self, dist: Tensor) -> tuple[Tensor, Tensor]:
        return self.edge_attr, torch.ones_like(dist).unsqueeze(-1)


class _StaleRadialPaiNN(PaiNN):
    """PaiNN whose radial filter reads the detached `edge_attr`; directions still come from `pos`.

    The exact bug `painn.py::predict_energy` warns about: forces are
    non-zero, rotation-equivariant and plausible, and miss the
    dE/drbf * drbf/dpos term entirely.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.radial_basis = _DetachedRadialBasis()

    def predict_energy(self, z, pos, edge_index, edge_attr, batch):
        self.radial_basis.edge_attr = edge_attr
        return super().predict_energy(z, pos, edge_index, edge_attr, batch)


def _vector_bias_mutant(ethanol) -> PaiNN:
    """`bias=True` on the U map over vector channels: adds a fixed direction."""
    model = build_painn(ethanol)
    torch.manual_seed(1)
    for block in model.update_blocks:
        biased = nn.Linear(model.hidden_channels, model.hidden_channels, bias=True).double()
        with torch.no_grad():
            biased.weight.copy_(block.U.weight)
            biased.bias.normal_(std=0.1)
        block.U = biased
    return model.eval()


def _absolute_position_mutant(ethanol) -> nn.Module:
    """Adds a term linear in absolute coordinates: translation-variant, net force != 0."""
    painn = build_painn(ethanol)
    direction = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64) / 14.0**0.5

    def energy(z, pos, edge_index, edge_attr, batch):
        per_atom = 0.1 * (pos @ direction)
        anchor = torch.zeros(int(batch.max()) + 1, dtype=pos.dtype).index_add(0, batch, per_atom)
        return painn.predict_energy(z, pos, edge_index, edge_attr, batch) + anchor

    return _EnergyFunctionModel(energy, painn).eval()


def _index_dependent_mutant(ethanol) -> nn.Module:
    """Weights atoms by their position in the array: rotation/translation-invariant, order-dependent."""
    painn = build_painn(ethanol)

    def energy(z, pos, edge_index, edge_attr, batch):
        centred = pos - pos.mean(dim=0)
        weights = torch.arange(1, pos.shape[0] + 1, dtype=pos.dtype)
        term = 0.1 * (weights * centred.pow(2).sum(dim=-1)).sum()
        return painn.predict_energy(z, pos, edge_index, edge_attr, batch) + term

    return _EnergyFunctionModel(energy, painn).eval()


def _stale_radial_mutant(ethanol) -> PaiNN:
    model = _StaleRadialPaiNN(
        hidden_channels=128, num_layers=3, num_rbf=DATASET_NUM_RBF, cutoff_radius=CUTOFF
    )
    n_atoms = ethanol.z.numel()
    model.set_energy_statistics(ethanol.energy_mean / n_atoms, ethanol.energy_std / n_atoms)
    return model.double().eval()


def _stale_energy_only_mutant(ethanol) -> nn.Module:
    """The MPNN-prototype bug: energy from the precomputed `edge_attr`, forces by autograd -> zeros."""
    schnet = _bare_schnet()

    def energy(z, pos, edge_index, edge_attr, batch):
        return schnet.predict_energy(z, edge_index, edge_attr, batch)

    return _EnergyFunctionModel(energy, schnet).eval()


def _batch_blind_mutant(ethanol) -> nn.Module:
    """Pools every atom in the batch into one energy and hands it to every graph."""
    painn = build_painn(ethanol)

    def energy(z, pos, edge_index, edge_attr, batch):
        total = painn.predict_energy(z, pos, edge_index, edge_attr, torch.zeros_like(batch))
        return total.expand(int(batch.max()) + 1)

    return _EnergyFunctionModel(energy, painn).eval()


@pytest.fixture(scope="module")
def mutants(ethanol) -> dict[str, nn.Module]:
    return {
        "vector_bias": _vector_bias_mutant(ethanol),
        "absolute_position": _absolute_position_mutant(ethanol),
        "index_dependent": _index_dependent_mutant(ethanol),
        "stale_radial": _stale_radial_mutant(ethanol),
        "stale_energy_only": _stale_energy_only_mutant(ethanol),
        "batch_blind": _batch_blind_mutant(ethanol),
    }


# --------------------------------------------------------------------------
# Checks: each returns the worst error over the four frames
# --------------------------------------------------------------------------

class Transform(NamedTuple):
    coordinates: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]  # (z, pos) -> (z', pos')
    forces: Callable[[Tensor], Tensor]  # F(x) -> expected F(x')


def _random_rotation(seed: int) -> Tensor:
    """A uniformly random proper rotation (det = +1)."""
    generator = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _orthogonal(q: Tensor) -> Transform:
    # `pos` holds row vectors, so x -> Q x for every atom is `pos @ Q.T`.
    return Transform(lambda z, pos: (z, pos @ q.T), lambda forces: forces @ q.T)


ROTATIONS = {
    "rotation_a": _orthogonal(_random_rotation(7)),
    "rotation_b": _orthogonal(_random_rotation(8)),
    "rotation_c": _orthogonal(_random_rotation(9)),
    # det = -1. None of these models builds a pseudo-vector (no cross
    # products), so they are E(3)-invariant and a reflection must pass too.
    "improper_rotation": _orthogonal(
        torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)) @ _random_rotation(10)
    ),
}

# Large relative to the 5 A cutoff, so float64 slack cannot absorb it.
TRANSLATION = Transform(
    lambda z, pos: (z, pos + torch.tensor([13.5, -7.25, 91.0], dtype=torch.float64)),
    lambda forces: forces,
)

# New atom k is old atom PERM[k]. Mixes elements (z is permuted too).
PERM = torch.tensor([4, 0, 7, 2, 8, 1, 6, 3, 5])
PERMUTATION = Transform(lambda z, pos: (z[PERM], pos[PERM]), lambda forces: forces[PERM])


def symmetry_error(model: nn.Module, ethanol, transform: Transform) -> tuple[float, float]:
    """Worst (|dE|, |dF|) after transforming each frame and rebuilding its graph."""
    worst_energy = worst_force = 0.0
    for pos in ethanol.frames:
        energy, forces = predict(model, graph_for(ethanol.z, pos))
        z_new, pos_new = transform.coordinates(ethanol.z, pos)
        energy_new, forces_new = predict(model, graph_for(z_new, pos_new))
        worst_energy = max(worst_energy, (energy_new - energy).abs().max().item())
        worst_force = max(worst_force, (forces_new - transform.forces(forces)).abs().max().item())
    return worst_energy, worst_force


def finite_difference_error(model: nn.Module, z: Tensor, pos: Tensor, forces: Tensor) -> float:
    """max |F - F_fd| / max |F_fd|, with the graph and `edge_attr` rebuilt at every displaced geometry.

    Normalised by the finite-difference forces, not the model's, so a model
    returning zero forces scores 1.0 rather than dividing by zero.
    """
    numerical = torch.zeros_like(pos)
    for atom in range(pos.shape[0]):
        for axis in range(3):
            plus, minus = pos.clone(), pos.clone()
            plus[atom, axis] += FD_STEP
            minus[atom, axis] -= FD_STEP
            energy_plus = predict(model, graph_for(z, plus))[0]
            energy_minus = predict(model, graph_for(z, minus))[0]
            numerical[atom, axis] = -(energy_plus - energy_minus).item() / (2 * FD_STEP)
    return ((forces - numerical).abs().max() / numerical.abs().max()).item()


def worst_finite_difference_error(model: nn.Module, ethanol) -> float:
    errors = []
    for pos in ethanol.frames:
        _, forces = predict(model, graph_for(ethanol.z, pos))
        errors.append(finite_difference_error(model, ethanol.z, pos, forces))
    return max(errors)


def worst_net_force(model: nn.Module, ethanol) -> float:
    return max(
        predict(model, graph_for(ethanol.z, pos))[1].sum(dim=0).abs().max().item()
        for pos in ethanol.frames
    )


def batch_error(model: nn.Module, subset, batch_size: int) -> tuple[float, float, float]:
    """Worst (|dE|, |dF|) between batched and one-at-a-time passes, plus the energy spread."""
    from torch_geometric.loader import DataLoader

    def run(size: int) -> tuple[Tensor, Tensor]:
        outputs = [predict(model, graph_from_loader(b)) for b in DataLoader(subset, batch_size=size)]
        return torch.cat([e for e, _ in outputs]), torch.cat([f for _, f in outputs])

    single_energy, single_forces = run(1)
    batched_energy, batched_forces = run(batch_size)
    assert batched_energy.shape == single_energy.shape == (len(subset),)
    return (
        (batched_energy - single_energy).abs().max().item(),
        (batched_forces - single_forces).abs().max().item(),
        (single_energy.max() - single_energy.min()).item(),
    )


def edge_attr_sensitivity(model: nn.Module, ethanol) -> float:
    """Largest change in E or F when `edge_attr` is zeroed, replaced by noise, or omitted."""
    worst = 0.0
    for pos in ethanol.frames:
        graph = graph_for(ethanol.z, pos)
        energy, forces = predict(model, graph)
        generator = torch.Generator().manual_seed(0)
        noise = torch.randn(graph.edge_attr.shape, generator=generator, dtype=torch.float64) * 100
        for corrupted in (torch.zeros_like(graph.edge_attr), noise, None):
            try:
                energy_c, forces_c = predict(model, Graph(graph.z, pos, graph.edge_index, corrupted, graph.batch))
            except (TypeError, RuntimeError, AttributeError):
                return float("inf")  # cannot run without edge_attr, so it depends on it
            worst = max(worst, (energy_c - energy).abs().max().item(), (forces_c - forces).abs().max().item())
    return worst


def energy_change_with_edge_attr_frozen(model: nn.Module, ethanol) -> float:
    """Smallest |dE| over frames when atom 0 moves 0.05 A but the graph and `edge_attr` do not."""
    smallest = float("inf")
    for pos in ethanol.frames:
        graph = graph_for(ethanol.z, pos)
        moved = pos.clone()
        moved[0, 0] += 0.05
        energy = predict(model, graph)[0]
        energy_moved = predict(model, Graph(graph.z, moved, graph.edge_index, graph.edge_attr, graph.batch))[0]
        smallest = min(smallest, (energy_moved - energy).abs().max().item())
    return smallest


def smallest_atom_force(model: nn.Module, ethanol) -> float:
    return min(
        predict(model, graph_for(ethanol.z, pos))[1].norm(dim=1).min().item()
        for pos in ethanol.frames
    )


# --------------------------------------------------------------------------
# 1-3. Rotation, translation, permutation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(ROTATIONS))
def test_rotation_leaves_energy_unchanged_and_rotates_forces(model, ethanol, name):
    """E(Qx) = E(x) and F(Qx) = Q F(x). Tolerance: TOL_E / TOL_F (1e-9)."""
    energy_error, force_error = symmetry_error(model, ethanol, ROTATIONS[name])
    assert energy_error <= TOL_E, f"|dE| = {energy_error:.3e} kcal/mol"
    assert force_error <= TOL_F, f"|F(Qx) - Q F(x)| = {force_error:.3e} kcal/mol/A"


def test_translation_leaves_energy_and_forces_unchanged(model, ethanol):
    """E(x + t) = E(x), F(x + t) = F(x). Tolerance: TOL_E / TOL_F (1e-9)."""
    energy_error, force_error = symmetry_error(model, ethanol, TRANSLATION)
    assert energy_error <= TOL_E, f"|dE| = {energy_error:.3e} kcal/mol"
    assert force_error <= TOL_F, f"|dF| = {force_error:.3e} kcal/mol/A"


def test_permuting_atoms_leaves_energy_unchanged_and_permutes_forces(model, ethanol):
    """Graph rebuilt from the permuted frame, so edge order changes too. Tolerance: 1e-9."""
    energy_error, force_error = symmetry_error(model, ethanol, PERMUTATION)
    assert energy_error <= TOL_E, f"|dE| = {energy_error:.3e} kcal/mol"
    assert force_error <= TOL_F, f"|F_perm - F[perm]| = {force_error:.3e} kcal/mol/A"


# --------------------------------------------------------------------------
# 4. Forces are the gradient of the reported energy
# --------------------------------------------------------------------------

def test_forces_match_central_finite_differences(model, ethanol):
    """F = -dE/dR against central differences, h = FD_STEP (1e-4 A). Tolerance: FD_TOL (1e-4 relative).

    The graph and `edge_attr` are rebuilt at each displaced geometry, as the
    data pipeline would, so a model reading detached features cannot pass.
    """
    error = worst_finite_difference_error(model, ethanol)
    assert error <= FD_TOL, f"max |F - F_fd| / max |F_fd| = {error:.3e}"
    assert smallest_atom_force(model, ethanol) > 1e-6, "forces are ~zero; the check would be vacuous"


# --------------------------------------------------------------------------
# 5. No net force on an isolated molecule
# --------------------------------------------------------------------------

def test_net_force_is_zero(model, ethanol):
    """|sum_i F_i| per component. Tolerance: TOL_F (1e-9)."""
    net = worst_net_force(model, ethanol)
    assert net <= TOL_F, f"|sum F| = {net:.3e} kcal/mol/A"


# --------------------------------------------------------------------------
# 6. Batching through the real PyG loader
# --------------------------------------------------------------------------

@pytest.mark.parametrize("batch_size", [2, 4])
def test_batched_predictions_match_single_frame_predictions(model, train_subset, batch_size, request):
    """Real `MD17Dataset` frames collated by `DataLoader`. Tolerance: TOL_E / TOL_F (1e-9)."""
    if request.node.callspec.params["model"].removesuffix("_trained") in STOCHASTIC:
        request.applymarker(pytest.mark.xfail(strict=True, reason=BATCH_COUPLING_REASON))
    energy_error, force_error, spread = batch_error(model, train_subset, batch_size)
    assert energy_error <= TOL_E, f"|dE| = {energy_error:.3e} kcal/mol"
    assert force_error <= TOL_F, f"|dF| = {force_error:.3e} kcal/mol/A"
    # Distinct frames, or a model returning frame 0's result everywhere passes.
    assert spread > 1e-3


# --------------------------------------------------------------------------
# 7. Gradient flow: geometry is read from `pos`, never from `edge_attr`
# --------------------------------------------------------------------------

def test_every_atom_receives_a_force(model, ethanol):
    """A zero row means that atom's coordinates never reached the energy. Threshold: 1e-6."""
    smallest = smallest_atom_force(model, ethanol)
    assert smallest > 1e-6, f"smallest per-atom |F| = {smallest:.3e} kcal/mol/A"


def test_edge_attr_does_not_reach_the_force_path(model, ethanol):
    """Zeroed, noisy (x100) or missing `edge_attr` must change nothing. Tolerance: exact (0.0)."""
    change = edge_attr_sensitivity(model, ethanol)
    assert change == 0.0, (
        f"edge_attr changed E or F by {change:.3e}. edge_attr from MD17Dataset is detached "
        "from pos, so a model that reads it reports forces that are NOT -dE/dR."
    )


def test_energy_is_recomputed_from_pos(model, ethanol):
    """Moving one atom 0.05 A with `edge_attr` frozen must change the energy. Threshold: 1e-6."""
    change = energy_change_with_edge_attr_frozen(model, ethanol)
    assert change > 1e-6, f"|dE| = {change:.3e}: the energy does not depend on pos"


def test_loader_fed_forces_are_the_true_gradient(model, train_subset, ethanol):
    """The real loader hands over detached `edge_attr`; forces must still match FD. Tolerance: FD_TOL.

    This is the end-to-end form of check 7: the exact tensors training sees,
    against finite differences of the energy over rebuilt geometry.
    """
    from torch_geometric.loader import DataLoader

    batch = next(iter(DataLoader(train_subset, batch_size=1)))
    # The upstream fact the whole check rests on.
    assert batch.edge_attr.grad_fn is None and not batch.edge_attr.requires_grad
    assert not batch.pos.requires_grad

    _, forces = predict(model, graph_from_loader(batch))
    error = finite_difference_error(model, ethanol.z, batch.pos.double(), forces)
    assert error <= FD_TOL, (
        f"FORCES ARE NOT -dE/dR on loader data: relative error {error:.3e}. "
        "The model is reading geometry from the detached edge_attr."
    )


def test_painn_energy_only_entry_point_agrees_and_ignores_edge_attr(ethanol):
    """PaiNN-specific: `predict_energy` is the same function of `pos` as the force path.

    The other models' `predict_energy` reads `edge_attr` by design, so this
    is not run on the controls. Tolerance: TOL_E / TOL_F (1e-9).
    """
    model = build_painn(ethanol)
    for pos in ethanol.frames:
        graph = graph_for(ethanol.z, pos)
        energy, forces = predict(model, graph)
        pos_grad = pos.clone().requires_grad_(True)
        energy_only = model.predict_energy(graph.z, pos_grad, graph.edge_index, None, graph.batch)
        (gradient,) = torch.autograd.grad(energy_only.sum(), pos_grad)
        assert (energy_only.detach() - energy).abs().max().item() <= TOL_E
        assert (-gradient - forces).abs().max().item() <= TOL_F


# --------------------------------------------------------------------------
# Negative controls: every check above must be able to fail
# --------------------------------------------------------------------------

def test_vector_bias_mutant_fails_rotation(mutants, ethanol):
    energy_error, force_error = symmetry_error(mutants["vector_bias"], ethanol, ROTATIONS["rotation_a"])
    assert energy_error > MARGIN * TOL_E and force_error > MARGIN * TOL_F


def test_absolute_position_mutant_fails_translation_and_net_force(mutants, ethanol):
    energy_error, force_error = symmetry_error(mutants["absolute_position"], ethanol, TRANSLATION)
    assert energy_error > MARGIN * TOL_E
    assert worst_net_force(mutants["absolute_position"], ethanol) > MARGIN * TOL_F


def test_index_dependent_mutant_fails_permutation_only(mutants, ethanol):
    mutant = mutants["index_dependent"]
    energy_error, force_error = symmetry_error(mutant, ethanol, PERMUTATION)
    assert energy_error > MARGIN * TOL_E and force_error > MARGIN * TOL_F
    # Not a mutant that is simply broken everywhere: rotation still passes.
    assert max(symmetry_error(mutant, ethanol, ROTATIONS["rotation_a"])) <= TOL_F


def test_stale_edge_attr_mutant_passes_every_symmetry_check(mutants, ethanol):
    """Why checks 4 and 7 exist: this bug is invisible to rotation, translation and permutation."""
    mutant = mutants["stale_radial"]
    for transform in (ROTATIONS["rotation_a"], TRANSLATION, PERMUTATION):
        assert max(symmetry_error(mutant, ethanol, transform)) <= TOL_F
    assert worst_net_force(mutant, ethanol) <= TOL_F
    assert smallest_atom_force(mutant, ethanol) > 1e-6  # plausible, non-zero forces


def test_stale_edge_attr_mutant_fails_gradient_checks(mutants, ethanol, train_subset):
    from torch_geometric.loader import DataLoader

    mutant = mutants["stale_radial"]
    assert worst_finite_difference_error(mutant, ethanol) > MARGIN * FD_TOL
    assert edge_attr_sensitivity(mutant, ethanol) > 0.0

    batch = next(iter(DataLoader(train_subset, batch_size=1)))
    _, forces = predict(mutant, graph_from_loader(batch))
    assert finite_difference_error(mutant, ethanol.z, batch.pos.double(), forces) > MARGIN * FD_TOL


def test_energy_only_mutant_fails_gradient_checks(mutants, ethanol):
    mutant = mutants["stale_energy_only"]
    assert smallest_atom_force(mutant, ethanol) == 0.0
    assert energy_change_with_edge_attr_frozen(mutant, ethanol) == 0.0
    assert worst_finite_difference_error(mutant, ethanol) > MARGIN * FD_TOL


def test_batch_blind_mutant_fails_batching(mutants, train_subset):
    energy_error, force_error, _ = batch_error(mutants["batch_blind"], train_subset, batch_size=2)
    assert energy_error > MARGIN * TOL_E and force_error > MARGIN * TOL_F
