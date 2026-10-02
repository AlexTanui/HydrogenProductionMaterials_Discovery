"""Tests for ml/models/physnet.py.

No data and no training required - every case builds a small random
molecule in memory, so this runs in seconds and can gate every commit.

The one that matters most is `test_forces_match_finite_differences`:
`ROADMAP.md` weeks 3-4 call for exactly this check, and it is what
distinguishes a model that predicts forces from one that returns a
plausibly-shaped tensor of zeros.

    pytest tests/test_physnet.py
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PhysNet needs torch installed")

from ml.models.physnet import (  # noqa: E402
    PhysNet,
    build_model,
    cosine_cutoff,
    rbf_from_positions,
    shifted_softplus,
)


def make_molecule(n_atoms=9, cutoff=5.0, seed=0, spread=2.2):
    """A small random molecule plus its neighbour graph, like MD17 ethanol."""
    g = torch.Generator().manual_seed(seed)
    z = torch.tensor([6, 6, 8] + [1] * (n_atoms - 3))[:n_atoms]
    pos = torch.rand(n_atoms, 3, generator=g, dtype=torch.float64) * spread
    dist = torch.cdist(pos, pos)
    mask = (dist <= cutoff) & ~torch.eye(n_atoms, dtype=torch.bool)
    row, col = mask.nonzero(as_tuple=True)
    return z, pos, torch.stack([row, col], dim=0)


def make_model(seed=0, **kwargs):
    torch.manual_seed(seed)
    kwargs.setdefault("hidden_channels", 32)
    kwargs.setdefault("num_modules", 2)
    kwargs.setdefault("num_layers", 2)
    model = PhysNet(**kwargs).double()
    model.eval()
    return model


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------

def test_shifted_softplus_is_zero_at_zero():
    """ssp(0) == 0 is the property that makes it a drop-in for a
    zero-centred activation; a plain softplus offsets every layer."""
    assert abs(float(shifted_softplus(torch.zeros(1)))) < 1e-12


def test_rbf_matches_the_dataset_side_expansion():
    """The model differentiates the graph the dataset built - so its RBF
    must equal ml/data/md17.py::_gaussian_rbf, or training and evaluation
    silently see different edge features."""
    from ml.data.md17 import _gaussian_rbf

    _z, pos, edge_index = make_molecule()
    rbf_model, dist = rbf_from_positions(pos, edge_index, num_rbf=16, cutoff=5.0)
    rbf_dataset = _gaussian_rbf(dist, num_rbf=16, cutoff=5.0)
    assert torch.allclose(rbf_model, rbf_dataset, atol=1e-10)


def test_cosine_cutoff_is_one_at_zero_and_zero_past_cutoff():
    d = torch.tensor([0.0, 2.5, 4.999, 5.0, 7.0], dtype=torch.float64)
    cut = cosine_cutoff(d, cutoff=5.0)
    assert abs(float(cut[0]) - 1.0) < 1e-12
    assert float(cut[3]) == 0.0 and float(cut[4]) == 0.0
    assert float(cut[2]) < 1e-5, "envelope must decay to ~0 as it reaches the cutoff"
    assert torch.all(cut[:-1] >= cut[1:] - 1e-12), "must be monotonically decreasing"


# --------------------------------------------------------------------------
# shapes and the bake-off interface
# --------------------------------------------------------------------------

def test_predict_energy_returns_one_value_per_graph():
    model = make_model()
    z, pos, edge_index = make_molecule()
    rbf, _ = rbf_from_positions(pos, edge_index)
    energy = model.predict_energy(z, edge_index, rbf, None)
    assert energy.shape == (1,)


def test_predict_energy_accepts_both_circulating_signatures():
    """SCRUM-50's ticket says predict_energy(z, edge_index, edge_attr,
    batch); ml/models/registry.py's docstring says
    predict_energy(z, pos, edge_index, edge_attr, batch). Until the team
    settles on one, this model answers to both and gives the same
    number."""
    model = make_model(seed=13)
    z, pos, edge_index = make_molecule(seed=13)
    rbf, _ = rbf_from_positions(pos, edge_index)

    ticket_form = model.predict_energy(z, edge_index, rbf, None)
    registry_form = model.predict_energy(z, pos, edge_index, rbf, None)
    assert torch.allclose(ticket_form, registry_form, atol=1e-12)


def test_exposes_the_registry_required_interface():
    """ml/models/registry.py::build_model refuses to construct a model
    missing any of these, so a mis-registered class fails where the cause
    is obvious rather than deep inside an eval loop."""
    for name in ("predict_energy", "predict_energy_and_forces", "config"):
        assert hasattr(PhysNet, name), f"registry requires {name}()"


def test_config_round_trips_through_the_constructor():
    """A checkpoint stores config() and evaluation rebuilds from it, so it
    must reconstruct the identical model rather than a similar one."""
    model = make_model(seed=14, hidden_channels=48, num_modules=2, num_layers=3)
    rebuilt = PhysNet(**model.config()).double()
    assert rebuilt.config() == model.config()
    assert sum(p.numel() for p in rebuilt.parameters()) == sum(p.numel() for p in model.parameters())


def test_exposes_cutoff_radius_for_the_evaluate_harness():
    """ml/training/evaluate.py probes getattr(model, 'cutoff_radius') to
    confirm a checkpoint is scored at the radius it was trained at. Under
    the short name alone that check finds nothing and passes vacuously."""
    model = make_model(cutoff=4.5)
    assert getattr(model, "cutoff_radius", None) == 4.5


def test_evaluate_harness_call_signature():
    """evaluate.py calls predict_energy_and_forces positionally with
    (batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch),
    where x is [N, 1]. Exercised exactly as written there."""
    model = make_model(seed=15)
    z, pos, edge_index = make_molecule(n_atoms=7, seed=15)
    x = z.reshape(-1, 1)  # PyG carries atomic number as a column
    edge_attr, _ = rbf_from_positions(pos, edge_index)
    batch = torch.zeros(7, dtype=torch.long)

    energy, forces = model.predict_energy_and_forces(x, pos, edge_index, edge_attr, batch)
    assert energy.shape == (1,) and forces.shape == (7, 3)


def test_predict_energy_and_forces_shapes():
    model = make_model()
    z, pos, edge_index = make_molecule(n_atoms=9)
    energy, forces = model.predict_energy_and_forces(z, pos, edge_index, None, None)
    assert energy.shape == (1,)
    assert forces.shape == (9, 3)


def test_batching_matches_per_molecule_evaluation():
    """Two molecules in one batch must give the same energies as running
    them separately - the check that index_add_ aggregation respects
    graph boundaries."""
    model = make_model()
    z1, pos1, ei1 = make_molecule(n_atoms=9, seed=1)
    z2, pos2, ei2 = make_molecule(n_atoms=7, seed=2)

    e1 = model.predict_energy_and_forces(z1, pos1, ei1)[0]
    e2 = model.predict_energy_and_forces(z2, pos2, ei2)[0]

    z = torch.cat([z1, z2])
    pos = torch.cat([pos1, pos2])
    edge_index = torch.cat([ei1, ei2 + z1.shape[0]], dim=1)
    batch = torch.cat([torch.zeros(z1.shape[0], dtype=torch.long),
                       torch.ones(z2.shape[0], dtype=torch.long)])

    energies, forces = model.predict_energy_and_forces(z, pos, edge_index, None, batch)
    assert energies.shape == (2,)
    assert forces.shape == (16, 3)
    assert torch.allclose(energies, torch.cat([e1, e2]), atol=1e-8)


# --------------------------------------------------------------------------
# forces - the checks that matter
# --------------------------------------------------------------------------

def test_forces_are_not_identically_zero():
    """The failure this guards against is silent.

    edge_attr from MD17Dataset is built in numpy, so it has no gradient
    path to positions. A model that reads only edge_attr returns forces
    of exactly zero, with no error raised - and with force_loss_weight
    at 100, training would look healthy while learning nothing about
    forces. predict_energy_and_forces recomputes the basis from pos to
    avoid this; this test is what keeps it that way.
    """
    model = make_model()
    z, pos, edge_index = make_molecule()
    stale_edge_attr = torch.zeros(edge_index.shape[1], 16, dtype=torch.float64)

    _energy, forces = model.predict_energy_and_forces(z, pos, edge_index, stale_edge_attr)
    assert forces.abs().max() > 1e-8, "forces are all zero - the autograd path is broken"


def test_forces_match_finite_differences():
    """F = -dE/dR, verified numerically. ROADMAP.md weeks 3-4.

    Central differences in float64. If autograd and finite differences
    disagree, the analytic force is wrong however plausible it looks.
    """
    model = make_model(seed=3)
    z, pos, edge_index = make_molecule(n_atoms=6, seed=3)

    _energy, forces = model.predict_energy_and_forces(z, pos, edge_index)

    def energy_at(p):
        rbf, dist = rbf_from_positions(p, edge_index, model.num_rbf, model.cutoff)
        cut = cosine_cutoff(dist, model.cutoff)
        with torch.no_grad():
            atomic = model._atomic_energies(z.reshape(-1).long(), rbf, cut, edge_index)
        return float(atomic.sum())

    h = 1e-5
    numerical = torch.zeros_like(forces)
    for i in range(pos.shape[0]):
        for k in range(3):
            plus, minus = pos.clone(), pos.clone()
            plus[i, k] += h
            minus[i, k] -= h
            numerical[i, k] = -(energy_at(plus) - energy_at(minus)) / (2 * h)

    assert torch.allclose(forces, numerical, atol=1e-6), (
        f"autograd forces disagree with finite differences; "
        f"max abs diff {float((forces - numerical).abs().max()):.3e}"
    )


def test_forces_sum_to_zero():
    """Newton's third law: with no external field, internal forces cancel.

    Follows from the energy depending only on relative positions. A
    non-zero net force means something in the model reads absolute
    coordinates - a real bug, and one that makes any MD run drift.
    """
    model = make_model(seed=4)
    z, pos, edge_index = make_molecule(n_atoms=8, seed=4)
    _energy, forces = model.predict_energy_and_forces(z, pos, edge_index)
    assert forces.sum(dim=0).abs().max() < 1e-8


# --------------------------------------------------------------------------
# physical invariances
# --------------------------------------------------------------------------

def test_energy_is_translation_invariant():
    model = make_model(seed=5)
    z, pos, edge_index = make_molecule(seed=5)
    e1 = model.predict_energy_and_forces(z, pos, edge_index)[0]
    e2 = model.predict_energy_and_forces(z, pos + 7.5, edge_index)[0]
    assert torch.allclose(e1, e2, atol=1e-9)


def test_energy_is_rotation_invariant():
    """Only interatomic distances enter the model, so a rigid rotation
    must leave the energy untouched."""
    model = make_model(seed=6)
    z, pos, edge_index = make_molecule(seed=6)
    theta = 0.7
    rot = torch.tensor([[np.cos(theta), -np.sin(theta), 0.0],
                        [np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, 1.0]], dtype=torch.float64)
    e1 = model.predict_energy_and_forces(z, pos, edge_index)[0]
    e2 = model.predict_energy_and_forces(z, pos @ rot.T, edge_index)[0]
    assert torch.allclose(e1, e2, atol=1e-9)


def test_energy_is_permutation_invariant():
    """Relabelling atoms must not change the energy."""
    model = make_model(seed=7)
    z, pos, edge_index = make_molecule(n_atoms=8, seed=7)
    perm = torch.randperm(8, generator=torch.Generator().manual_seed(11))
    inverse = torch.argsort(perm)

    e1 = model.predict_energy_and_forces(z, pos, edge_index)[0]
    e2 = model.predict_energy_and_forces(z[perm], pos[perm], inverse[edge_index])[0]
    assert torch.allclose(e1, e2, atol=1e-9)


def test_energy_is_extensive_in_atom_count():
    """Two copies of a molecule, far enough apart to share no edges,
    should have exactly twice the energy. This is what makes the
    atom-wise decomposition meaningful rather than decorative."""
    model = make_model(seed=8)
    z, pos, edge_index = make_molecule(n_atoms=6, seed=8)

    single = model.predict_energy_and_forces(z, pos, edge_index)[0]
    far = pos + torch.tensor([500.0, 0.0, 0.0], dtype=torch.float64)
    z2 = torch.cat([z, z])
    pos2 = torch.cat([pos, far])
    edge_index2 = torch.cat([edge_index, edge_index + z.shape[0]], dim=1)

    doubled = model.predict_energy_and_forces(z2, pos2, edge_index2)[0]
    assert torch.allclose(doubled, 2 * single, atol=1e-8)


# --------------------------------------------------------------------------
# the energy-offset problem
# --------------------------------------------------------------------------

def test_shift_initialisation_puts_output_in_the_right_range():
    """MD17 ethanol sits near -97,000 kcal/mol while a freshly
    initialised network outputs O(1). init_shift_from_energies closes
    that gap before training starts, instead of asking the optimiser to
    travel five orders of magnitude first."""
    model = make_model(seed=9)
    z, pos, edge_index = make_molecule(n_atoms=9, seed=9)

    before = float(model.predict_energy_and_forces(z, pos, edge_index)[0])
    assert abs(before) < 100, "untrained model should start near zero without a shift"

    target = torch.full((64,), -97000.0, dtype=torch.float64)
    model.init_shift_from_energies(z, target)
    after = float(model.predict_energy_and_forces(z, pos, edge_index)[0])
    assert abs(after - (-97000.0)) < abs(before - (-97000.0)), "shift must move output toward the data"
    assert abs(after + 97000.0) < 5000.0, f"expected ~-97000 after shift, got {after:.1f}"


def test_shift_does_not_break_force_calculation():
    """A per-element constant has zero position derivative, so adding it
    must leave forces untouched."""
    model = make_model(seed=10)
    z, pos, edge_index = make_molecule(seed=10)
    _e, f_before = model.predict_energy_and_forces(z, pos, edge_index)
    model.init_shift_from_energies(z, torch.full((32,), -97000.0, dtype=torch.float64))
    _e2, f_after = model.predict_energy_and_forces(z, pos, edge_index)
    assert torch.allclose(f_before, f_after, atol=1e-10)


# --------------------------------------------------------------------------
# config wiring
# --------------------------------------------------------------------------

def test_build_model_reads_a_dict_config():
    """Works against a plain dict now, and against ml/config.py's object
    once Ruturaj's schema lands - neither is hardcoded."""
    model = build_model({
        "model": {"hidden_channels": 64, "num_modules": 2, "num_layers": 3},
        "data": {"cutoff_radius": 4.0},
    })
    assert model.hidden_channels == 64
    assert model.cutoff == 4.0
    assert len(model.modules_output) == 2


def test_model_trains_one_step_without_error():
    """A single optimiser step on a combined energy+force loss with the
    bake-off's fixed 1:100 weighting - proves the force term is
    differentiable, which create_graph=True is what enables."""
    torch.manual_seed(12)
    model = PhysNet(hidden_channels=32, num_modules=1, num_layers=1).double()
    model.train()
    z, pos, edge_index = make_molecule(n_atoms=6, seed=12)
    target_e = torch.tensor([-97000.0], dtype=torch.float64)
    target_f = torch.randn(6, 3, dtype=torch.float64)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    energy, forces = model.predict_energy_and_forces(z, pos, edge_index)
    loss = 1.0 * (energy - target_e).abs().mean() + 100.0 * (forces - target_f).abs().mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert any(g.abs().max() > 0 for g in grads), "all gradients are zero"
    opt.step()
