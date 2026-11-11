"""Basic correctness for Phase 3's first-draft model (ml/models/graph_stochastic.py).

Deliberately NOT an extension of tests/ml/test_painn_physics.py's six-model
matrix (mutants, rotation/translation/permutation, real MD17 frames) - that
file is Fazin's, and giving graph_stochastic the same negative-control
treatment the other six models get is real follow-up work for whoever owns
that integration, not something to fold in here without that review. This
file only checks the two things a brand-new model must not get wrong before
anyone trains it for real: forces are actually -dE/dR, and it round-trips
through the registry/checkpoint contract the whole harness depends on.
"""
from __future__ import annotations

import torch

from ml.models.graph_stochastic import GraphStochastic
from ml.models.registry import MODEL_REGISTRY, build_model


def _toy_graph(num_atoms=5, num_rbf=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randint(1, 10, (num_atoms,), generator=g)
    pos = torch.randn(num_atoms, 3, generator=g, dtype=torch.float64)
    row = torch.arange(num_atoms).repeat_interleave(num_atoms)
    col = torch.arange(num_atoms).repeat(num_atoms)
    mask = row != col
    row, col = row[mask], col[mask]
    edge_index = torch.stack([row, col])
    edge_attr = torch.zeros(edge_index.shape[1], num_rbf, dtype=torch.float64)  # unused: forces live-recompute RBF
    return z, pos, edge_index, edge_attr


def test_graph_stochastic_is_registered():
    assert MODEL_REGISTRY["graph_stochastic"] is GraphStochastic


def test_config_round_trips_through_build_model():
    model = GraphStochastic(hidden_channels=16, num_layers=2, num_rbf=8, kl_weight=0.01)
    rebuilt = build_model("graph_stochastic", **model.config())
    assert rebuilt.config() == model.config()


def test_forces_match_finite_difference_energy_gradient():
    torch.manual_seed(0)
    model = GraphStochastic(hidden_channels=16, num_layers=2, num_rbf=8, kl_weight=0.0).double().eval()
    z, pos, edge_index, edge_attr = _toy_graph()

    # Freeze this forward's noise draws so finite-difference perturbations
    # of pos are compared against the SAME sampled weights/noise, not a
    # fresh draw each call - the gradient check would otherwise be
    # comparing two different stochastic functions.
    state = torch.get_rng_state()

    def energy_at(p):
        torch.set_rng_state(state)
        pos_req = p.clone().requires_grad_(True)
        e, _ = model.predict_energy_and_forces(z, pos_req, edge_index, edge_attr)
        return e

    torch.set_rng_state(state)
    pos_req = pos.clone().requires_grad_(True)
    _, forces = model.predict_energy_and_forces(z, pos_req, edge_index, edge_attr)

    step = 1e-4
    fd_forces = torch.zeros_like(pos)
    for i in range(pos.shape[0]):
        for d in range(3):
            plus, minus = pos.clone(), pos.clone()
            plus[i, d] += step
            minus[i, d] -= step
            fd_forces[i, d] = -(energy_at(plus).item() - energy_at(minus).item()) / (2 * step)

    max_fd = fd_forces.abs().max().item()
    rel_err = (forces.detach() - fd_forces).abs().max().item() / max(max_fd, 1e-12)
    assert rel_err < 1e-3, f"forces disagree with finite-difference dE/dR: rel_err={rel_err}"


def test_kl_divergence_is_finite_and_nonzero_at_init():
    torch.manual_seed(0)
    model = GraphStochastic(hidden_channels=16, num_layers=2, num_rbf=8)
    z, pos, edge_index, edge_attr = _toy_graph()
    model.predict_energy_and_forces(z, pos.float(), edge_index, edge_attr.float())
    kl = model.kl_divergence()
    assert torch.isfinite(kl)
    assert kl.item() > 0  # softplus(-3) != PRIOR_SIGMA, so KL is not trivially zero at init


def test_predict_distribution_has_nonzero_spread():
    torch.manual_seed(0)
    model = GraphStochastic(hidden_channels=16, num_layers=2, num_rbf=8, num_mc_samples=8).eval()
    z, pos, edge_index, edge_attr = _toy_graph(seed=1)
    pos = pos.float()
    batch = torch.zeros(z.shape[0], dtype=torch.long)
    e_mean, e_var, f_mean, f_var = model.predict_distribution(
        z, pos, edge_index, edge_attr.float(), batch=batch,
    )
    assert torch.isfinite(e_mean).all() and torch.isfinite(e_var).all()
    assert torch.isfinite(f_mean).all() and torch.isfinite(f_var).all()
    assert (e_var > 0).all()  # repeated sampling must actually disagree with itself
