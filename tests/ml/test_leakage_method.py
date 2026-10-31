"""Tests for ml/utils/leakage.py: the distance and search the SCRUM-47
leakage audit rests on. Synthetic data only - the checks against real
gold files are in test_split_leakage.py.

A wrong distance would make every number in the audit wrong without any
error, so the RMSD is checked against an explicit-rotation Kabsch, against
the invariances it claims, and the fast search against brute force.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.utils import leakage

Z_ETHANOL = np.array([6, 6, 8, 1, 1, 1, 1, 1, 1])


def _ethanol_like(rng: np.random.Generator) -> np.ndarray:
    """Heavy atoms spread out, each hydrogen ~1 A from 'its' heavy atom:
    H 3-4 on atom 0, H 5-7 on atom 1, H 8 on atom 2 (ethanol's grouping)."""
    heavy = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [-0.7, 1.2, 0.0]])
    parents = [0, 0, 1, 1, 1, 2]
    h = heavy[parents] + rng.normal(size=(6, 3)) * 0.2 + np.array([0.0, 0.0, 1.0])
    return np.concatenate([heavy, h]) + rng.normal(size=(9, 3)) * 0.05


def _explicit_kabsch(P: np.ndarray, Q: np.ndarray) -> float:
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(U @ Vt))
    rotation = U @ np.diag([1.0, 1.0, d]) @ Vt
    return float(np.sqrt(((P @ rotation - Q) ** 2).sum() / len(P)))


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def test_rmsd_matches_explicit_rotation_kabsch():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(200, 9, 3))
    Q = rng.normal(size=(200, 9, 3))
    fast = leakage.kabsch_rmsd(torch.tensor(P), torch.tensor(Q)).numpy()
    slow = np.array([_explicit_kabsch(p, q) for p, q in zip(P, Q)])
    assert np.max(np.abs(fast - slow)) < 1e-10


def test_near_duplicates_are_resolved_to_well_below_a_trajectory_step():
    """Consecutive MD frames differ by ~0.01 A; the distance must resolve
    that, not lose it to cancellation."""
    rng = np.random.default_rng(1)
    P = rng.normal(size=(200, 9, 3))
    Q = P + rng.normal(size=P.shape) * 0.001
    fast = leakage.kabsch_rmsd(torch.tensor(P), torch.tensor(Q)).numpy()
    slow = np.array([_explicit_kabsch(p, q) for p, q in zip(P, Q)])
    assert np.max(np.abs(fast - slow)) < 1e-9


def test_rotated_and_translated_copy_is_at_distance_zero():
    rng = np.random.default_rng(2)
    P = _ethanol_like(rng)
    Q = P @ _random_rotation(rng).T + np.array([3.0, -2.0, 7.0])
    assert leakage.kabsch_rmsd(torch.tensor(P), torch.tensor(Q)).item() < 1e-7


def test_a_mirror_image_counts_only_when_reflections_are_allowed():
    rng = np.random.default_rng(3)
    P = _ethanol_like(rng)
    M = P * np.array([-1.0, 1.0, 1.0])
    assert leakage.kabsch_rmsd(torch.tensor(P), torch.tensor(M)).item() > 0.1
    assert leakage.kabsch_rmsd(torch.tensor(P), torch.tensor(M), allow_reflection=True).item() < 1e-7


def test_closed_form_eigenvalues_match_eigvalsh_including_degenerate_cases():
    rng = np.random.default_rng(4)
    A = torch.tensor(rng.normal(size=(5000, 3, 3)))
    A = A @ A.transpose(-1, -2)
    cases = [A, 2.0 * torch.eye(3, dtype=torch.float64).expand(10, 3, 3),
             torch.diag_embed(torch.tensor([[1.0, 1.0, 3.0]], dtype=torch.float64)).expand(10, 3, 3)]
    for M in cases:
        scale = M.abs().amax(dim=(-1, -2))[..., None]
        error = (leakage._sym3_eigvals(M) - torch.linalg.eigvalsh(M)).abs() / scale
        assert float(error.max()) < 1e-8


def test_ethanol_groups_and_relabellings():
    rng = np.random.default_rng(5)
    groups = leakage.equivalent_atom_groups(Z_ETHANOL, _ethanol_like(rng))
    assert [g.tolist() for g in groups] == [[0], [1], [2], [3, 4], [5, 6, 7], [8]]
    perms = leakage.group_permutations(groups, 9)
    assert len(perms) == 2 * 6  # 2! x 3!
    assert perms[0].tolist() == list(range(9))
    assert len({tuple(p) for p in perms}) == len(perms)


def test_swapping_equivalent_hydrogens_is_at_distance_zero():
    rng = np.random.default_rng(6)
    P = _ethanol_like(rng)
    groups = leakage.equivalent_atom_groups(Z_ETHANOL, P)
    perms = leakage.group_permutations(groups, 9)
    swapped = P[perms[7]] @ _random_rotation(rng).T
    assert leakage.kabsch_rmsd(torch.tensor(swapped), torch.tensor(P)).item() > 0.1
    best, _ = leakage.min_rmsd_over_permutations(torch.tensor(swapped)[None], torch.tensor(P)[None, None],
                                                 torch.tensor(perms))
    assert best.item() < 1e-7


def test_descriptors_have_the_invariances_they_claim():
    rng = np.random.default_rng(7)
    P = _ethanol_like(rng)
    groups = leakage.equivalent_atom_groups(Z_ETHANOL, P)
    moved = P @ _random_rotation(rng).T + 4.0
    mirrored = P * np.array([1.0, -1.0, 1.0])
    relabelled = P[leakage.group_permutations(groups, 9)[5]]
    dv = leakage.distance_vector(torch.tensor(P))
    for other in (moved, mirrored):
        assert torch.allclose(leakage.distance_vector(torch.tensor(other)), dv, atol=1e-10)
    inv = leakage.invariant_descriptor(P[None], groups)
    for other in (moved, mirrored, relabelled):
        assert np.allclose(leakage.invariant_descriptor(other[None], groups), inv, atol=1e-10)


def test_shortlisted_nearest_neighbours_match_brute_force():
    """A random-walk 'trajectory' with random rotations and relabellings
    mixed in: the shortlist must find the same closest frame as checking
    every frame under every relabelling."""
    rng = np.random.default_rng(8)
    start = _ethanol_like(rng)
    groups = leakage.equivalent_atom_groups(Z_ETHANOL, start)
    perms = leakage.group_permutations(groups, 9)
    steps = rng.normal(size=(1500, 9, 3)) * 0.02
    trajectory = start + np.cumsum(steps, axis=0)
    ref = trajectory[:1200]
    queries = trajectory[1200:][::10].copy()
    for i in range(len(queries)):
        queries[i] = queries[i][perms[rng.integers(len(perms))]] @ _random_rotation(rng).T
    fast, index = leakage.nearest_neighbours(queries, ref, groups, k=64, device="cpu")
    for q, value, i in zip(queries, fast, index):
        exact, exact_i = leakage.exact_nearest(q, ref, groups, device="cpu")
        assert value == pytest.approx(exact, abs=1e-9)
        assert value == pytest.approx(
            leakage.min_rmsd_over_permutations(torch.tensor(q)[None], torch.tensor(ref[i])[None, None],
                                               torch.tensor(perms))[0].item(), abs=1e-12)


def test_split_gaps():
    tr, va, te = np.arange(0, 80), np.arange(80, 90), np.arange(90, 100)
    assert leakage.split_gaps(tr, va, te) == {"train_to_val": 0, "val_to_test": 0, "train_to_test": 10}
    # frames 70-74 lie between train and val, 85-89 between val and test
    assert leakage.split_gaps(np.arange(0, 70), np.arange(75, 85), np.arange(90, 100)) == {
        "train_to_val": 5, "val_to_test": 5, "train_to_test": 20}
    interleaved = leakage.split_gaps(np.arange(0, 100, 2), np.arange(1, 100, 4), np.arange(3, 100, 4))
    assert interleaved == {"train_to_val": None, "val_to_test": None, "train_to_test": None}
