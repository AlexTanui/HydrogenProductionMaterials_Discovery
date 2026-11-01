"""Tests for ml/data/md17.py: the graph every model trains on (`build_graph`),
the in-memory sample (`MD17Sample`) and element lookup.

`build_graph` was used by the PaiNN tests but never checked itself (22 Oct
audit, item 9): its edge set, symmetry and cutoff inclusivity are pinned
here against a brute-force float64 reference. Tests only - no change to
ml/data/.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from ml.data.md17 import MD17Sample, _gaussian_rbf, _symbol_to_z, build_graph

CUTOFF = 5.0


def _frame(n_atoms: int = 9, seed: int = 0, box: float = 6.0) -> np.ndarray:
    """Random positions with no pair distance within 1e-3 A of the cutoff, so
    float32 rounding can't flip an edge between build_graph and the reference."""
    rng = np.random.default_rng(seed)
    while True:
        pos = rng.uniform(0.0, box, size=(n_atoms, 3)).astype(np.float32)
        d = np.linalg.norm(pos[:, None, :].astype(np.float64) - pos[None, :, :], axis=-1)
        if not np.any(np.abs(d - CUTOFF) < 1e-3):
            return pos


def _edges(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return set(map(tuple, edge_index.t().tolist()))


@pytest.mark.parametrize("seed", range(5))
def test_edges_are_exactly_the_ordered_pairs_within_the_cutoff(seed):
    pos = _frame(seed=seed)
    edge_index, _ = build_graph(pos, cutoff=CUTOFF)
    expected = {
        (i, j) for i, j in itertools.permutations(range(len(pos)), 2)
        if np.linalg.norm(pos[i].astype(np.float64) - pos[j]) <= CUTOFF
    }
    assert _edges(edge_index) == expected
    assert edge_index.shape[1] == len(expected), "an edge appears more than once"


def test_graph_is_symmetric_and_has_no_self_loops():
    edge_index, edge_attr = build_graph(_frame(seed=7), cutoff=CUTOFF)
    edges = _edges(edge_index)
    assert all((j, i) in edges for i, j in edges)
    assert all(i != j for i, j in edges)
    position = {edge: k for k, edge in enumerate(map(tuple, edge_index.t().tolist()))}
    for (i, j), k in position.items():
        torch.testing.assert_close(edge_attr[k], edge_attr[position[(j, i)]], rtol=0, atol=0)


def test_cutoff_is_inclusive():
    """A pair exactly at the cutoff is an edge; just beyond it is not.
    (3, 4, 0) is exactly 5 A from the origin in float32 too."""
    edges = _edges(build_graph(np.array([[0, 0, 0], [3, 4, 0]], dtype=np.float32), cutoff=CUTOFF)[0])
    assert edges == {(0, 1), (1, 0)}
    beyond = np.array([[0, 0, 0], [3, 4.001, 0]], dtype=np.float32)
    assert build_graph(beyond, cutoff=CUTOFF)[0].shape[1] == 0


def test_edge_attr_is_the_rbf_of_each_edge_distance():
    pos = _frame(seed=3)
    edge_index, edge_attr = build_graph(pos, cutoff=CUTOFF, num_rbf=16)
    assert edge_attr.shape == (edge_index.shape[1], 16)
    p = torch.as_tensor(pos)
    distance = (p[edge_index[0]] - p[edge_index[1]]).norm(dim=-1)
    torch.testing.assert_close(edge_attr, _gaussian_rbf(distance, num_rbf=16, cutoff=CUTOFF), rtol=1e-5, atol=1e-6)


def test_edge_set_is_unchanged_by_rotation_and_translation():
    pos = _frame(seed=11)
    q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
    moved = (pos.astype(np.float64) @ q + np.array([2.5, -1.0, 4.0])).astype(np.float32)
    assert _edges(build_graph(pos)[0]) == _edges(build_graph(moved)[0])


# --------------------------------------------------------------------------
# MD17Sample and element lookup
# --------------------------------------------------------------------------

def _sample(**overrides) -> MD17Sample:
    fields = dict(molecule="toy", theory="dft", z=[6, 1, 1], R=np.zeros((4, 3, 3)),
                  E=np.zeros(4), F=np.zeros((4, 3, 3)))
    return MD17Sample(**{**fields, **overrides})


def test_sample_normalises_dtypes_and_promotes_a_single_frame():
    sample = _sample(R=np.zeros((3, 3)), F=np.zeros((3, 3)), E=np.zeros(1))
    assert sample.z.dtype == np.int64 and sample.R.dtype == sample.F.dtype == sample.E.dtype == np.float32
    assert sample.R.shape == (1, 3, 3) and sample.n_configs == 1 and sample.n_atoms == 3


@pytest.mark.parametrize("bad, match", [
    (dict(R=np.zeros((4, 2, 3))), "inconsistent with 3 atoms"),
    (dict(F=np.zeros((4, 3, 2))), "F shape"),
    (dict(E=np.zeros(5)), "E has 5 entries"),
])
def test_sample_rejects_disagreeing_shapes_at_construction(bad, match):
    with pytest.raises(ValueError, match=match):
        _sample(**bad)


def test_known_symbols_map_to_atomic_numbers_and_unknown_ones_raise():
    assert [_symbol_to_z(s) for s in ("H", "C", "N", "O")] == [1, 6, 7, 8]
    with pytest.raises(ValueError, match="_PERIODIC_TABLE"):
        _symbol_to_z("Xx")
