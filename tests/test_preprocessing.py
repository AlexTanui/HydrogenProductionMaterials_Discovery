"""Tests for ml/data/preprocessing.py: the trajectory-block splitter and the
silver-step validation (22 Oct audit, item 9 - neither had a direct test).
Tests only - no change to ml/data/.
"""
from __future__ import annotations

import numpy as np
import pytest

from ml.data.md17 import MD17Sample
from ml.data.preprocessing import trajectory_block_split, validate_sample


@pytest.mark.parametrize("n", [10, 97, 2000, 555092])
@pytest.mark.parametrize("fractions", [(0.8, 0.1, 0.1), (0.7, 0.15, 0.15), (0.6, 0.2, 0.2)])
def test_block_split_is_three_contiguous_ordered_blocks_covering_every_frame(n, fractions):
    train, val, test = trajectory_block_split(n, *fractions)
    together = np.concatenate([train, val, test])
    assert np.array_equal(together, np.arange(n)), "blocks must be back to back, in order, covering every frame"
    assert len(train) == round(n * fractions[0]) and len(val) == round(n * fractions[1])


def test_block_split_defaults_reproduce_the_ethanol_dft_gold_sizes():
    assert tuple(map(len, trajectory_block_split(555092))) == (444074, 55509, 55509)


def test_block_split_has_no_randomness():
    first, second = trajectory_block_split(1234), trajectory_block_split(1234)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_block_split_rejects_fractions_that_do_not_sum_to_one():
    with pytest.raises(AssertionError, match="sum to 1.0"):
        trajectory_block_split(100, 0.8, 0.1, 0.2)


def _sample(**overrides) -> MD17Sample:
    fields = dict(molecule="toy", theory="dft", z=[6, 1], R=np.zeros((3, 2, 3)),
                  E=np.zeros(3), F=np.zeros((3, 2, 3)))
    return MD17Sample(**{**fields, **overrides})


def test_a_clean_sample_has_no_problems():
    assert validate_sample(_sample()) == []


@pytest.mark.parametrize("field", ["R", "E", "F"])
def test_nan_in_any_array_is_reported(field):
    array = np.zeros((3,) if field == "E" else (3, 2, 3))
    array.flat[1] = np.nan
    assert validate_sample(_sample(**{field: array})) == [f"NaN values in {field}"]


def test_empty_samples_are_reported():
    assert "zero atoms" in validate_sample(_sample(z=[], R=np.zeros((3, 0, 3)), F=np.zeros((3, 0, 3))))
    assert "zero configurations" in validate_sample(_sample(R=np.zeros((0, 2, 3)), E=np.zeros(0),
                                                            F=np.zeros((0, 2, 3))))


def test_a_declared_unit_other_than_the_project_units_is_reported():
    """The check CLAUDE.md relies on: metrics assume kcal/mol and A without converting."""
    problems = validate_sample(_sample(r_unit="Bohr", e_unit="eV"))
    assert problems == ["unexpected r_unit 'Bohr', expected 'Ang'", "unexpected e_unit 'eV', expected 'kcal/mol'"]


# --- SCRUM-72: purge buffer and split provenance -------------------------

def test_purge_zero_is_the_original_split():
    for got, expected in zip(trajectory_block_split(555092, purge=0), trajectory_block_split(555092)):
        assert np.array_equal(got, expected)


def test_purge_leaves_a_gap_of_exactly_that_many_frames_and_keeps_train():
    tr0, _, _ = trajectory_block_split(1000)
    tr, va, te = trajectory_block_split(1000, purge=10)
    assert np.array_equal(tr, tr0)
    assert va[0] - tr[-1] - 1 == 10 and te[0] - va[-1] - 1 == 10
    assert te[-1] == 999


def test_purge_that_empties_a_block_is_rejected():
    with pytest.raises(AssertionError):
        trajectory_block_split(100, purge=10)


def _silver(tmp_path, n=1000):
    rng = np.random.default_rng(0)
    path = tmp_path / "silver.npz"
    np.savez_compressed(path, z=np.array([6, 1, 1]), R=rng.random((n, 3, 3)), E=rng.random(n),
                        F=rng.random((n, 3, 3)), molecule="x", theory="dft",
                        has_literature_split=False, literature_test_mask=np.zeros(n, bool))
    return path


def test_gold_file_records_its_split_provenance(tmp_path):
    import json
    from ml.data.preprocessing import PURGE_BUFFER_FRAMES, silver_to_gold
    g = np.load(silver_to_gold(_silver(tmp_path), tmp_path / "uracil_dft.npz"), allow_pickle=False)
    for key in ("source_sha256", "split_method", "split_params", "git_commit"):
        assert key in g.files
    assert str(g["split_method"]) == "trajectory_block"
    assert json.loads(str(g["split_params"]))["purge_frames"] == PURGE_BUFFER_FRAMES
    assert g["val_idx"][0] - g["train_idx"][-1] - 1 == PURGE_BUFFER_FRAMES


def test_exempt_gold_file_keeps_the_unpurged_split(tmp_path):
    import json
    from ml.data.preprocessing import silver_to_gold
    g = np.load(silver_to_gold(_silver(tmp_path), tmp_path / "ethanol_dft.npz"), allow_pickle=False)
    for got, expected in zip((g["train_idx"], g["val_idx"], g["test_idx"]), trajectory_block_split(1000)):
        assert np.array_equal(got, expected)
    params = json.loads(str(g["split_params"]))
    assert params["purge_frames"] == 0 and params["purge_exemption"]


def test_source_hash_is_stable_across_regenerations(tmp_path):
    from ml.data.preprocessing import silver_to_gold
    silver = _silver(tmp_path)
    a = np.load(silver_to_gold(silver, tmp_path / "a" / "uracil_dft.npz"))["source_sha256"]
    b = np.load(silver_to_gold(silver, tmp_path / "b" / "uracil_dft.npz"))["source_sha256"]
    assert str(a) == str(b)
