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
