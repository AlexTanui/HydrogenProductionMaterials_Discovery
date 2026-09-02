"""Tests for the bronze -> silver -> gold pipeline (ml/data/preprocessing.py).

Everything here runs against synthetic fixtures built in a tmp_path, so
the suite needs none of the ~814MB of real bronze data and stays fast
enough to run on every commit. The fixtures deliberately mimic the real
source quirks: MD22-shaped arrays, bytes-typed npz metadata, and a
literature-split zip.

    pytest tests/test_preprocessing.py
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

from ml.data import preprocessing as pp
from ml.data.md17 import MD17Sample, _as_str, _infer_molecule


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def make_npz(path, n_configs=100, n_atoms=8, bytes_units=False, seed=0):
    """Writes a bronze-shaped .npz. `bytes_units` reproduces the sources
    that store metadata as bytes rather than numpy str_."""
    rng = np.random.default_rng(seed)
    z = rng.integers(1, 9, size=n_atoms)
    arrays = dict(
        z=z,
        R=rng.normal(size=(n_configs, n_atoms, 3)).astype(np.float32),
        E=rng.normal(size=(n_configs,)).astype(np.float32),
        F=rng.normal(size=(n_configs, n_atoms, 3)).astype(np.float32),
        r_unit=b"Ang" if bytes_units else "Ang",
        e_unit=b"kcal/mol" if bytes_units else "kcal/mol",
    )
    np.savez(path, **arrays)
    return path


def make_split_zip(path, n_train=40, n_test=10, n_atoms=8):
    """Writes a zip holding train/test .npz members, like the CCSD sources."""
    with zipfile.ZipFile(path, "w") as zf:
        for seed, (name, n) in enumerate((("train", n_train), ("test", n_test))):
            buf = io.BytesIO()
            make_npz(buf, n_configs=n, n_atoms=n_atoms, seed=seed)
            zf.writestr(f"molecule-{name}.npz", buf.getvalue())
        # macOS resource-fork junk that must be ignored, not loaded
        zf.writestr("__MACOSX/._molecule-train.npz", b"\x00\x05\x16\x07not real")
    return path


# --------------------------------------------------------------------------
# metadata handling
# --------------------------------------------------------------------------

@pytest.mark.parametrize("filename,expected", [
    ("md17_ethanol.npz", "ethanol"),
    ("md17_uracil.npz", "uracil"),
    ("md22_DHA.npz", "DHA"),
    ("md22_AT-AT-CG-CG.npz", "AT-AT-CG-CG"),
    ("md22_double-walled_nanotube.npz", "double-walled_nanotube"),
    ("benzene2018_dft.npz", "benzene2018"),
])
def test_infer_molecule_strips_dataset_prefix(filename, expected):
    """md22_ was not stripped before, leaving molecules named 'md22_DHA'."""
    assert _infer_molecule(filename) == expected


@pytest.mark.parametrize("raw,expected", [
    (b"Ang", "Ang"),
    ("Ang", "Ang"),
    (np.array(b"kcal/mol"), "kcal/mol"),
    (np.array("kcal/mol"), "kcal/mol"),
])
def test_as_str_decodes_without_bytes_repr(raw, expected):
    """str(np.array(b'Ang')) is \"b'Ang'\", which fails the unit check."""
    assert _as_str(raw) == expected


def test_bytes_typed_units_pass_validation(tmp_path):
    """A source storing units as bytes must not fail the unit check."""
    path = make_npz(tmp_path / "md22_DHA.npz", bytes_units=True)
    from ml.data.md17 import load_npz
    sample = load_npz(path, molecule="DHA", theory="dft")
    assert sample.r_unit == "Ang"
    assert pp.validate_sample(sample) == []


# --------------------------------------------------------------------------
# bronze file checks
# --------------------------------------------------------------------------

def test_lfs_pointer_is_reported_clearly(tmp_path):
    pointer = tmp_path / "md22_DHA.npz"
    pointer.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:62570396de06fabeb734b20cb7b9aa7e0379403fedc842c8cf6239a8b8a8d738\n"
        b"size 125691777\n"
    )
    with pytest.raises(ValueError, match="git-lfs pointer"):
        pp.check_bronze_file(pointer)


def test_missing_bronze_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pp.check_bronze_file(tmp_path / "nope.npz")


def test_real_file_passes_check(tmp_path):
    pp.check_bronze_file(make_npz(tmp_path / "md22_DHA.npz"))


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def test_validate_sample_catches_nans_and_bad_units():
    z = np.array([1, 6, 8])
    R = np.zeros((5, 3, 3), dtype=np.float32)
    E = np.zeros(5, dtype=np.float32)
    F = np.zeros((5, 3, 3), dtype=np.float32)

    assert pp.validate_sample(MD17Sample("x", "dft", z, R, E, F)) == []

    E_nan = E.copy()
    E_nan[2] = np.nan
    problems = pp.validate_sample(MD17Sample("x", "dft", z, R, E_nan, F))
    assert "NaN values in E" in problems

    problems = pp.validate_sample(MD17Sample("x", "dft", z, R, E, F, r_unit="bohr"))
    assert any("unexpected r_unit" in p for p in problems)


def test_shape_mismatch_is_rejected_at_construction():
    """E with fewer entries than R has configs must not load silently."""
    with pytest.raises(ValueError, match="entries"):
        MD17Sample("x", "dft", np.array([1, 6]), np.zeros((5, 2, 3), dtype=np.float32),
                   np.zeros(3, dtype=np.float32), np.zeros((5, 2, 3), dtype=np.float32))


# --------------------------------------------------------------------------
# splitting - the property that actually protects the science
# --------------------------------------------------------------------------

def test_block_split_is_contiguous_and_ordered():
    """Adjacent MD frames are near-identical, so the split must be
    contiguous blocks in trajectory order - a random split leaks
    near-duplicate structures into test and inflates accuracy."""
    train, val, test = pp.trajectory_block_split(1000)
    assert len(train) == 800 and len(val) == 100 and len(test) == 100
    assert train.max() < val.min() < test.min()
    assert np.array_equal(np.concatenate([train, val, test]), np.arange(1000))


def test_block_split_rejects_ratios_that_do_not_sum_to_one():
    with pytest.raises(AssertionError):
        pp.trajectory_block_split(100, train=0.8, val=0.3, test=0.1)


# --------------------------------------------------------------------------
# end-to-end, MD22 shapes
# --------------------------------------------------------------------------

def test_md22_npz_promotes_to_gold_with_block_split(tmp_path):
    """The MD22 path: plain npz in, no literature split, block split out."""
    bronze = make_npz(tmp_path / "md22_buckyball-catcher.npz",
                      n_configs=200, n_atoms=148, bytes_units=True)
    silver = pp.bronze_to_silver(bronze, tmp_path / "silver",
                                 molecule="buckyball-catcher", theory="dft")
    assert silver.name == "buckyball-catcher_dft.npz"
    assert not bool(np.load(silver, allow_pickle=True)["has_literature_split"])

    gold = pp.silver_to_gold(silver, tmp_path / "gold" / "buckyball-catcher_dft.npz")
    data = np.load(gold, allow_pickle=True)
    assert data["z"].shape == (148,)
    assert data["R"].shape == (200, 148, 3)
    assert not bool(data["used_literature_split"])
    assert len(data["train_idx"]) == 160
    assert data["train_idx"].max() < data["test_idx"].min()


def test_largest_md22_system_survives_the_pipeline(tmp_path):
    """370 atoms is 15x MD17's largest - shape handling must not assume small."""
    bronze = make_npz(tmp_path / "md22_double-walled_nanotube.npz",
                      n_configs=50, n_atoms=370)
    silver = pp.bronze_to_silver(bronze, tmp_path / "silver",
                                 molecule="double-walled-nanotube", theory="dft")
    gold = pp.silver_to_gold(silver, tmp_path / "gold" / "nanotube.npz")
    assert np.load(gold, allow_pickle=True)["F"].shape == (50, 370, 3)


def test_literature_split_is_preserved_not_resplit(tmp_path):
    """The MD17 CCSD sources ship a published split that must survive."""
    bronze = make_split_zip(tmp_path / "aspirin_ccsd.zip", n_train=40, n_test=10)
    silver = pp.bronze_to_silver(bronze, tmp_path / "silver",
                                 molecule="aspirin", theory="ccsd")
    data = np.load(silver, allow_pickle=True)
    assert bool(data["has_literature_split"])
    assert data["literature_test_mask"].sum() == 10

    gold = np.load(pp.silver_to_gold(silver, tmp_path / "gold" / "aspirin_ccsd.npz"),
                   allow_pickle=True)
    assert bool(gold["used_literature_split"])
    # the shipped 10 test frames stay the test set, untouched
    assert sorted(gold["test_idx"]) == list(range(40, 50))
    assert len(gold["train_idx"]) + len(gold["val_idx"]) == 40


def test_no_leakage_between_any_two_splits(tmp_path):
    bronze = make_npz(tmp_path / "md22_DHA.npz", n_configs=137, n_atoms=56)
    silver = pp.bronze_to_silver(bronze, tmp_path / "silver", molecule="DHA", theory="dft")
    gold = np.load(pp.silver_to_gold(silver, tmp_path / "gold" / "DHA_dft.npz"),
                   allow_pickle=True)
    train, val, test = set(gold["train_idx"]), set(gold["val_idx"]), set(gold["test_idx"])
    assert not (train & val) and not (train & test) and not (val & test)
    assert len(train | val | test) == 137


def test_run_dataset_processes_the_whole_md22_roster(tmp_path, monkeypatch):
    """run_dataset('md22') finds all four sources and writes gold for each."""
    monkeypatch.setattr(pp, "DATA_DIR", tmp_path)
    bronze_dir = tmp_path / "bronze" / "md22"
    bronze_dir.mkdir(parents=True)
    for filename, _molecule, _theory, _lit in pp.MD22_SOURCES:
        make_npz(bronze_dir / filename, n_configs=60, n_atoms=20, bytes_units=True)

    results = pp.run_dataset("md22")
    assert len(results) == len(pp.MD22_SOURCES)
    assert {m for m, _t, _p in results} == {m for _f, m, _t, _l in pp.MD22_SOURCES}
    for _molecule, _theory, gold_path in results:
        assert gold_path.exists()
        assert not bool(np.load(gold_path, allow_pickle=True)["used_literature_split"])


def test_run_dataset_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown dataset"):
        pp.run_dataset("md99")


def test_md22_sources_match_the_download_script():
    """The roster here and in scripts/download_bronze_data.sh must agree,
    or a file gets fetched and never cleaned (or vice versa)."""
    script = (pp.ROOT_DIR / "scripts" / "download_bronze_data.sh").read_text()
    for filename, _molecule, _theory, _lit in pp.MD22_SOURCES:
        assert f"md22/{filename}|" in script, f"{filename} missing from download script"


def test_no_md22_source_claims_a_literature_split():
    """MD22 papers quote a train-set size, not a shipped index split."""
    assert not any(lit for _f, _m, _t, lit in pp.MD22_SOURCES)
