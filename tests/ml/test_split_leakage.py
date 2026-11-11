"""Leakage checks on the real gold splits (SCRUM-47, part 3a).

The ticket's acceptance criteria, as tests:

1. **Nearest neighbour.** No held-out frame is a near-duplicate of a
   training frame: its aligned RMSD to the closest training frame must
   exceed what `NN_THRESHOLD_LAG` MD steps move the molecule.
2. **Random-split control.** A measurement, not a test - see
   docs/reviews/leakage-audit-2026-10-31.md.
3. **Structure.** Validation and test are contiguous blocks after train;
   the purge buffer between blocks meets the agreed size; the gold file
   records how its split was made; and the split can't change silently -
   its indices are pinned here by fingerprint, so a changed splitter fails
   loudly.

The nearest-neighbour threshold was approved on 31 Oct (2 MD steps). The
purge buffer is a proposal (10 frames for new splits, the existing ethanol
split kept), so its test skips with the measured gaps (`pytest -rs` shows
them) rather than pass or fail against a number not yet agreed. Keeping
the existing split means validation's first frame stays a near-duplicate
of the last training frame: that one case is a strict expected failure,
and the frames the benchmarked runs actually train and select on are
checked separately and pass. The full measurements are
reports/leakage/*.json, from scripts/leakage_audit.py.

Data-dependent tests skip when the gold file is missing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from ml.data.preprocessing import trajectory_block_split
from ml.utils.leakage import equivalent_atom_groups, lag_rmsd, nearest_neighbours, split_gaps

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "gold" / "md17"
# The train/val frames every benchmarked run trains and selects on
# (benchmark.py checks that all runs in a table share them).
USED_SUBSETS = ROOT / "experiments" / "runs" / "phase1_baseline" / "subsets.json"

# Pinned splits: (frames, (train, val, test) sizes, sha256 of the int64
# bytes of train_idx, val_idx, test_idx concatenated). A change here means
# every run on that gold file needs re-scoring - regenerate only in a
# commit that says why the split changed.
SPLIT_FINGERPRINTS = {
    "ethanol_dft": (555092, (444074, 55509, 55509),
                    "d515f0e9957e912fff507455c4b2f80b7d473728cea13e8ae80737fca7cfd4c0"),
    "ethanol_ccsd_t": (2000, (889, 111, 1000),
                       "55f385cf2332d9056aaed6f496e7bebd2df52c6a9547ce2144b309432d4b0290"),
}

# Evidence for both in docs/reviews/leakage-audit-2026-10-31.md.
# Proposed to the data-pipeline review on 31 Oct: 10 frames for new splits,
# the existing ethanol split kept. Not agreed yet, so None.
AGREED_PURGE_BUFFER_FRAMES: Optional[int] = None
# Approved 31 Oct: no held-out frame closer to a training frame than the
# median RMSD across 2 MD steps (0.028 A for ethanol DFT).
NN_THRESHOLD_LAG: Optional[int] = 2

# What a gold file should record about its own split (proposed key names).
PROVENANCE_KEYS = ("source_sha256", "split_method", "split_params", "git_commit")


def _gold(name: str):
    path = GOLD / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"{path} not present - run `python -m ml.data.preprocessing`")
    return np.load(path, allow_pickle=False)


def _fingerprint(*indices: np.ndarray) -> str:
    return hashlib.sha256(b"".join(np.ascontiguousarray(i, dtype=np.int64).tobytes() for i in indices)).hexdigest()


# --------------------------------------------------------------------------
# 3. Structure
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", SPLIT_FINGERPRINTS)
def test_blocks_are_disjoint_and_cover_every_frame(name):
    d = _gold(name)
    tr, va, te = d["train_idx"], d["val_idx"], d["test_idx"]
    together = np.concatenate([tr, va, te])
    assert len(np.unique(together)) == len(together), "a frame is in more than one split"
    assert np.array_equal(np.sort(together), np.arange(len(d["R"]))), "not every frame is in a split"


@pytest.mark.parametrize("name", SPLIT_FINGERPRINTS)
def test_val_and_test_are_contiguous_blocks_after_train(name):
    d = _gold(name)
    tr, va, te = d["train_idx"], d["val_idx"], d["test_idx"]
    for split, idx in (("train", tr), ("val", va), ("test", te)):
        assert np.all(np.diff(idx) == 1), f"{name}: {split} is not one contiguous block"
    assert tr.max() < va.min() and va.max() < te.min(), f"{name}: blocks are not in train < val < test order"


@pytest.mark.parametrize("name", SPLIT_FINGERPRINTS)
def test_split_indices_match_their_pinned_fingerprint(name):
    d = _gold(name)
    n_frames, sizes, fingerprint = SPLIT_FINGERPRINTS[name]
    tr, va, te = d["train_idx"], d["val_idx"], d["test_idx"]
    assert len(d["R"]) == n_frames
    assert (len(tr), len(va), len(te)) == sizes
    assert _fingerprint(tr, va, te) == fingerprint, (
        f"{name}'s split changed. Every run trained or scored on it is now on different frames: "
        "re-score them, and update SPLIT_FINGERPRINTS in a commit that says why."
    )


def test_block_split_gold_file_is_exactly_what_the_splitter_produces():
    d = _gold("ethanol_dft")
    assert not bool(d["used_literature_split"])
    for got, expected in zip((d["train_idx"], d["val_idx"], d["test_idx"]), trajectory_block_split(len(d["R"]))):
        assert np.array_equal(got, expected)


def test_trajectory_block_split_has_not_changed():
    """Pins the splitter itself, without data: 80/10/10 contiguous blocks,
    back to back. Adding a purge buffer will (rightly) fail this - update
    it, and SPLIT_FINGERPRINTS, in the same commit."""
    tr, va, te = trajectory_block_split(555092)
    assert (len(tr), len(va), len(te)) == (444074, 55509, 55509)
    assert (tr[0], tr[-1], va[0], va[-1], te[0], te[-1]) == (0, 444073, 444074, 499582, 499583, 555091)
    assert split_gaps(tr, va, te) == {"train_to_val": 0, "val_to_test": 0, "train_to_test": 55509}


@pytest.mark.parametrize("name", SPLIT_FINGERPRINTS)
def test_purge_buffer_meets_the_agreed_size(name):
    d = _gold(name)
    gaps = split_gaps(d["train_idx"], d["val_idx"], d["test_idx"])
    if AGREED_PURGE_BUFFER_FRAMES is None:
        pytest.skip(f"purge buffer not agreed yet (10 frames proposed for new splits, existing ethanol "
                    f"split kept); {name} gaps: {gaps}")
    for boundary in ("train_to_val", "val_to_test"):
        assert gaps[boundary] is not None and gaps[boundary] >= AGREED_PURGE_BUFFER_FRAMES, (
            f"{name}: {gaps[boundary]} frames between blocks at {boundary}, "
            f"agreed buffer is {AGREED_PURGE_BUFFER_FRAMES}"
        )


@pytest.mark.xfail(strict=True, reason=(
    "SCRUM-47 finding: gold files record no source hash, split method/parameters or git commit. "
    "Fix in ml/data/preprocessing.py (silver_to_gold), then delete this marker - strict, so a "
    "fixed file turns this into a failure until it is."))
@pytest.mark.parametrize("name", SPLIT_FINGERPRINTS)
def test_gold_file_records_how_its_split_was_made(name):
    d = _gold(name)
    missing = [key for key in PROVENANCE_KEYS if key not in d.files]
    assert not missing, f"{name} records no {', '.join(missing)}"


# --------------------------------------------------------------------------
# 1. Nearest neighbour
# --------------------------------------------------------------------------

def _assert_no_near_duplicates(d, held: np.ndarray, reference: np.ndarray, label: str) -> None:
    """Probes the held-out frames nearest the reference in time (where a
    near-duplicate would be) plus an even spread over the rest; the full
    measurement over every frame is reports/leakage/ethanol_dft.json."""
    R, train_idx = d["R"], d["train_idx"]
    probe = np.unique(np.concatenate([held[:20], held[:: max(1, len(held) // 200)]]))
    groups = equivalent_atom_groups(d["z"], R[0])
    nn, _ = nearest_neighbours(R[probe], R[reference], groups, k=512)
    lags = lag_rmsd(R[train_idx], (1, 2, 3, 5, 10), groups, n_samples=1000)
    closest = int(probe[nn.argmin()])
    measured = (f"closest {label} frame {closest} is {nn.min():.4f} A away; lag medians "
                + ", ".join(f"{lag}: {np.median(v):.4f}" for lag, v in lags.items()))
    if NN_THRESHOLD_LAG is None:
        pytest.skip(f"threshold not approved yet; {measured}")
    threshold = float(np.median(lag_rmsd(R[train_idx], (NN_THRESHOLD_LAG,), groups, n_samples=1000)[NN_THRESHOLD_LAG]))
    assert nn.min() >= threshold, (
        f"{label} frame {closest} is a near-duplicate of a training frame: {nn.min():.4f} A, closer "
        f"than {NN_THRESHOLD_LAG} MD steps ({threshold:.4f} A). {measured}"
    )


@pytest.mark.parametrize("held_out", [
    "test",
    pytest.param("val", marks=pytest.mark.xfail(strict=True, reason=(
        "Known and kept: with no purge buffer, val frame 444074 is one MD step (0.0148 A) from "
        "train frame 444073. The existing ethanol split is kept (31 Oct) because no benchmarked "
        "run trains or selects on those frames - see the 'actually used' test below. Strict, so "
        "a re-split with a buffer makes this fail until the marker is removed."))),
])
def test_no_held_out_frame_is_a_near_duplicate_of_a_training_frame(held_out):
    d = _gold("ethanol_dft")
    _assert_no_near_duplicates(d, d[f"{held_out}_idx"], d["train_idx"], f"{held_out} (whole block)")


@pytest.mark.parametrize("held_out", ["val", "test"])
def test_no_held_out_frame_is_a_near_duplicate_of_the_training_frames_actually_used(held_out):
    """What the benchmarked runs saw: the train and val stride subsets
    (their uncovered tails leave 4,114 frames between the used train and
    val frames, and 519 between the used val frames and test)."""
    d = _gold("ethanol_dft")
    if not USED_SUBSETS.exists():
        pytest.skip(f"{USED_SUBSETS} not present")
    subsets = json.loads(USED_SUBSETS.read_text(encoding="utf-8"))
    train_used = np.asarray(subsets["train"]["gold_indices"])
    held = np.asarray(subsets["val"]["gold_indices"]) if held_out == "val" else d["test_idx"]
    _assert_no_near_duplicates(d, held, train_used, f"{held_out} (used)")
