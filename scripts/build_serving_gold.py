"""Build the test-split-only gold copy the hosted backend serves from.

The hosted (Vercel) backend only ever reads each gold file's *test* split:
predictions index into MD17Dataset(split="test") and the molecule catalog
only reads z/molecule/theory. Vercel also rejects any uploaded file over
100 MB, and data/gold/md17/ethanol_dft.npz is ~110 MB (already compressed).
So this writes deploy/gold/md17/{molecule}_{theory}.npz holding just the
test frames, in their original order, with test_idx = arange(n_test) and
empty train/val indices. MD17Dataset(path, split="test") then yields
exactly the same frames, by the same frame_index, as the full gold file -
asserted below for every file before anything is written.

The real gold files are never modified. deploy/ is gitignored; re-run this
before `vercel deploy` whenever data/gold changes:

    python scripts/build_serving_gold.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "gold" / "md17"
DST = ROOT / "deploy" / "gold" / "md17"
_FRAME_KEYS = ("R", "E", "F")


def build_one(src: Path, dst: Path) -> tuple[int, int]:
    with np.load(src, allow_pickle=True) as data:
        full = {k: data[k] for k in data.files}
    test_idx = full["test_idx"]
    out = {k: v for k, v in full.items() if k not in _FRAME_KEYS and not k.endswith("_idx")}
    for k in _FRAME_KEYS:
        out[k] = full[k][test_idx]
    out["test_idx"] = np.arange(len(test_idx), dtype=np.int64)
    out["train_idx"] = np.zeros(0, dtype=np.int64)
    out["val_idx"] = np.zeros(0, dtype=np.int64)
    out["serving_subset_of"] = np.array(src.name)

    # Same frames, same order, as the full file's test split.
    for k in _FRAME_KEYS:
        assert np.array_equal(out[k][out["test_idx"]], full[k][test_idx]), (src.name, k)
    np.savez_compressed(dst, **out)
    return len(test_idx), dst.stat().st_size


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    for src in sorted(SRC.glob("*.npz")):
        n, size = build_one(src, DST / src.name)
        print(f"{src.name}: {n} test frames, {size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
