"""Build the random-split control for the SCRUM-47 leakage audit (part 3b).

    python scripts/make_random_split_control.py            # ethanol_dft, seed 0

Writes a gold-format copy of data/gold/md17/ethanol_dft.npz whose
train/val/test indices are a *random frame-level* split - deliberately
the leaky split CLAUDE.md forbids - so that training the Phase 1 baseline
on it, with everything else unchanged, measures how much leakage would
inflate the reported MAE. It is a control, never a result.

Where it goes, and why there: data/external/leakage_controls/ is
gitignored (data/external/**), and the backend only scans
data/{bronze,silver,gold}/md17, so the file can't be picked up as a
dataset by the app, and it isn't under data/gold/ where training reads
real splits. The config that trains on it is
experiments/configs/leakctl_mpnn_random_split.yaml (run name not
phase1_*, so it can't enter the Phase 1 table; its scoring-data
fingerprint differs anyway, so benchmark.py would refuse to mix it).

How the split is made, exactly:
- source: the gold file's z, R, E, F, unchanged (sha256 recorded);
- permutation = numpy.random.default_rng(seed).permutation(n_frames);
- train = sorted(permutation[:n_train]), val = sorted(next n_val),
  test = sorted(the rest), with n_train/n_val/n_test equal to the gold
  file's own block sizes (444,074 / 55,509 / 55,509 for ethanol_dft).
The same facts go into the file itself (control_* keys) and into a JSON
next to it, with the git commit this script ran at.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT_DIR / "data" / "gold" / "md17" / "ethanol_dft.npz"
DEFAULT_OUT = ROOT_DIR / "data" / "external" / "leakage_controls" / "ethanol_dft_random_split.npz"


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.run(["git", *args], cwd=ROOT_DIR, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def random_split(n_frames: int, sizes: tuple[int, int, int], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorted train/val/test frame indices, drawn uniformly at random."""
    if sum(sizes) != n_frames:
        raise ValueError(f"split sizes {sizes} don't add up to {n_frames} frames")
    permutation = np.random.default_rng(seed).permutation(n_frames)
    n_train, n_val, _ = sizes
    return (np.sort(permutation[:n_train]), np.sort(permutation[n_train:n_train + n_val]),
            np.sort(permutation[n_train + n_val:]))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    gold = np.load(args.source, allow_pickle=False)
    sizes = tuple(len(gold[f"{s}_idx"]) for s in ("train", "val", "test"))
    train_idx, val_idx, test_idx = random_split(len(gold["R"]), sizes, args.seed)
    assert set(train_idx).isdisjoint(val_idx) and set(train_idx).isdisjoint(test_idx) \
        and set(val_idx).isdisjoint(test_idx)

    record = {
        "purpose": "SCRUM-47 leakage control: random frame-level split (deliberately leaky). Not a result.",
        "built_by": "scripts/make_random_split_control.py",
        "date": date.today().isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if _git("status", "--porcelain") is None else bool(_git("status", "--porcelain")),
        "source": str(args.source.resolve().relative_to(ROOT_DIR)).replace("\\", "/"),
        "source_sha256": _sha256(args.source),
        "method": "numpy.random.default_rng(seed).permutation(n_frames); first n_train -> train, "
                  "next n_val -> val, rest -> test; each sorted",
        "seed": args.seed,
        "sizes": {"train": sizes[0], "val": sizes[1], "test": sizes[2]},
        "indices_sha256": hashlib.sha256(b"".join(np.ascontiguousarray(i, dtype=np.int64).tobytes()
                                                  for i in (train_idx, val_idx, test_idx))).hexdigest(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out, z=gold["z"], R=gold["R"], E=gold["E"], F=gold["F"],
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        molecule=str(gold["molecule"]), theory=str(gold["theory"]), used_literature_split=False,
        control_purpose=record["purpose"], control_method=record["method"], control_seed=args.seed,
        control_source_sha256=record["source_sha256"], control_git_commit=record["git_commit"] or "",
    )
    record["sha256"] = _sha256(args.out)
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
