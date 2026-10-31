"""Split-leakage audit of one gold file (SCRUM-47): structure, scale, nearest neighbours.

    python scripts/leakage_audit.py data/gold/md17/ethanol_dft.npz
    python scripts/leakage_audit.py data/gold/md17/ethanol_ccsd_t.npz --run none

Writes reports/leakage/<molecule>_<theory>.json (override with --out). Reads
the gold file and, for "frames actually used", one run record's
subsets.json (default experiments/runs/phase1_baseline: every benchmarked
run shares its train/val frames - benchmark.py checks that). Changes
nothing else. Method and distance: ml/utils/leakage.py.

What it records:
- **split**: block sizes, index ranges, gaps between blocks, whether the
  file's split is exactly what `trajectory_block_split` produces today,
  and which provenance keys the gold file carries;
- **time order**: lag-1 RMSD against random-pair RMSD - the lag scale only
  means something if frame order is trajectory order;
- **lag_rmsd**: median/p5/p95 RMSD at frame lags (what a near-duplicate is);
- **nearest neighbours**: min / p1 / p5 / median / max RMSD from each query
  frame to its closest reference frame, for test->train, val->train,
  test->val, and against the frames the runs actually trained and
  selected on; plus the closest pair and its distance in frame index;
- **validation**: the shortlist checked against brute force on a sample;
- **reflection**: the same nearest-neighbour distances allowing mirror
  images, on a sample (a distance-based model can't tell them apart).

Thresholds are deliberately not applied here: this measures, and
docs/reviews/leakage-audit-2026-10-31.md proposes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ml.data.preprocessing import trajectory_block_split  # noqa: E402
from ml.utils import leakage  # noqa: E402

LAGS = (1, 2, 3, 5, 10, 20, 50, 100, 1000, 10000, 100000)
PROVENANCE_KEYS = ("source_sha256", "split_method", "split_params", "git_commit")


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


def _stats(values: np.ndarray) -> dict[str, float]:
    return {"n": int(len(values)), "min": float(values.min()), "p1": float(np.percentile(values, 1)),
            "p5": float(np.percentile(values, 5)), "median": float(np.median(values)),
            "max": float(values.max())}


def _nn_block(name: str, R: np.ndarray, query_idx: np.ndarray, ref_idx: np.ndarray, groups, k: int,
              lag_medians: dict[int, float]) -> dict[str, Any]:
    t0 = time.perf_counter()
    rmsd, pos = leakage.nearest_neighbours(R[query_idx], R[ref_idx], groups, k=k)
    match = ref_idx[pos]
    i = int(rmsd.argmin())
    block = {
        **_stats(rmsd),
        "closest_pair": {"query_frame": int(query_idx[i]), "reference_frame": int(match[i]),
                         "rmsd": float(rmsd[i]), "frames_apart": int(abs(query_idx[i] - match[i]))},
        "below_lag_median": {str(lag): int((rmsd < m).sum()) for lag, m in lag_medians.items()
                             if lag <= 20},
        "first_query_frames": [{"query_frame": int(query_idx[j]), "rmsd": float(rmsd[j]),
                                "reference_frame": int(match[j])} for j in range(min(5, len(rmsd)))],
        "seconds": round(time.perf_counter() - t0, 1),
    }
    print(f"  {name:28s} min {block['min']:.4f}  p1 {block['p1']:.4f}  median {block['median']:.4f}  "
          f"[{block['seconds']}s]", flush=True)
    return block


def audit(gold: Path, run: Optional[Path], k: int, n_validate: int, seed: int) -> dict[str, Any]:
    start = time.perf_counter()
    d = np.load(gold)
    z, R = d["z"], d["R"]
    tr, va, te = d["train_idx"], d["val_idx"], d["test_idx"]
    molecule, theory = str(d["molecule"]), str(d["theory"])
    groups = leakage.equivalent_atom_groups(z, R[0])
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(R), size=min(3000, len(R)), replace=False)
    print(f"{gold.name}: {len(R)} frames; train {len(tr)} / val {len(va)} / test {len(te)}", flush=True)

    expected = trajectory_block_split(len(R))
    split = {
        "sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "ranges": {name: [int(idx.min()), int(idx.max())] for name, idx in (("train", tr), ("val", va), ("test", te))},
        "contiguous": {name: bool(np.all(np.diff(idx) == 1)) for name, idx in (("train", tr), ("val", va), ("test", te))},
        "gaps": leakage.split_gaps(tr, va, te),
        "used_literature_split": bool(d["used_literature_split"]),
        "matches_trajectory_block_split": bool(all(np.array_equal(a, b) for a, b in zip((tr, va, te), expected))),
        "keys_in_file": sorted(d.files),
        "provenance_keys_present": [key for key in PROVENANCE_KEYS if key in d.files],
        "indices_sha256": hashlib.sha256(b"".join(np.ascontiguousarray(i, dtype=np.int64).tobytes()
                                                  for i in (tr, va, te))).hexdigest(),
    }

    # Scale: lag RMSD inside the train block; random pairs for context.
    lags = leakage.lag_rmsd(R[tr], LAGS, groups, n_samples=2000, seed=seed)
    a, b = tr[rng.integers(0, len(tr), 2000)], tr[rng.integers(0, len(tr), 2000)]
    random_pairs = leakage.min_rmsd_over_permutations(
        torch.as_tensor(R[a], dtype=torch.float64), torch.as_tensor(R[b], dtype=torch.float64)[:, None],
        torch.as_tensor(leakage.group_permutations(groups, R.shape[1])))[0].numpy()
    in_order = float(np.median(lags[1])) < 0.25 * float(np.median(random_pairs))
    lag_block = {str(lag): {"median": float(np.median(v)), "p5": float(np.percentile(v, 5)),
                            "p95": float(np.percentile(v, 95))} for lag, v in lags.items()}
    lag_medians = {lag: float(np.median(v)) for lag, v in lags.items()}
    print(f"  lag-1 median {lag_medians[1]:.4f}, random pairs median {np.median(random_pairs):.4f} "
          f"-> trajectory order: {in_order}", flush=True)

    nn = {
        "test_to_train": _nn_block("test -> train", R, te, tr, groups, k, lag_medians),
        "val_to_train": _nn_block("val -> train", R, va, tr, groups, k, lag_medians),
        "test_to_val": _nn_block("test -> val", R, te, va, groups, k, lag_medians),
    }
    used = None
    if run is not None:
        subsets = json.loads((run / "subsets.json").read_text(encoding="utf-8"))
        tr_used = np.asarray(subsets["train"]["gold_indices"])
        va_used = np.asarray(subsets["val"]["gold_indices"])
        used = {"run": str(run.relative_to(ROOT_DIR)).replace("\\", "/"),
                "train_used": len(tr_used), "val_used": len(va_used)}
        nn["test_to_train_used"] = _nn_block("test -> train (used)", R, te, tr_used, groups, k, lag_medians)
        nn["val_used_to_train_used"] = _nn_block("val (used) -> train (used)", R, va_used, tr_used, groups, k,
                                                 lag_medians)

    # Shortlist vs brute force: random test and val frames, plus the val
    # frames next to the train block (where the closest matches are).
    t0 = time.perf_counter()
    half = n_validate // 2
    probe = np.concatenate([rng.choice(te, half, replace=False), rng.choice(va, half - 3, replace=False), va[:3]])
    fast, _ = leakage.nearest_neighbours(R[probe], R[tr], groups, k=k)
    exact = np.array([leakage.exact_nearest(R[i], R[tr], groups)[0] for i in probe])
    validation = {"queries": int(len(probe)), "reference": "train", "k": k,
                  "exact_matches": int((np.abs(fast - exact) <= 1e-9).sum()),
                  "worst_excess": float(np.max(fast - exact)), "seconds": round(time.perf_counter() - t0, 1)}
    print(f"  shortlist vs brute force: {validation['exact_matches']}/{validation['queries']} exact", flush=True)

    # Reflection sensitivity on a sample of test frames.
    refl_q = rng.choice(te, min(2000, len(te)), replace=False)
    proper, _ = leakage.nearest_neighbours(R[refl_q], R[tr], groups, k=k)
    mirrored, _ = leakage.nearest_neighbours(R[refl_q], R[tr], groups, k=k, allow_reflection=True)
    reflection = {"queries": int(len(refl_q)), "median_proper": float(np.median(proper)),
                  "median_with_reflection": float(np.median(mirrored)),
                  "min_proper": float(proper.min()), "min_with_reflection": float(mirrored.min())}

    return {
        "schema_version": 1,
        "generated_by": "scripts/leakage_audit.py",
        "date": date.today().isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if _git("status", "--porcelain") is None else bool(_git("status", "--porcelain")),
        "gold": {"path": str(gold.relative_to(ROOT_DIR)).replace("\\", "/") if gold.is_relative_to(ROOT_DIR) else str(gold),
                 "sha256": _sha256(gold), "molecule": molecule, "theory": theory, "n_frames": int(len(R))},
        "metric": {"distance": "Kabsch-aligned RMSD (proper rotations), min over equivalent-H relabellings",
                   "units": "angstrom", "equivalent_groups": [g.tolist() for g in groups],
                   "relabellings": int(len(leakage.group_permutations(groups, R.shape[1]))),
                   "groups_stable_on_sample": bool(leakage.check_groups_stable(z, R[sample], groups)),
                   "shortlist_k": k, "seed": seed},
        "split": split,
        "time_order": {"lag1_median": lag_medians[1], "random_pair_median": float(np.median(random_pairs)),
                       "in_trajectory_order": in_order},
        "lag_rmsd": lag_block,
        "frames_used": used,
        "nearest_neighbours": nn,
        "validation": validation,
        "reflection": reflection,
        "seconds": round(time.perf_counter() - start, 1),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gold", type=Path)
    parser.add_argument("--run", default="experiments/runs/phase1_baseline",
                        help="run record whose subsets.json gives the frames actually used, or 'none'")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--validate", type=int, default=20, help="brute-force checks (about 3 s each)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    gold = args.gold if args.gold.is_absolute() else (ROOT_DIR / args.gold)
    run = None if args.run == "none" else ROOT_DIR / args.run
    result = audit(gold.resolve(), run, args.k, args.validate, args.seed)
    out = args.out or ROOT_DIR / "reports" / "leakage" / f"{result['gold']['molecule']}_{result['gold']['theory']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({result['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
