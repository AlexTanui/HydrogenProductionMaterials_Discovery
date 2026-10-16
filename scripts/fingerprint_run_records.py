"""Add scoring-data fingerprints to run records scored before evaluate.py recorded them.

For every experiments/runs/<name>/test_metrics.json without `data.fingerprint`:

1. re-score experiments/checkpoints/<name>.pt with
   `evaluate_checkpoint` - the stored split, seed and batch size;
2. compare every stored metric with the re-score, relative difference
   |fresh - stored| / |stored|;
3. only if **every metric of every run** is within
   `evaluate.REPRODUCIBILITY_RTOL` (1e-5 relative), write the fresh
   fingerprint into `data.fingerprint` and the measured differences into a
   `rescore` block. Otherwise nothing is written and the script exits 1.

The stored metrics are never changed. Re-scoring reproduces numbers that
were already reported and selects or tunes nothing, so the test split is
still scored once in the sense that matters: no decision is made from it.
Scoring on a multithreaded CPU is not bit-reproducible (the force backward
pass sums in a thread-dependent order), which is why the comparison uses a
tolerance and records the actual differences rather than claiming identity.

    python scripts/fingerprint_run_records.py            # dry run: report only
    python scripts/fingerprint_run_records.py --write    # add fingerprints
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterator

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch  # noqa: E402

from ml.training import evaluate  # noqa: E402


def _numeric_leaves(tree: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """(dotted path, value) for every int/float leaf of a nested dict/list."""
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from _numeric_leaves(value, f"{prefix}{key}.")
    elif isinstance(tree, list):
        for index, value in enumerate(tree):
            yield from _numeric_leaves(value, f"{prefix}{index}.")
    elif isinstance(tree, (int, float)) and not isinstance(tree, bool):
        yield prefix.rstrip("."), tree


def compare(stored: dict, fresh: dict) -> dict[str, float]:
    """Relative difference of every numeric metric: the metrics tree plus the benchmark row."""
    stored_leaves = dict(_numeric_leaves({"metrics": stored["metrics"],
                                          "benchmark_row": stored["benchmark_row"]}))
    fresh_leaves = dict(_numeric_leaves({"metrics": fresh["metrics"],
                                         "benchmark_row": fresh["benchmark_row"]}))
    if stored_leaves.keys() != fresh_leaves.keys():
        raise ValueError(f"metric fields differ: {sorted(stored_leaves.keys() ^ fresh_leaves.keys())}")
    differences = {}
    for path, old in stored_leaves.items():
        new = fresh_leaves[path]
        differences[path] = 0.0 if old == new else abs(new - old) / abs(old)
    return differences


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT_DIR, capture_output=True, text=True,
                          check=True).stdout.strip()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", default=str(ROOT_DIR / "experiments" / "runs"))
    parser.add_argument("--checkpoints-dir", default=str(ROOT_DIR / "experiments" / "checkpoints"))
    parser.add_argument("--write", action="store_true", help="write the fingerprints (default: dry run)")
    args = parser.parse_args(argv)

    pending = []
    for run_dir in sorted(Path(args.runs_dir).iterdir()):
        path = run_dir / "test_metrics.json"
        if not path.is_file():
            continue
        stored = json.loads(path.read_text(encoding="utf-8"))
        if "fingerprint" in stored["data"]:
            print(f"{run_dir.name}: already fingerprinted, skipped")
            continue
        fresh = evaluate.evaluate_checkpoint(
            Path(args.checkpoints_dir) / f"{run_dir.name}.pt",
            split=stored["data"]["split"],
            batch_size=stored["timing"]["batch_size"],
            seed=stored["seed"],
        )
        differences = compare(stored, fresh)
        worst = max(differences.values())
        within = worst <= evaluate.REPRODUCIBILITY_RTOL
        print(f"{run_dir.name}: {len(differences)} metrics, max relative difference {worst:.2e} "
              f"({'within' if within else 'OUTSIDE'} {evaluate.REPRODUCIBILITY_RTOL:g})")
        pending.append((path, stored, fresh, differences, within))

    if not pending:
        print("nothing to fingerprint")
        return 0
    if not all(within for *_, within in pending):
        print("Not every metric reproduced within tolerance - nothing written.")
        return 1
    if not args.write:
        print("Dry run - pass --write to add the fingerprints.")
        return 0

    provenance = {
        "date": date.today().isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "evaluate_py_sha256": _file_sha256(ROOT_DIR / "ml" / "training" / "evaluate.py"),
        "script": "scripts/fingerprint_run_records.py",
        "torch_num_threads": torch.get_num_threads(),
    }
    for path, stored, fresh, differences, _ in pending:
        stored["data"]["fingerprint"] = fresh["data"]["fingerprint"]
        stored["rescore"] = {
            "purpose": "add data.fingerprint; reproduces the stored metrics and tunes nothing",
            **provenance,
            "seed": fresh["seed"],
            "batch_size": fresh["timing"]["batch_size"],
            "tolerance_relative": evaluate.REPRODUCIBILITY_RTOL,
            "max_relative_difference": max(differences.values()),
            "relative_differences": differences,
            "stored_metrics_unchanged": True,
        }
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(stored, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
