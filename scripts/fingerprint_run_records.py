"""Independently re-score every run record and add/verify its scoring-data fingerprint (SCRUM-71).

For every experiments/runs/<name>/test_metrics.json:

1. re-score experiments/checkpoints/<name>.pt with
   `evaluate_checkpoint` - the stored split, seed and batch size;
2. compare every stored metric with the re-score, relative difference
   |fresh - stored| / |stored|;
3. only if **every metric of every run** is within
   `evaluate.REPRODUCIBILITY_RTOL` (1e-5 relative), write the measured
   differences into a `rescore` block (plus `data.fingerprint`, if the
   record predates evaluate.py recording one). Otherwise nothing is
   written and the script exits 1.

**Every run is re-scored, not just ones missing a fingerprint** (SCRUM-71 -
the original behaviour here skipped any run that already had one, which
meant no real checkpoint was ever actually independently re-run and
diffed, only legacy records got a fingerprint backfilled). A record that
already has a fingerprint gets its fingerprint verified (the freshly
re-scored data must hash to the same value - a mismatch means the scoring
data itself changed since the record was written, and is reported as an
error, not silently overwritten) and its `rescore` block added or
refreshed. A record with no fingerprint yet (legacy) gets both: the
fingerprint backfilled and the rescore block added, exactly as before.

The stored metrics are never changed. Re-scoring reproduces numbers that
were already reported and selects or tunes nothing, so the test split is
still scored once in the sense that matters: no decision is made from it.
Scoring on a multithreaded CPU is not bit-reproducible (the force backward
pass sums in a thread-dependent order), which is why the comparison uses a
tolerance and records the actual differences rather than claiming identity.

    python scripts/fingerprint_run_records.py            # dry run: report only
    python scripts/fingerprint_run_records.py --write    # add fingerprints + rescore blocks
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
    """Relative difference of every numeric metric the stored record holds:
    the metrics tree plus the benchmark row.

    Only stored fields are compared: the question is whether a re-score
    reproduces what was reported. A newer evaluate.py can record fields an
    older record predates (e.g. per-quantity UQ keys in the benchmark row);
    those are new, not a mismatch. A stored field the re-score no longer
    produces is one, and raises.
    """
    stored_leaves = dict(_numeric_leaves({"metrics": stored["metrics"],
                                          "benchmark_row": stored["benchmark_row"]}))
    fresh_leaves = dict(_numeric_leaves({"metrics": fresh["metrics"],
                                         "benchmark_row": fresh["benchmark_row"]}))
    missing = sorted(stored_leaves.keys() - fresh_leaves.keys())
    if missing:
        raise ValueError(f"the re-score no longer produces stored metric fields: {missing}")
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
    fingerprint_mismatches = []
    for run_dir in sorted(Path(args.runs_dir).iterdir()):
        path = run_dir / "test_metrics.json"
        if not path.is_file():
            continue
        stored = json.loads(path.read_text(encoding="utf-8"))
        had_fingerprint = "fingerprint" in stored["data"]
        checkpoint_path = Path(args.checkpoints_dir) / f"{run_dir.name}.pt"
        if not checkpoint_path.is_file():
            # experiments/checkpoints/ is gitignored - a run trained on a
            # different machine (or whose checkpoint was since deleted to
            # save disk) has no local file to re-score against. Not an
            # error for this script's purpose: it can't independently
            # verify this run right now, but that's a missing artifact,
            # not a reproducibility failure.
            print(f"{run_dir.name}: no local checkpoint ({checkpoint_path.name}) - cannot re-score, skipped")
            continue
        fresh = evaluate.evaluate_checkpoint(
            checkpoint_path,
            split=stored["data"]["split"],
            batch_size=stored["timing"]["batch_size"],
            seed=stored["seed"],
        )
        if had_fingerprint and fresh["data"]["fingerprint"] != stored["data"]["fingerprint"]:
            fingerprint_mismatches.append(run_dir.name)
            print(f"{run_dir.name}: FINGERPRINT MISMATCH - the scoring data itself differs from "
                  f"when this was first scored (stored {stored['data']['fingerprint']!r}, "
                  f"fresh {fresh['data']['fingerprint']!r})")
            continue
        differences = compare(stored, fresh)
        worst = max(differences.values())
        within = worst <= evaluate.REPRODUCIBILITY_RTOL
        print(f"{run_dir.name}: {len(differences)} metrics, max relative difference {worst:.2e} "
              f"({'within' if within else 'OUTSIDE'} {evaluate.REPRODUCIBILITY_RTOL:g})"
              f"{' [fingerprint verified, already present]' if had_fingerprint else ' [backfilling fingerprint]'}")
        pending.append((path, stored, fresh, differences, within, had_fingerprint))

    if fingerprint_mismatches:
        print(f"{len(fingerprint_mismatches)} run(s) have a fingerprint mismatch - "
              f"investigate before trusting their stored metrics: {fingerprint_mismatches}")
        return 1
    if not pending:
        print("nothing to re-score")
        return 0
    if not all(within for *_, within, _ in pending):
        print("Not every metric reproduced within tolerance - nothing written.")
        return 1
    if not args.write:
        print("Dry run - pass --write to add/refresh the rescore blocks.")
        return 0

    provenance = {
        "date": date.today().isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "evaluate_py_sha256": _file_sha256(ROOT_DIR / "ml" / "training" / "evaluate.py"),
        "script": "scripts/fingerprint_run_records.py",
        "torch_num_threads": torch.get_num_threads(),
    }
    for path, stored, fresh, differences, _, had_fingerprint in pending:
        if not had_fingerprint:
            stored["data"]["fingerprint"] = fresh["data"]["fingerprint"]
        stored["rescore"] = {
            "purpose": ("independently re-score; reproduces the stored metrics and tunes nothing"
                        if had_fingerprint else
                        "add data.fingerprint; reproduces the stored metrics and tunes nothing"),
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
