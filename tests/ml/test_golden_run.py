"""Golden-run regression guard (SCRUM-38): a fixed-seed Phase 1 run whose
metrics must not move unless a change is meant to move them.

The loader, graph construction, models and metric code are shared by every
model and phase, so a change to any of them shifts every benchmark number
with nothing to announce it. This test trains phase1_baseline's MPNN for a
few epochs on a small fixed set of real ethanol (DFT) frames, through the
real path, and compares the scored metrics with the committed reference
values in tests/ml/golden/phase1_mpnn_golden_run.json:

- **Path.** `train()` (MD17Dataset -> build_graph -> registry model ->
  ScaledEnergyModel -> loss, optimiser, checkpoint) and then
  `evaluate_checkpoint()` (MetricAccumulator). Nothing is reimplemented.
- **Frames.** `GOLDEN_TRAIN_FRAMES` from the real train block and
  `GOLDEN_VAL_FRAMES` from the real val block, evenly spaced, copied into
  a temporary gold file whose test split is empty - no test frame is
  read. Training selects on those val frames, and they are what is scored.
- **Config.** experiments/configs/phase1_baseline.yaml as written, with
  only the budget (`GOLDEN_EPOCHS` and the two frame counts) changed and
  `gold_path` pointed at the temporary file. A change to that config's
  model or optimiser settings therefore moves the golden numbers too.
- **One epoch** (16 optimiser steps). Early training swings hard, and it
  amplifies the tiny summation-order differences between thread counts:
  scored after one epoch they stay near 1e-6 relative, but after two
  they already reach 1e-4 to 1e-3 and keep growing, which would leave a
  tolerance either too loose to catch a real change or flaky.
- **Threads.** torch is pinned to `GOLDEN_THREADS` for the run and
  restored afterwards, and the count is stored with the reference values.
  The same config ended 22% apart on 200-epoch force MAE across two
  machines with different thread counts, so the count is part of the
  reference rather than left to the machine.
- **Output.** pytest's temporary directory only; nothing in experiments/
  or reports/.

A failure names each metric that moved, from what to what, and by how
much. If the gold frames themselves changed, it says that instead, since
the metrics are then not comparable.

**Regenerating the reference** - only for a change that is meant to move
the numbers, never to make a failure go away without knowing its cause:

    python tests/ml/test_golden_run.py --regenerate --reason "why the numbers moved"

from a clean tree, then commit tests/ml/golden/phase1_mpnn_golden_run.json
on its own, with the reason in the commit message. The reason, commit and
platform are stored in the file too.

**Measuring run-to-run variation** (each run in a fresh process):

    python tests/ml/test_golden_run.py --measure --threads 1 4 24 --repeats 3
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest
import torch

from ml.config import ExperimentConfig
from ml.training.evaluate import evaluate_checkpoint
from ml.training.train import environment, git_state, train

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "gold" / "md17" / "ethanol_dft.npz"
BASE_CONFIG = ROOT / "experiments" / "configs" / "phase1_baseline.yaml"
REFERENCE = Path(__file__).resolve().parent / "golden" / "phase1_mpnn_golden_run.json"

# The run. Changing any of these changes what the reference describes:
# regenerate it in the next commit (--regenerate needs a clean tree), and
# test_reference_describes_this_golden_run fails until then.
GOLDEN_NAME = "golden_phase1_mpnn"
GOLDEN_TRAIN_FRAMES = 512
GOLDEN_VAL_FRAMES = 256
GOLDEN_EPOCHS = 1
GOLDEN_THREADS = 1

# Largest accepted relative difference, |now - reference| / |reference|, per
# metric. Not agreed yet, so None: the test then skips, printing the measured
# differences (`pytest -rs`), rather than pass or fail against a number
# nobody has approved.
GOLDEN_RTOL: Optional[float] = None


# --------------------------------------------------------------------------
# The golden run
# --------------------------------------------------------------------------

def golden_frames(gold) -> dict[str, np.ndarray]:
    """Gold indices of the golden train and val frames: evenly spaced over each real block.

    `endpoint=False` stops short of each block's last frame, so the last
    train frame and the first val frame - one MD step apart, a near-duplicate
    pair (docs/reviews/leakage-audit-2026-10-31.md) - are never both chosen.
    """
    frames = {}
    for split, size in (("train", GOLDEN_TRAIN_FRAMES), ("val", GOLDEN_VAL_FRAMES)):
        idx = gold[f"{split}_idx"]
        frames[split] = idx[np.linspace(0, len(idx), size, endpoint=False).astype(int)]
    return frames


def _sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def write_golden_gold(path: Path) -> dict[str, Any]:
    """Copy the golden frames into a gold file at `path` (test split empty); describe them."""
    with np.load(GOLD, allow_pickle=False) as gold:
        frames = golden_frames(gold)
        chosen = np.concatenate([frames["train"], frames["val"]])
        assert not np.isin(chosen, gold["test_idx"]).any(), "a golden frame is a test frame"
        R, E, F = gold["R"][chosen], gold["E"][chosen], gold["F"][chosen]
        n_train = len(frames["train"])
        np.savez(
            path, z=gold["z"], R=R, E=E, F=F,
            train_idx=np.arange(n_train), val_idx=np.arange(n_train, len(chosen)),
            test_idx=np.arange(0), molecule=gold["molecule"], theory=gold["theory"],
            used_literature_split=False,
        )
        z = gold["z"]
    return {
        "source": GOLD.relative_to(ROOT).as_posix(),
        "selection": "numpy.linspace(0, len(split_idx), n, endpoint=False).astype(int) over the real train_idx and val_idx",
        **{f"{split}_frames": int(len(idx)) for split, idx in frames.items()},
        **{f"{split}_first_last_gold_index": [int(idx[0]), int(idx[-1])] for split, idx in frames.items()},
        "gold_indices_sha256": _sha256(chosen.astype(np.int64)),
        "data_sha256": _sha256(z.astype(np.int64), R, E, F),
    }


def golden_config(gold_path: Path) -> ExperimentConfig:
    """phase1_baseline.yaml as written, on the golden frames with the golden budget."""
    cfg = ExperimentConfig.from_yaml(BASE_CONFIG)
    return dataclasses.replace(
        cfg,
        name=GOLDEN_NAME,
        data=dataclasses.replace(cfg.data, gold_path=str(gold_path)),
        train=dataclasses.replace(
            cfg.train, epochs=GOLDEN_EPOCHS,
            train_subset=GOLDEN_TRAIN_FRAMES, val_subset=GOLDEN_VAL_FRAMES,
        ),
    )


def golden_run(workdir: Path, threads: int = GOLDEN_THREADS) -> dict[str, Any]:
    """Train and score the golden run under `workdir`, with torch pinned to `threads`.

    The caller's thread count and RNG state are restored afterwards.
    """
    workdir = Path(workdir)
    gold_path = workdir / "ethanol_dft_golden.npz"
    frames = write_golden_gold(gold_path)
    cfg = golden_config(gold_path)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        with torch.random.fork_rng():
            start = time.perf_counter()
            record = train(cfg, output_root=workdir, score_test=False)
            result = evaluate_checkpoint(
                cfg.checkpoint_path(workdir / "checkpoints"), split="val", batch_size=cfg.train.batch_size,
            )
            seconds = time.perf_counter() - start
        env = environment(torch.device(cfg.train.device))
    finally:
        torch.set_num_threads(previous_threads)
    scored = result["metrics"]["per_key"][0]
    return {
        "metrics": {field: scored[field] for field in sorted(scored["fields"])},
        "units": scored["units"],
        "scored": {"split": "val", "n_frames": scored["n_frames"]},
        "frames": frames,
        "best_epoch": record["best_epoch"],
        "epochs_run": record["epochs_run"],
        "torch_num_threads": threads,
        "environment": env,
        "seconds": seconds,
    }


def budget() -> dict[str, Any]:
    return {
        "base_config": BASE_CONFIG.relative_to(ROOT).as_posix(),
        "name": GOLDEN_NAME,
        "epochs": GOLDEN_EPOCHS,
        "train_frames": GOLDEN_TRAIN_FRAMES,
        "val_frames": GOLDEN_VAL_FRAMES,
        "torch_num_threads": GOLDEN_THREADS,
    }


def relative_differences(reference: dict[str, float], current: dict[str, float]) -> dict[str, float]:
    return {field: abs(current[field] - reference[field]) / abs(reference[field])
            for field in reference if field in current}


def drift_report(reference: dict[str, Any], current: dict[str, Any], rtol: float) -> list[str]:
    """One line per metric outside `rtol` (or missing on either side); empty if nothing moved."""
    ref, now = reference["metrics"], current["metrics"]
    lines = []
    for field in sorted(set(ref) | set(now)):
        if field not in now:
            lines.append(f"{field}: in the reference ({ref[field]:.6g}) but no longer computed")
        elif field not in ref:
            lines.append(f"{field}: computed now ({now[field]:.6g}) but not in the reference")
        else:
            change = now[field] - ref[field]
            relative = abs(change) / abs(ref[field])
            if relative > rtol:
                lines.append(f"{field}: {ref[field]:.6g} -> {now[field]:.6g} "
                             f"({change:+.3g}, {relative:.2e} relative; tolerance {rtol:g})")
    return lines


def _describe_machine(run: dict[str, Any]) -> str:
    env = run["environment"]
    return f"{env['platform']}, {env['processor']}, torch {env['torch']}, {run['torch_num_threads']} thread(s)"


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def _reference() -> dict[str, Any]:
    if not REFERENCE.exists():
        pytest.fail(f"{REFERENCE} is missing: generate it (see this module's docstring)")
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def test_reference_describes_this_golden_run():
    """The committed reference was made with this module's frames, budget and thread count."""
    reference = _reference()
    assert reference["budget"] == budget(), (
        "the golden run's settings changed but the reference wasn't regenerated: "
        f"reference {reference['budget']}, now {budget()}"
    )


def test_drift_report_names_the_metric_and_how_far_it_moved():
    reference = {"metrics": {"energy_mae_total": 0.5, "force_mae_component": 2.0, "gone": 1.0}}
    current = {"metrics": {"energy_mae_total": 0.5000001, "force_mae_component": 2.2, "new": 3.0}}
    lines = drift_report(reference, current, rtol=1e-4)
    assert lines == [
        "force_mae_component: 2 -> 2.2 (+0.2, 1.00e-01 relative; tolerance 0.0001)",
        "gone: in the reference (1) but no longer computed",
        "new: computed now (3) but not in the reference",
    ]


def test_golden_run_matches_the_reference(tmp_path):
    if not GOLD.exists():
        pytest.skip(f"{GOLD} not present - run `python -m ml.data.preprocessing`")
    reference = _reference()
    current = golden_run(tmp_path, threads=reference["torch_num_threads"])
    machines = f"reference: {_describe_machine(reference)}; now: {_describe_machine(current)}"

    assert current["frames"] == reference["frames"], (
        "the golden frames changed (gold file regenerated, or the frame selection changed), so the "
        f"metrics are not comparable. Reference {reference['frames']}, now {current['frames']}"
    )
    assert current["scored"] == reference["scored"]
    assert (current["epochs_run"], current["best_epoch"]) == (reference["epochs_run"], reference["best_epoch"]), (
        f"checkpoint selection changed: the reference ran {reference['epochs_run']} epochs and kept epoch "
        f"{reference['best_epoch']}; this run ran {current['epochs_run']} and kept {current['best_epoch']}. "
        f"({machines})"
    )

    differences = relative_differences(reference["metrics"], current["metrics"])
    measured = ", ".join(f"{field} {value:.1e}" for field, value in sorted(differences.items()))
    if GOLDEN_RTOL is None:
        pytest.skip(f"golden-run tolerance not agreed yet; relative differences from the reference: "
                    f"{measured} ({machines})")
    moved = drift_report(reference, current, GOLDEN_RTOL)
    assert not moved, (
        f"golden run moved (SCRUM-38): {len(moved)} metric(s) outside tolerance.\n  "
        + "\n  ".join(moved)
        + f"\n{machines}.\nIf a change is meant to move these numbers, regenerate the reference and "
        "say why in the commit (this module's docstring); otherwise find the change that moved them."
    )


# --------------------------------------------------------------------------
# Regenerate and measure (not collected by pytest)
# --------------------------------------------------------------------------

def _regenerate(reason: str) -> int:
    git = git_state(ROOT)
    if git["git_dirty"]:
        print("refusing to regenerate from a dirty tree: the recorded commit would not describe "
              "the code that made the numbers. Commit first.", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="golden_run_") as workdir:
        run = golden_run(Path(workdir))
    reference = {
        "what": "SCRUM-38 golden run; see tests/ml/test_golden_run.py",
        "reason": reason,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git["git_commit"],
        "budget": budget(),
        **run,
    }
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {REFERENCE.relative_to(ROOT).as_posix()} ({run['seconds']:.1f} s): {run['metrics']}")
    return 0


def _measure(threads: list[int], repeats: int, out: Optional[Path]) -> int:
    """Run the golden run `repeats` times at each thread count, each in a fresh process."""
    runs = []
    with tempfile.TemporaryDirectory(prefix="golden_measure_") as tmp:
        for count in threads:
            for repeat in range(repeats):
                path = Path(tmp) / f"run_{count}_{repeat}.json"
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--run-once",
                                "--threads", str(count), "--out", str(path)],
                               cwd=ROOT, check=True, capture_output=True)
                run = json.loads(path.read_text(encoding="utf-8"))
                runs.append(run)
                print(f"threads {count:>2} repeat {repeat}: {run['seconds']:6.1f} s  "
                      + "  ".join(f"{k} {v:.10g}" for k, v in run["metrics"].items()), flush=True)
    first = {count: next(r for r in runs if r["torch_num_threads"] == count) for count in threads}
    summary = {"same_thread_count": {}, "vs_first_thread_count": {}}
    for count in threads:
        same = [r for r in runs if r["torch_num_threads"] == count]
        summary["same_thread_count"][count] = max(
            max(relative_differences(same[0]["metrics"], r["metrics"]).values()) for r in same)
        summary["vs_first_thread_count"][count] = max(
            relative_differences(first[threads[0]]["metrics"], first[count]["metrics"]).values())
    print("largest relative difference between repeats at one thread count:",
          {k: f"{v:.1e}" for k, v in summary["same_thread_count"].items()})
    print(f"largest relative difference from {threads[0]} thread(s):",
          {k: f"{v:.1e}" for k, v in summary["vs_first_thread_count"].items()})
    if out:
        out.write_text(json.dumps({"runs": runs, "summary": summary}, indent=2) + "\n", encoding="utf-8")
    return 0


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate or measure the SCRUM-38 golden run.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--regenerate", action="store_true", help="rewrite the committed reference values")
    action.add_argument("--measure", action="store_true", help="repeat the run at several thread counts")
    action.add_argument("--run-once", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reason", help="why the numbers moved (required with --regenerate)")
    parser.add_argument("--threads", type=int, nargs="+", default=[GOLDEN_THREADS])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.regenerate:
        if not args.reason:
            parser.error("--regenerate needs --reason: the regeneration commit must say why")
        return _regenerate(args.reason)
    if args.measure:
        return _measure(args.threads, args.repeats, args.out)
    with tempfile.TemporaryDirectory(prefix="golden_once_") as workdir:
        run = golden_run(Path(workdir), threads=args.threads[0])
    args.out.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
