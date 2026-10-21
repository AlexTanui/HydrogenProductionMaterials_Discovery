"""Comparison table across trained models, built from the run records alone.

Reads every `experiments/runs/*/run.json` and `test_metrics.json` (written
by ml/training/train.py, the test metrics by ml/training/evaluate.py) and
writes one table as CSV and Markdown:

    python -m ml.training.benchmark                       # -> reports/phase1_benchmark.{csv,md}
    python -m ml.training.benchmark --runs-dir DIR --out reports/NAME

With no `--runs-dir`, only `phase1_*` run directories are read (see
`main`'s `--name-prefix`): the shared `experiments/runs/` also holds
Phase 2/3 runs (phase2_blip) under a genuinely different protocol, and
this report is scoped to the Phase 1 bake-off. Pass `--runs-dir` and the
filter drops to "everything there" by default - naming a directory is
already choosing what's in it.

**Refuses to compare unlike runs - but only on what must be shared.**
A benchmark table comparing models scored under different *training
conditions* is worse than no table (CLAUDE.md); a table that also
refuses to show two models with different *model* settings is just
broken, and did exactly that the first time a Phase 2 run existed
alongside Phase 1 ones (Fazin's review of SCRUM-31/61). So every run
must match the first on all of:

- `data:` and `train:` - the config's locked blocks, as resolved;
- the training loop's fixed settings recorded in `run.json["protocol"]`
  (base loss, optimiser, scheduler, clipping, shuffle, epoch cap, the two
  loss weights), which live in code rather than config, so two runs at
  different commits could share configs and still differ here -
  **except** `MODEL_SPECIFIC_PROTOCOL_KEYS` (`kl_weight`, BLIP's NLL
  training settings, and `loss_override`, its NLL replacement for the
  base loss): those belong to one model, not to the shared protocol, so
  they are reported instead of gated on, and the report names them;
- the exact train and val frames (sha256 of their gold indices);
- the test scoring: split, molecule, theory, frame count, seed and batch
  size, and the **scoring-data fingerprint** (sha256 of the test indices
  and of z, R, E, F at them, recorded by evaluate.py), so a different gold
  file of the same size cannot pass.

Any remaining difference raises `ProtocolMismatch` naming the run and the
setting. Runs that are not complete, reportable test-split results are
refused too, and so are runs not trained from a clean commit (`git_dirty`
must be false): accepted results must come from one (SCRUM-48), or the
recorded commit does not fully describe the run.

Numbers are copied from the records, never recomputed: the CSV keeps them
at full precision (so tests can check them exactly), the Markdown rounds
them for reading. Each row carries the git commit it was trained at. ECE,
uncertainty-error Spearman, and the `MODEL_SPECIFIC_PROTOCOL_KEYS` columns
are present but empty for a model that doesn't define them (every Phase 1
model, for the BLIP-only ones) - same convention throughout. ECE/Spearman
come from evaluate.py's benchmark_row, whose definitions are Dongxiao's.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "COLUMNS", "ProtocolMismatch", "RunResult",
    "load_runs", "check_same_protocol", "build_rows", "render_csv", "render_markdown", "main",
]

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = ROOT_DIR / "experiments" / "runs"
DEFAULT_OUT = ROOT_DIR / "reports" / "phase1_benchmark"
PROTOCOL_DOC = "docs/phase1_protocol.md"

EXPECTED_UNITS = {"energy": "kcal/mol", "force": "kcal/mol/Ang"}

# run["protocol"] keys that exist for one model (today, only BLIP) and are
# allowed to differ between runs on purpose: they describe what that model
# itself does, not the shared training-loop conditions the comparison is
# supposed to hold fixed. Excluded from check_same_protocol's equality
# check (see protocol_signature) and reported instead: the numbers as
# columns, loss_override's prose under the report's Loss line. Found by
# Fazin's review of SCRUM-31/61: without this split, every optional
# per-model hook (kl_weight, then train_num_mc_samples/nll_warmup_epochs)
# silently became a new must-match field for every model, so retraining
# even one Phase 1 model after the hook existed produced a
# ProtocolMismatch against the other four that predated it - simulated
# and confirmed, not hypothetical.
#
# "loss" is deliberately not here (SCRUM-62): it is the shared base loss,
# and it lives in train.py rather than the config, so two runs at
# different commits can share a config and still train on different
# losses (an L1 PhysNet, a per-atom energy target). BLIP's NLL replacement
# is recorded separately, as loss_override, so the base stays comparable.
MODEL_SPECIFIC_PROTOCOL_KEYS = ("kl_weight", "train_num_mc_samples", "nll_warmup_epochs", "loss_override")

# (CSV column, Markdown header). Order is the table's column order.
COLUMNS = [
    ("run", "Run"),
    ("model", "Model"),
    ("phase", "Phase"),
    ("parameters", "Parameters"),
    ("energy_mae_kcal_per_mol", "Energy MAE"),
    ("energy_rmse_kcal_per_mol", "Energy RMSE"),
    ("force_mae_kcal_per_mol_per_ang", "Force MAE"),
    ("force_rmse_kcal_per_mol_per_ang", "Force RMSE"),
    ("epochs_run", "Epochs run"),
    ("best_epoch", "Best epoch"),
    ("stop_reason", "Stop reason"),
    ("training_time_s", "Training time"),
    ("inference_ms_per_frame", "Inference (ms/frame)"),
    ("ece", "ECE"),
    ("uncertainty_error_spearman", "Unc.-error Spearman"),
    ("kl_weight", "KL weight"),
    ("train_num_mc_samples", "Train MC samples"),
    ("nll_warmup_epochs", "NLL warmup epochs"),
    ("git_commit", "Git commit"),
]


class ProtocolMismatch(ValueError):
    """Runs that were not trained and scored under one protocol."""


@dataclass
class RunResult:
    name: str
    path: Path
    run: dict[str, Any]
    test: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_runs(runs_dir: Path = RUNS_DIR, name_prefix: str = "") -> list[RunResult]:
    """Every run record under `runs_dir` whose directory name starts with
    `name_prefix` (default "": every run record), sorted by name. Refuses
    non-results.

    The empty default means "take every run record I'm handed" - right
    for a caller that already points `runs_dir` at exactly the runs it
    wants (an isolated test fixture, or a hand-picked directory). It is
    wrong for `main()`'s CLI, which defaults `runs_dir` to the live,
    shared `experiments/runs/` - that directory now holds Phase 2/3 runs
    too (phase2_blip, with a genuinely different protocol: kl_weight > 0,
    its own loss term), and scanning it unfiltered picked those up right
    alongside the five Phase 1 runs, so `check_same_protocol` raised
    `ProtocolMismatch` the moment a Phase 2 run first existed - not a bug
    in that check, a missing filter at the one call site that needed one.
    `main()` passes `name_prefix="phase1_"` by default for exactly that
    reason; this function's own default is unfiltered on purpose.
    """
    runs_dir = Path(runs_dir)
    run_dirs = (
        sorted(p for p in runs_dir.iterdir() if p.name.startswith(name_prefix) and (p / "run.json").is_file())
        if runs_dir.is_dir() else []
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"no run records ({name_prefix or '*'}*/run.json) under {runs_dir}"
        )
    results = []
    for path in run_dirs:
        run = _read_json(path / "run.json")
        if run.get("status") != "complete" or run.get("reportable") is not True:
            raise ValueError(
                f"{path.name}: status={run.get('status')!r}, reportable={run.get('reportable')!r}. "
                "Only complete, reportable runs go in a benchmark table."
            )
        if run.get("git_dirty") is not False:
            # True, or None (git unavailable): either way, no clean commit on record.
            raise ValueError(
                f"{path.name}: git_dirty={run.get('git_dirty')!r} at commit {str(run.get('git_commit'))[:7]}. "
                "Accepted results must come from a clean commit (SCRUM-48) - retrain it from one."
            )
        test_path = path / "test_metrics.json"
        if not test_path.is_file():
            raise FileNotFoundError(f"{path.name}: no test_metrics.json - the run was never scored")
        test = _read_json(test_path)
        if test["data"]["split"] != "test":
            raise ValueError(f"{path.name}: scored on {test['data']['split']!r}, not the test split")
        if len(test["metrics"]["per_key"]) != 1:
            raise ValueError(f"{path.name}: expected one (molecule, theory) key in the test metrics")
        units = test["metrics"]["per_key"][0]["units"]
        for quantity, unit in EXPECTED_UNITS.items():
            if units[quantity] != unit:
                raise ValueError(f"{path.name}: {quantity} in {units[quantity]!r}, table expects {unit!r}")
        if run["n_parameters"] != test["n_parameters"]:
            raise ValueError(f"{path.name}: run.json and test_metrics.json disagree on the parameter count")
        if "fingerprint" not in test["data"]:
            raise ValueError(
                f"{path.name}: test_metrics.json has no scoring-data fingerprint, so the data it was "
                "scored on cannot be checked. Re-score it: python scripts/fingerprint_run_records.py"
            )
        results.append(RunResult(path.name, path, run, test))
    return results


def protocol_signature(result: RunResult) -> dict[str, Any]:
    """Everything that must match for two runs to share a table.

    "training loop" is `run["protocol"]` minus `MODEL_SPECIFIC_PROTOCOL_KEYS`
    - those vary per model on purpose (BLIP's kl_weight, its NLL warmup) and
    are reported as their own columns in `build_rows`, not gated on here.
    """
    run, test = result.run, result.test
    return {
        "data": run["config"]["data"],
        "train": run["config"]["train"],
        "training loop": {k: v for k, v in run["protocol"].items() if k not in MODEL_SPECIFIC_PROTOCOL_KEYS},
        "train frames": {"sha256": run["subsets"]["train"]["gold_indices_sha256"]},
        "val frames": {"sha256": run["subsets"]["val"]["gold_indices_sha256"]},
        "test scoring": {
            **{k: test["data"][k] for k in ("split", "molecule", "theory", "n_configs")},
            "seed": test.get("seed"),
            "batch_size": test["timing"]["batch_size"],
            "data_fingerprint": test["data"]["fingerprint"]["sha256"],
        },
    }


def check_same_protocol(results: list[RunResult]) -> dict[str, Any]:
    """Raise ProtocolMismatch unless every run matches the first. Returns the shared signature."""
    reference = protocol_signature(results[0])
    problems = []
    for result in results[1:]:
        signature = protocol_signature(result)
        for section, expected in reference.items():
            actual = signature[section]
            for key in sorted(set(expected) | set(actual)):
                if expected.get(key) != actual.get(key):
                    problems.append(
                        f"{result.name}: {section}.{key} = {actual.get(key)!r}, but "
                        f"{results[0].name} has {expected.get(key)!r}"
                    )
    if problems:
        raise ProtocolMismatch(
            "these runs were not trained and scored under one protocol, so they cannot share "
            "a table:\n  " + "\n  ".join(problems)
        )
    return reference


def build_rows(results: list[RunResult]) -> list[dict[str, Any]]:
    """One row per run, every number copied from its records unchanged."""
    rows = []
    for result in results:
        run, test = result.run, result.test
        metrics = test["metrics"]["per_key"][0]
        row = test["benchmark_row"]
        rows.append({
            "run": result.name,
            "model": run["model"],
            "phase": run["phase"],
            "parameters": run["n_parameters"],
            "energy_mae_kcal_per_mol": metrics["energy_mae_total"],
            "energy_rmse_kcal_per_mol": metrics["energy_rmse_total"],
            "force_mae_kcal_per_mol_per_ang": metrics["force_mae_component"],
            "force_rmse_kcal_per_mol_per_ang": metrics["force_rmse_component"],
            "epochs_run": run["epochs_run"],
            "best_epoch": run["best_epoch"],
            "stop_reason": run["stop_reason"],
            "training_time_s": run["wall_clock_s"],
            "inference_ms_per_frame": test["timing"]["inference_s_per_config"] * 1000,
            "ece": row["ece"],
            "uncertainty_error_spearman": row["uncertainty_correlation"],
            # Model-specific (MODEL_SPECIFIC_PROTOCOL_KEYS) - not gated on,
            # shown so a reader can see why e.g. two BLIP runs differ.
            # .get(): absent entirely for a run recorded before the key
            # existed, same convention as ece/uncertainty_error_spearman.
            "kl_weight": run["protocol"].get("kl_weight"),
            "train_num_mc_samples": run["protocol"].get("train_num_mc_samples"),
            "nll_warmup_epochs": run["protocol"].get("nll_warmup_epochs"),
            "git_commit": run["git_commit"],
        })
    return rows


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    return repr(value) if isinstance(value, float) else str(value)  # repr: exact round trip


def render_csv(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([column for column, _ in COLUMNS])
    for row in rows:
        writer.writerow([_csv_value(row[column]) for column, _ in COLUMNS])
    return buffer.getvalue()


def _md_value(column: str, value: Any) -> str:
    if value is None:
        return ""
    if column == "parameters":
        return f"{value:,}"
    if column == "training_time_s":
        return f"{value / 3600:.2f} h"
    if column == "inference_ms_per_frame":
        return f"{value:.3f}"
    if column == "git_commit":
        return f"`{value[:7]}`"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _budget_text(train: dict[str, Any]) -> str:
    """Train/val frame budget. A subset size of None means the whole split: "all", no stride."""
    train_size, val_size, seed = train["train_subset"], train["val_subset"], train["train_subset_seed"]
    if train_size is not None and val_size is not None:
        return f"{train_size:,} train and {val_size:,} val frames (stride subsets, seed {seed})"
    if train_size is None and val_size is None:
        return "all train and val frames"

    def part(size: Optional[int], split: str) -> str:
        return f"all {split} frames" if size is None else f"{size:,} {split} frames (stride subset, seed {seed})"

    return f"{part(train_size, 'train')} and {part(val_size, 'val')}"


def render_markdown(results: list[RunResult], rows: list[dict[str, Any]], shared: dict[str, Any],
                    title: str = "Phase 1 benchmark") -> str:
    data, train = shared["data"], shared["train"]
    loop, scoring = shared["training loop"], shared["test scoring"]
    unchecked = ", ".join(f"`{key}`" for key in MODEL_SPECIFIC_PROTOCOL_KEYS)
    overrides = [(result.name, result.run["protocol"].get("loss_override")) for result in results]
    lines = [
        f"# {title}: {data['molecule']} ({data['theory']})",
        "",
        "Generated by `python -m ml.training.benchmark` from `experiments/runs/*/run.json` and "
        "`test_metrics.json`. Do not edit by hand; regenerate. The protocol and its limitations are "
        f"in [{PROTOCOL_DOC}](../{PROTOCOL_DOC}).",
        "",
        f"All {len(rows)} runs were checked to share one protocol before being tabulated: identical "
        "`data:` and `train:` blocks, training-loop settings, train and val frames, and test scoring. "
        f"Not checked, because each describes one model rather than the shared protocol: {unchecked}.",
        "",
        "| " + " | ".join(header for _, header in COLUMNS) + " |",
        "|" + "|".join("---" for _ in COLUMNS) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md_value(column, row[column]) for column, _ in COLUMNS) + " |")
    lines += [
        "",
        "- **Energy**: kcal/mol, error in total molecular energy (`energy_mae_total`, "
        "`energy_rmse_total`). **Forces**: kcal/mol/Å, per Cartesian component "
        "(`force_mae_component`, `force_rmse_component`).",
        f"- **Test split**: {scoring['n_configs']:,} frames of {scoring['molecule']} "
        f"({scoring['theory']}), scored once by `ml/training/evaluate.py` with seed "
        f"{scoring['seed']} and batch size {scoring['batch_size']}. Scoring-data fingerprint "
        f"(sha256 of the test indices and of z, R, E, F at them): `{scoring['data_fingerprint']}`.",
        "- **Training time**: wall-clock of the training loop, excluding test scoring. "
        "**Inference**: forward pass plus force backward pass per frame, batch size "
        f"{scoring['batch_size']}, excluding graph construction.",
        "- **ECE / uncertainty-error Spearman**: empty for deterministic models. The definitions "
        "are Dongxiao's; Phase 2/3 rows fill these columns.",
        "",
        "## Shared protocol",
        "",
        f"- Data: `{data['gold_path']}`, cutoff {data['cutoff_radius']} Å, seed {data['seed']}.",
        f"- Budget: {_budget_text(train)}; batch {train['batch_size']}; up to "
        f"{train['epochs']} epochs, early-stopping patience {train['patience']}; "
        f"device {train['device']}.",
        f"- Loss: {loop['loss']}; weights {loop['energy_loss_weight']} / "
        f"{loop['force_loss_weight']}.",
        *(f"  - {name} replaces it (model-specific `loss_override`, not checked): {text}."
          for name, text in overrides if text),
        f"- Optimiser: {loop['optimizer']['name']}, lr {loop['optimizer']['lr']}, weight decay "
        f"{loop['optimizer']['weight_decay']}; {loop['scheduler']['name']} (factor "
        f"{loop['scheduler']['factor']}, patience {loop['scheduler']['patience']}); gradient "
        f"clipping {loop['grad_clip_norm']}; shuffle: {loop['shuffle']}.",
        "",
        "## Run records",
        "",
        "| Run | Record | Git commit | Dirty tree | Device | Threads | Checkpoint sha256 "
        "| Re-score max relative difference |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        run = result.run
        env = run["environment"]
        rescore = result.test.get("rescore")
        reproduced = f"{rescore['max_relative_difference']:.1e}" if rescore else "not re-scored"
        lines.append(
            f"| {result.name} | [experiments/runs/{result.name}/](../experiments/runs/{result.name}/) "
            f"| `{run['git_commit'][:7]}` | {run['git_dirty']} | {env['device']} "
            f"| {env['torch_num_threads']} | `{run.get('checkpoint_sha256', '')[:12]}` | {reproduced} |"
        )
    if any("rescore" in result.test for result in results):
        lines += [
            "",
            "Runs scored before evaluate.py recorded fingerprints were re-scored by "
            "`scripts/fingerprint_run_records.py` to add one. Their stored metrics are unchanged; "
            "the last column is the largest relative difference the re-score measured against them "
            "(tolerance 1e-5 relative, because multithreaded CPU scoring is not bit-reproducible).",
        ]
    return "\n".join(lines) + "\n"


def build(runs_dir: Path = RUNS_DIR, title: str = "Phase 1 benchmark",
          name_prefix: str = "") -> tuple[str, str, list[dict]]:
    """(csv text, markdown text, rows) for every `name_prefix*` run under `runs_dir`."""
    results = load_runs(runs_dir, name_prefix=name_prefix)
    shared = check_same_protocol(results)
    rows = build_rows(results)
    return render_csv(rows), render_markdown(results, rows, shared, title=title), rows


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the comparison table from the run records.")
    parser.add_argument("--runs-dir", default=None,
                        help=f"default {RUNS_DIR} - the live, shared run directory, which also means "
                             "--name-prefix defaults to 'phase1_' there (see --name-prefix)")
    parser.add_argument("--name-prefix", default=None,
                        help="only include run directories whose name starts with this. Defaults to "
                             "'phase1_' when --runs-dir is left at its default (the shared "
                             "experiments/runs/, which holds Phase 2/3 runs under other protocols too), "
                             "and to '' (every run record) when --runs-dir is given explicitly - a "
                             "caller who names a directory has already chosen what's in it")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="output path without extension; writes .csv and .md")
    parser.add_argument("--title", default="Phase 1 benchmark")
    args = parser.parse_args(argv)

    runs_dir_given = args.runs_dir is not None
    runs_dir = Path(args.runs_dir) if runs_dir_given else RUNS_DIR
    name_prefix = args.name_prefix if args.name_prefix is not None else ("" if runs_dir_given else "phase1_")

    csv_text, md_text, rows = build(runs_dir, title=args.title, name_prefix=name_prefix)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # newline="" so the files are byte-identical to the strings (tests compare them).
    with open(out.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
        f.write(csv_text)
    with open(out.with_suffix(".md"), "w", encoding="utf-8", newline="") as f:
        f.write(md_text)
    print(md_text)
    print(f"Wrote {out.with_suffix('.csv')} and {out.with_suffix('.md')} ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
