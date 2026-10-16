"""Tests for ml/training/benchmark.py: the comparison table built from the run records.

Two properties matter most, because each one failing would put a wrong
number in the report without any error:

1. **Unlike runs are refused.** Runs that differ in their `data:` or
   `train:` blocks, their training-loop settings, their train/val frames or
   their test scoring cannot share a table.
2. **The table reproduces the records exactly.** Every number in the CSV
   is the run record's own value - not recomputed, not rounded - and the
   committed reports/phase1_benchmark.{csv,md} match a fresh build.

The protocol-mismatch tests use small synthetic run records; the
reproduction tests read the real, committed experiments/runs/.
"""
from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path

import pytest

from ml.training import benchmark
from ml.training.benchmark import ProtocolMismatch, build, load_runs, main

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments" / "runs"
REPORT = ROOT / "reports" / "phase1_benchmark"


# --------------------------------------------------------------------------
# Synthetic run records
# --------------------------------------------------------------------------

_RUN = {
    "status": "complete",
    "reportable": True,
    "model": "Toy",
    "phase": "toy",
    "n_parameters": 10,
    "config": {
        "data": {"molecule": "ethanol", "theory": "dft", "gold_path": "data/gold/md17/ethanol_dft.npz",
                 "cutoff_radius": 5.0, "num_rbf": 16, "seed": 42},
        "train": {"epochs": 200, "batch_size": 32, "lr": 0.0005, "energy_loss_weight": 1.0,
                  "force_loss_weight": 100.0, "patience": 20, "device": "cpu",
                  "train_subset": 10000, "train_subset_seed": 42, "val_subset": 5000},
    },
    "protocol": {"loss": "MSE", "energy_loss_weight": 1.0, "force_loss_weight": 100.0,
                 "optimizer": {"name": "Adam", "lr": 0.0005, "weight_decay": 1e-06},
                 "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.5, "patience": 5},
                 "grad_clip_norm": 10.0, "shuffle": "seeded generator", "max_epochs": 200},
    "subsets": {"train": {"gold_indices_sha256": "a" * 64}, "val": {"gold_indices_sha256": "b" * 64}},
    "epochs_run": 200,
    "best_epoch": 199,
    "stop_reason": "max_epochs",
    "wall_clock_s": 1234.5678,
    "git_commit": "0123456789abcdef0123456789abcdef01234567",
    "git_dirty": False,
    "environment": {"device": "cpu", "torch_num_threads": 4},
    "checkpoint_sha256": "c" * 64,
}
_TEST = {
    "data": {"split": "test", "molecule": "ethanol", "theory": "dft", "n_configs": 55509,
             "fingerprint": {"algorithm": "sha256", "sha256": "e" * 64}},
    "seed": 0,
    "n_parameters": 10,
    "timing": {"batch_size": 32, "inference_s_per_config": 0.000512},
    "metrics": {"per_key": [{
        "units": {"energy": "kcal/mol", "length": "Ang", "force": "kcal/mol/Ang"},
        "energy_mae_total": 0.0531, "energy_rmse_total": 0.0678,
        "force_mae_component": 0.1196, "force_rmse_component": 0.2040,
    }]},
    "benchmark_row": {"ece": None, "uncertainty_correlation": None},
}


def _set(record: dict, dotted: str, value) -> None:
    *parents, leaf = dotted.split(".")
    for key in parents:
        record = record[key]
    record[leaf] = value


def _write_run(runs_dir: Path, name: str, run_changes=None, test_changes=None) -> Path:
    run, test = copy.deepcopy(_RUN), copy.deepcopy(_TEST)
    for dotted, value in (run_changes or {}).items():
        _set(run, dotted, value)
    for dotted, value in (test_changes or {}).items():
        _set(test, dotted, value)
    path = runs_dir / name
    path.mkdir(parents=True)
    (path / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (path / "test_metrics.json").write_text(json.dumps(test), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# 1. Unlike runs are refused
# --------------------------------------------------------------------------

@pytest.mark.parametrize("record, dotted, value, reported", [
    ("run", "config.train.lr", 0.001, "train.lr"),
    ("run", "config.train.batch_size", 256, "train.batch_size"),
    ("run", "config.train.train_subset", 20000, "train.train_subset"),
    ("run", "config.data.theory", "ccsd_t", "data.theory"),
    ("run", "config.data.gold_path", "data/gold/md17/ethanol_ccsd_t.npz", "data.gold_path"),
    ("run", "protocol.grad_clip_norm", 1.0, "training loop.grad_clip_norm"),
    ("run", "subsets.train.gold_indices_sha256", "d" * 64, "train frames.sha256"),
    ("test", "seed", 1, "test scoring.seed"),
    ("test", "timing.batch_size", 1, "test scoring.batch_size"),
    # Same molecule, theory and frame count - different data.
    ("test", "data.fingerprint.sha256", "f" * 64, "test scoring.data_fingerprint"),
])
def test_runs_under_different_protocols_are_refused(tmp_path, record, dotted, value, reported):
    _write_run(tmp_path, "a_reference")
    changes = {dotted: value}
    _write_run(tmp_path, "b_different", **({"run_changes": changes} if record == "run" else {"test_changes": changes}))
    with pytest.raises(ProtocolMismatch, match=rf"b_different: {reported} ="):
        build(tmp_path)


def test_a_key_present_in_only_one_run_is_a_mismatch(tmp_path):
    _write_run(tmp_path, "a_reference")
    _write_run(tmp_path, "b_extra", run_changes={"config.train.extra_setting": 1})
    with pytest.raises(ProtocolMismatch, match="train.extra_setting"):
        build(tmp_path)


def test_runs_under_one_protocol_are_tabulated(tmp_path):
    # Different models, sizes and commits - only model: and provenance differ.
    _write_run(tmp_path, "a", run_changes={"model": "A", "phase": "a", "n_parameters": 5},
               test_changes={"n_parameters": 5})
    _write_run(tmp_path, "b", run_changes={"model": "B", "git_commit": "f" * 40})
    _, _, rows = build(tmp_path)
    assert [row["run"] for row in rows] == ["a", "b"]


@pytest.mark.parametrize("run_changes, test_changes, match", [
    ({"reportable": False}, {}, "reportable"),
    ({"status": "interrupted"}, {}, "status"),
    ({}, {"data.split": "val"}, "not the test split"),
    ({}, {"metrics.per_key.0.units.energy": "eV"}, "kcal/mol"),
    ({"n_parameters": 11}, {}, "parameter count"),
])
def test_records_that_are_not_comparable_results_are_refused(tmp_path, run_changes, test_changes, match):
    if test_changes.get("metrics.per_key.0.units.energy"):
        # _set cannot index a list by name; edit the unit directly.
        path = _write_run(tmp_path, "a")
        test = json.loads((path / "test_metrics.json").read_text(encoding="utf-8"))
        test["metrics"]["per_key"][0]["units"]["energy"] = "eV"
        (path / "test_metrics.json").write_text(json.dumps(test), encoding="utf-8")
    else:
        _write_run(tmp_path, "a", run_changes=run_changes, test_changes=test_changes)
    with pytest.raises(ValueError, match=match):
        load_runs(tmp_path)


def test_run_without_a_scoring_data_fingerprint_is_refused(tmp_path):
    path = _write_run(tmp_path, "a")
    test = json.loads((path / "test_metrics.json").read_text(encoding="utf-8"))
    del test["data"]["fingerprint"]
    (path / "test_metrics.json").write_text(json.dumps(test), encoding="utf-8")
    with pytest.raises(ValueError, match="no scoring-data fingerprint"):
        load_runs(tmp_path)


def test_markdown_reports_the_fingerprint_and_any_rescore(tmp_path):
    _write_run(tmp_path, "a")
    _write_run(tmp_path, "b", test_changes={"rescore": {"max_relative_difference": 2.2e-08}})
    _, md_text, _ = build(tmp_path)
    assert "`" + "e" * 64 + "`" in md_text
    assert md_text.count("not re-scored") == 1 and "| 2.2e-08 |" in md_text


@pytest.mark.parametrize("train_subset, val_subset, seed, expected", [
    # Whole splits: subset sizes (and the seed) are None in the config.
    (None, None, None, "Budget: all train and val frames; batch 32"),
    (10000, None, 42, "Budget: 10,000 train frames (stride subset, seed 42) and all val frames; batch 32"),
    (None, 5000, 42, "Budget: all train frames and 5,000 val frames (stride subset, seed 42); batch 32"),
    (10000, 5000, 42, "Budget: 10,000 train and 5,000 val frames (stride subsets, seed 42); batch 32"),
])
def test_whole_split_runs_are_reported_as_all(tmp_path, train_subset, val_subset, seed, expected):
    changes = {"config.train.train_subset": train_subset, "config.train.val_subset": val_subset,
               "config.train.train_subset_seed": seed}
    _write_run(tmp_path, "a", run_changes=changes)
    _write_run(tmp_path, "b", run_changes=changes)
    csv_text, md_text, _ = build(tmp_path)
    assert expected in md_text
    if train_subset is None and val_subset is None:
        assert "stride" not in md_text
    assert _assert_csv_reproduces_records(csv_text, tmp_path) == 2


def test_unscored_run_and_empty_directory_are_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="no run records"):
        load_runs(tmp_path)
    path = _write_run(tmp_path, "a")
    (path / "test_metrics.json").unlink()
    with pytest.raises(FileNotFoundError, match="never scored"):
        load_runs(tmp_path)


# --------------------------------------------------------------------------
# 2. The table reproduces the records exactly
# --------------------------------------------------------------------------

def _expected_row(run: dict, test: dict) -> dict:
    metrics = test["metrics"]["per_key"][0]
    return {
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
        "ece": test["benchmark_row"]["ece"],
        "uncertainty_error_spearman": test["benchmark_row"]["uncertainty_correlation"],
        "git_commit": run["git_commit"],
    }


def _parse(cell: str, like):
    if cell == "":
        return None
    if isinstance(like, bool) or isinstance(like, str):
        return cell
    return type(like)(cell)


def _assert_csv_reproduces_records(csv_text: str, runs_dir: Path) -> int:
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert [column for column, _ in benchmark.COLUMNS] == list(rows[0])
    for row in rows:
        run = json.loads((runs_dir / row["run"] / "run.json").read_text(encoding="utf-8"))
        test = json.loads((runs_dir / row["run"] / "test_metrics.json").read_text(encoding="utf-8"))
        for column, expected in _expected_row(run, test).items():
            actual = _parse(row[column], expected) if expected is not None else _parse(row[column], "")
            assert actual == expected, f"{row['run']}.{column}: table {actual!r} != record {expected!r}"
    return len(rows)


def test_csv_reproduces_the_committed_run_records_exactly():
    if not RUNS.is_dir():
        pytest.skip("no experiments/runs/")
    csv_text, _, _ = build(RUNS)
    n_rows = _assert_csv_reproduces_records(csv_text, RUNS)
    assert n_rows == len([p for p in RUNS.iterdir() if (p / "run.json").is_file()])


def test_uncertainty_columns_carry_phase_2_values_when_present(tmp_path):
    _write_run(tmp_path, "a")
    _write_run(tmp_path, "b", run_changes={"model": "BLIP", "phase": "blip"},
               test_changes={"benchmark_row.ece": 0.0421, "benchmark_row.uncertainty_correlation": 0.613})
    csv_text, md_text, _ = build(tmp_path)
    rows = {row["run"]: row for row in csv.DictReader(io.StringIO(csv_text))}
    assert rows["a"]["ece"] == rows["a"]["uncertainty_error_spearman"] == ""
    assert float(rows["b"]["ece"]) == 0.0421 and float(rows["b"]["uncertainty_error_spearman"]) == 0.613
    assert _assert_csv_reproduces_records(csv_text, tmp_path) == 2
    assert "| 0.0421 | 0.6130 |" in md_text


def test_markdown_shows_each_value_and_each_commit(tmp_path):
    _write_run(tmp_path, "a")
    _, md_text, _ = build(tmp_path)
    row = next(line for line in md_text.splitlines() if line.startswith("| a |"))
    assert "| 0.0531 | 0.0678 | 0.1196 | 0.2040 |" in row
    assert "| 0.34 h | 0.512 |" in row  # 1234.5678 s; 0.000512 s/frame
    assert "`0123456`" in row
    assert "[experiments/runs/a/](../experiments/runs/a/)" in md_text


@pytest.mark.parametrize("suffix", [".csv", ".md"])
def test_committed_report_matches_a_fresh_build(suffix):
    """reports/phase1_benchmark.{csv,md} are regenerated whenever a run record changes."""
    if not RUNS.is_dir():
        pytest.skip("no experiments/runs/")
    csv_text, md_text, _ = build(RUNS)
    committed = REPORT.with_suffix(suffix).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert committed == (csv_text if suffix == ".csv" else md_text), (
        f"{REPORT.with_suffix(suffix).name} is out of date - run `python -m ml.training.benchmark`"
    )


def test_main_writes_both_files(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, "a")
    out = tmp_path / "out" / "bench"
    assert main(["--runs-dir", str(runs), "--out", str(out)]) == 0
    csv_text = out.with_suffix(".csv").read_text(encoding="utf-8")
    assert _assert_csv_reproduces_records(csv_text, runs) == 1
    assert out.with_suffix(".md").read_text(encoding="utf-8").startswith("# Phase 1 benchmark")
