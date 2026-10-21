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
    # Still must-match after 48f10a4 exempted MODEL_SPECIFIC_PROTOCOL_KEYS.
    ("run", "protocol.optimizer.lr", 0.001, "training loop.optimizer"),
    ("run", "protocol.force_loss_weight", 10.0, "training loop.force_loss_weight"),
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
    # phase1_ only: experiments/runs/ also holds Phase 2/3 runs (phase2_blip)
    # under a genuinely different protocol - reports/phase1_benchmark is
    # scoped to the Phase 1 bake-off, same default main() uses.
    csv_text, _, _ = build(RUNS, name_prefix="phase1_")
    n_rows = _assert_csv_reproduces_records(csv_text, RUNS)
    assert n_rows == len([p for p in RUNS.iterdir() if p.name.startswith("phase1_") and (p / "run.json").is_file()])


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
    csv_text, md_text, _ = build(RUNS, name_prefix="phase1_")
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


# --------------------------------------------------------------------------
# 3. Model-specific protocol keys (48f10a4, SCRUM-62 review)
# --------------------------------------------------------------------------
# xfail(strict=True) marks a defect the review found: the test states the
# behaviour we want, so fixing the defect turns it into an XPASS failure
# and the marker has to come off.

# train.py's own protocol.loss text on the MSE path - what every real record carries.
_MSE_LOSS = ("energy_loss_weight * MSE(E) + force_loss_weight * MSE(F), absolute E, plus kl_weight * "
             "model.kl_divergence() / n_batches when the model defines kl_weight (BLIP only; see "
             "ml/models/blip.py)")


def _md_cell(md_text: str, run: str, header: str) -> str:
    """One cell of the main Markdown table (the first table row for `run`)."""
    row = next(line for line in md_text.splitlines() if line.startswith(f"| {run} |"))
    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
    return cells[[h for _, h in benchmark.COLUMNS].index(header)]


@pytest.mark.parametrize("other_loss", [
    # PhysNet's own trainer (ml/training/train_physnet.py) uses L1 - SCRUM-40.
    _MSE_LOSS.replace("MSE(E)", "L1(E)").replace("MSE(F)", "L1(F)"),
    _MSE_LOSS.replace("absolute E", "per-atom E"),
], ids=["l1", "per_atom"])
def test_a_different_base_loss_is_refused(tmp_path, other_loss):
    _write_run(tmp_path, "a_reference", run_changes={"protocol.loss": _MSE_LOSS})
    _write_run(tmp_path, "b_different", run_changes={"protocol.loss": other_loss})
    with pytest.raises(ProtocolMismatch, match=r"b_different: training loop.loss ="):
        build(tmp_path)


def test_a_model_specific_loss_override_is_reported_not_refused(tmp_path):
    # MPNN next to an NLL-trained BLIP: same base loss, BLIP's replacement
    # recorded separately - tabulated, with the override shown under Loss.
    _write_run(tmp_path, "a_mpnn", run_changes={"protocol.loss": _MSE_LOSS, "protocol.loss_override": None})
    _write_run(tmp_path, "b_blip_nll", run_changes={"protocol.loss": _MSE_LOSS,
                                                    "protocol.loss_override": "Gaussian NLL after warmup",
                                                    "protocol.train_num_mc_samples": 4})
    _, md_text, rows = build(tmp_path)
    assert [row["run"] for row in rows] == ["a_mpnn", "b_blip_nll"]
    assert "b_blip_nll replaces it (model-specific `loss_override`, not checked): Gaussian NLL after warmup." in md_text
    assert "a_mpnn replaces it" not in md_text


def test_a_run_recorded_before_the_nll_hooks_shares_a_table_with_one_after(tmp_path):
    # The phase1_* records predate train_num_mc_samples/nll_warmup_epochs
    # (3fe964c); a model retrained today records them at their "off" values.
    # Retraining one model must not force a retrain of the rest.
    _write_run(tmp_path, "a_old", run_changes={"protocol.loss": _MSE_LOSS, "protocol.kl_weight": 0.0})
    _write_run(tmp_path, "b_new", run_changes={"protocol.loss": _MSE_LOSS, "protocol.kl_weight": 0.0,
                                               "protocol.train_num_mc_samples": 1,
                                               "protocol.nll_warmup_epochs": 0})
    csv_text, _, rows = build(tmp_path)
    assert [row["run"] for row in rows] == ["a_old", "b_new"]
    # Copied from the records, not normalised: absent -> blank, recorded -> its value.
    cells = {row["run"]: row for row in csv.DictReader(io.StringIO(csv_text))}
    assert (cells["a_old"]["train_num_mc_samples"], cells["a_old"]["nll_warmup_epochs"]) == ("", "")
    assert (cells["b_new"]["train_num_mc_samples"], cells["b_new"]["nll_warmup_epochs"]) == ("1", "0")


def test_mpnn_and_blip_with_different_kl_weights_share_a_table(tmp_path):
    _write_run(tmp_path, "phase1_baseline", run_changes={"model": "MPNN", "phase": "mpnn",
                                                         "protocol.kl_weight": 0.0})
    _write_run(tmp_path, "phase2_blip", run_changes={"model": "BLIP", "phase": "blip",
                                                     "protocol.kl_weight": 0.01})
    csv_text, md_text, _ = build(tmp_path)
    cells = {row["run"]: row for row in csv.DictReader(io.StringIO(csv_text))}
    assert float(cells["phase1_baseline"]["kl_weight"]) == 0.0
    assert float(cells["phase2_blip"]["kl_weight"]) == 0.01
    assert _md_cell(md_text, "phase1_baseline", "KL weight") == "0.0000"
    assert _md_cell(md_text, "phase2_blip", "KL weight") == "0.0100"


@pytest.mark.xfail(strict=True, reason="_md_value's 4-dp float format prints a kl_weight below 5e-5 as 0.0000")
def test_a_small_kl_weight_is_distinguishable_from_zero_in_the_markdown(tmp_path):
    _write_run(tmp_path, "a", run_changes={"protocol.kl_weight": 0.0})
    _write_run(tmp_path, "b", run_changes={"protocol.kl_weight": 1e-5})
    _, md_text, _ = build(tmp_path)
    assert _md_cell(md_text, "a", "KL weight") != _md_cell(md_text, "b", "KL weight")


@pytest.mark.xfail(strict=True, reason="a protocol key in neither list silently becomes must-match (SCRUM-61 trap)")
def test_an_unclassified_protocol_key_is_refused_with_a_message_to_classify_it(tmp_path):
    # A future per-model hook that train.py records but nobody adds to
    # MODEL_SPECIFIC_PROTOCOL_KEYS: today the old record clashes with the
    # new one on values, which reads like a real protocol difference.
    _write_run(tmp_path, "a_old")
    _write_run(tmp_path, "b_new", run_changes={"protocol.ensemble_size": 1})
    with pytest.raises(ValueError, match=r"ensemble_size.*classif"):
        build(tmp_path)


def test_a_run_with_no_recorded_loss_is_refused(tmp_path):
    _write_run(tmp_path, "a")
    path = _write_run(tmp_path, "b")
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    del run["protocol"]["loss"]
    (path / "run.json").write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ProtocolMismatch, match=r"b: training loop.loss = None"):
        build(tmp_path)


def test_markdown_names_the_protocol_keys_that_were_not_checked(tmp_path):
    _write_run(tmp_path, "a")
    _, md_text, _ = build(tmp_path)
    claim = next(line for line in md_text.splitlines() if "checked to share one protocol" in line)
    for key in benchmark.MODEL_SPECIFIC_PROTOCOL_KEYS:
        assert key in claim, key


@pytest.mark.xfail(strict=True, reason="pre-dates 48f10a4: the eval MC sample count is only in model_config, "
                                       "which protocol_signature doesn't compare")
def test_runs_scored_with_different_mc_sample_counts_are_refused(tmp_path):
    # CLAUDE.md: same MC sample count across phases. evaluate.py uses the
    # model's own num_mc_samples, recorded only under config.model / model_config.
    for name, n in (("a_blip", 20), ("b_blip", 5)):
        _write_run(tmp_path, name, run_changes={"config.model": {"phase": "blip", "num_mc_samples": n}},
                   test_changes={"model_config": {"num_mc_samples": n}})
    with pytest.raises(ProtocolMismatch, match="num_mc_samples"):
        build(tmp_path)
