"""Score a trained checkpoint on the untouched test split (SCRUM-52 / SCRUM-50).

Every model in the Phase 1 bake-off gets its final numbers from this file,
so it is **model-agnostic by construction**: it rebuilds whatever model the
checkpoint names via `ml/models/registry.py` and calls the shared
`predict_energy_and_forces` interface. There is no per-model branch here,
and adding one is how a benchmark table quietly stops comparing like with
like.

Run:

    python -m ml.training.evaluate --checkpoint experiments/checkpoints/phase1_painn.pt

Three properties this module is responsible for:

1. **The test split is read here and nowhere else.** `MD17Dataset` is
   constructed with the split baked into the gold file (`test_idx`) - no
   re-splitting, no RNG, no shuffling. Any evaluation that re-splits gold
   data is a bug (see CLAUDE.md).
2. **Streaming accumulation is preserved.** Metrics come from
   `ml.utils.metrics.MetricAccumulator`, which keeps running sums and
   divides exactly once. Nothing here averages a per-batch metric value:
   that is wrong for unequal batch sizes and always wrong for RMSE, and
   the last batch of a split is almost always short.
3. **The checkpoint's identity is checked against the data it is scored
   on.** A checkpoint trained on ethanol/dft scored against aspirin/ccsd
   gold would otherwise produce a complete, believable, meaningless table.

Uncertainty metrics (ECE, uncertainty-error correlation) are deliberately
absent. Those definitions are Dongxiao's; the fields appear in the output
as `null` with a note rather than being invented here.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
from torch_geometric.loader import DataLoader

from ml.data.datasets import MD17Dataset
from ml.models.registry import load_model
from ml.utils.contract import ModelOutput, ReferenceData, Units
from ml.utils.metrics import MetricAccumulator

__all__ = ["evaluate_checkpoint", "main"]

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT_DIR / "experiments" / "results"

SCHEMA_VERSION = 1

# Which of the eight metric fields the single-number benchmark row uses.
# Both are the MD17/SchNet/PaiNN reporting convention, and both are named
# explicitly in the output so no reader has to guess which one produced a
# number (see ml/utils/metrics.py).
BENCHMARK_ENERGY_FIELD = "energy_mae_total"
BENCHMARK_FORCE_FIELD = "force_mae_component"

_UQ_NOTE = (
    "ECE and uncertainty-error correlation are defined by Dongxiao and are not "
    "computed here; a deterministic Phase 1 model has no uncertainty by "
    "construction. Populated for Phase 2/3 once those definitions land."
)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    return torch.device(name)


def _check_identity(dataset: MD17Dataset, checkpoint: dict[str, Any], gold_path: Path) -> None:
    """Refuse to score a checkpoint against data it was not trained on.

    Molecule and theory are the dataset's identity (glossary.md section 5):
    ethanol exists at both DFT and CCSD(T) and their absolute energies sit
    on different scales, so a mismatch here yields a number that looks fine
    and means nothing.
    """
    trained_on = checkpoint.get("data", {})
    for field, actual in (("molecule", dataset.molecule), ("theory", dataset.theory)):
        expected = trained_on.get(field)
        if expected is not None and str(expected) != str(actual):
            raise ValueError(
                f"checkpoint was trained on {field}={expected!r} but {gold_path.name} "
                f"holds {field}={actual!r}. Scoring across molecules or levels of "
                "theory produces a believable, meaningless number - pass the "
                "matching --gold-path."
            )


def _check_cutoff(model: torch.nn.Module, cutoff_radius: float) -> Optional[str]:
    """Warn if the graph radius differs from the one the model expects.

    Probed generically with `getattr`, so this stays model-agnostic: a model
    that does not expose `cutoff_radius` simply skips the check. A mismatch
    is not fatal - a model may legitimately apply its own envelope inside a
    wider graph - but it silently changes which neighbours contribute, so it
    is recorded in the output either way.
    """
    expected = getattr(model, "cutoff_radius", None)
    if expected is None or float(expected) == float(cutoff_radius):
        return None
    return (
        f"graph built at cutoff_radius={cutoff_radius} but the model expects "
        f"{float(expected)}; messages beyond the model's radius are dropped by "
        "its own cutoff, so this is not the architecture that was trained"
    )


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    gold_path: Optional[str | Path] = None,
    split: str = "test",
    batch_size: int = 32,
    device: str = "cpu",
    num_workers: int = 0,
    units: Optional[Units] = None,
) -> dict[str, Any]:
    """Score one checkpoint and return the full result record.

    Args:
        checkpoint_path: written by `ml.models.registry.save_checkpoint`.
        gold_path: overrides the path recorded in the checkpoint. The gold
            file carries its own train/val/test indices; they are used as-is.
        split: `test` for reported results. Anything else is allowed for
            development but is stamped into the output and warned about, so
            a non-test number can never be mistaken for a final one.
        units: the gold stage stores no `r_unit`/`e_unit` fields, so this
            defaults to the project units (kcal/mol, Angstrom) that
            `validate_sample` enforces at the silver step. Passed explicitly
            only if a future source arrives in something else - MAE in the
            wrong unit is wrong by a silent constant factor rather than
            visibly broken (CLAUDE.md).
    """
    checkpoint_path = Path(checkpoint_path)
    torch_device = _resolve_device(device)

    model, checkpoint = load_model(checkpoint_path, map_location=str(torch_device))
    model.to(torch_device)
    model.eval()

    data_config = dict(checkpoint.get("data", {}))
    resolved_gold = Path(gold_path) if gold_path is not None else Path(data_config["gold_path"])
    if not resolved_gold.is_absolute():
        resolved_gold = ROOT_DIR / resolved_gold
    cutoff_radius = float(data_config.get("cutoff_radius", 5.0))
    num_rbf = int(data_config.get("num_rbf", 16))

    # The split comes from the gold file's own `{split}_idx`. Contiguous
    # trajectory blocks, deterministic, no RNG - never re-derived here.
    dataset = MD17Dataset(
        resolved_gold, split=split, cutoff_radius=cutoff_radius, num_rbf=num_rbf
    )
    _check_identity(dataset, checkpoint, resolved_gold)
    cutoff_warning = _check_cutoff(model, cutoff_radius)

    warnings: list[str] = []
    if cutoff_warning:
        warnings.append(cutoff_warning)
    if split != "test":
        warnings.append(
            f"scored on the {split!r} split, not 'test'. This is not a reportable "
            "number for the bake-off table."
        )
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    # shuffle=False: evaluation is order-independent by construction (the
    # accumulator sums), but a fixed order keeps timings comparable run to run.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    accumulator = MetricAccumulator()
    reference_units = units or Units()

    model_seconds = 0.0
    n_batches = 0
    wall_start = time.perf_counter()

    # no_grad for the loop; the model re-enables grad internally for the
    # F = -dE/dR pass, which is the only place a graph is needed.
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(torch_device)

            model_start = time.perf_counter()
            energy, forces = model.predict_energy_and_forces(
                batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch
            )
            if torch_device.type == "cuda":
                torch.cuda.synchronize()
            model_seconds += time.perf_counter() - model_start

            prediction = ModelOutput(
                E=energy.detach(), F=forces.detach(), units=reference_units
            )
            reference = ReferenceData(
                E=batch.y,
                F=batch.force,
                batch=batch.batch,
                molecule=dataset.molecule,
                theory=dataset.theory,
                units=reference_units,
                z=batch.x.squeeze(-1),
            )
            # One running sum per key; the single division happens in compute().
            accumulator.update(prediction, reference)
            n_batches += 1

    wall_seconds = time.perf_counter() - wall_start
    metrics = accumulator.compute()
    record = metrics["per_key"][0]
    n_configs = int(record["n_frames"])

    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(checkpoint_path),
        "phase": checkpoint["phase"],
        "model_config": checkpoint["model_config"],
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "data": {
            "gold_path": str(resolved_gold),
            "split": split,
            "molecule": dataset.molecule,
            "theory": dataset.theory,
            "cutoff_radius": cutoff_radius,
            "num_rbf": num_rbf,
            "n_configs": n_configs,
            "n_atoms_per_config": int(dataset.z.shape[0]),
        },
        "metrics": metrics,
        "timing": {
            "eval_wall_clock_s": wall_seconds,
            "model_inference_s": model_seconds,
            # The headline per-config cost: forward pass plus the backward
            # pass that produces forces, excluding graph construction and
            # collation. `eval_wall_clock_s` includes those.
            "inference_s_per_config": model_seconds / n_configs,
            "wall_clock_s_per_config": wall_seconds / n_configs,
            "batch_size": batch_size,
            "n_batches": n_batches,
            # Training wall-clock is measured by the training loop and
            # carried through the checkpoint; it is not re-derivable here.
            "train_wall_clock_s": checkpoint.get("train", {}).get("wall_clock_s"),
        },
        "environment": {
            "device": str(torch_device),
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        # glossary.md section 4's /benchmarks row shape, so the backend can
        # read experiments/results/*.json without reshaping.
        "benchmark_row": {
            "phase": checkpoint["phase"],
            "model": type(model).__name__,
            "energy_mae": record[BENCHMARK_ENERGY_FIELD],
            "energy_mae_field": BENCHMARK_ENERGY_FIELD,
            "energy_mae_unit": record["units"]["energy"],
            "force_mae": record[BENCHMARK_FORCE_FIELD],
            "force_mae_field": BENCHMARK_FORCE_FIELD,
            "force_mae_unit": record["units"]["force"],
            "ece": None,
            "uncertainty_correlation": None,
            "uncertainty_note": _UQ_NOTE,
        },
        "warnings": warnings,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score a checkpoint on the untouched test split.",
    )
    parser.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    parser.add_argument("--gold-path", default=None, help="override the checkpoint's gold file")
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="default 'test'; anything else is stamped into the output as non-reportable",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output",
        default=None,
        help="default experiments/results/<checkpoint stem>.json",
    )
    args = parser.parse_args(argv)

    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        gold_path=args.gold_path,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
    )

    default_output = RESULTS_DIR / f"{Path(args.checkpoint).stem}.json"
    output = Path(args.output) if args.output else default_output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    row = result["benchmark_row"]
    timing = result["timing"]
    print(f"\n{result['phase']} on {result['data']['molecule']}/{result['data']['theory']} "
          f"[{result['data']['split']}], {result['data']['n_configs']} configs")
    print(f"  energy MAE ({row['energy_mae_field']}) : "
          f"{row['energy_mae']:.6g} {row['energy_mae_unit']}")
    print(f"  force  MAE ({row['force_mae_field']})  : "
          f"{row['force_mae']:.6g} {row['force_mae_unit']}")
    print(f"  inference                             : "
          f"{timing['inference_s_per_config'] * 1e3:.4g} ms/config")
    print(f"  written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
