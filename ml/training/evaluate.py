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

Absolute energies need nothing here: `load_model` returns the model wrapped
in the harness's one `ScaledEnergyModel` (ml/models/scaling.py), with the
training subset's shift and scale restored from the checkpoint's buffers.

The output records a fingerprint of the scored data (`data.fingerprint`:
sha256 of the split's indices and of z, R, E, F at them), so a benchmark
can tell scorings on different data apart even when their size matches.

The scoring pass runs under a fixed, recorded RNG seed (`seed`, default 0)
in a forked RNG state, so a stochastic model such as BLIP scores the same
every time, and the caller's RNG is untouched. Deterministic models draw
no random numbers and are unaffected.

A model that exposes `predict_distribution` (checked generically via
`ScaledEnergyModel.supports_distribution`, no per-model branch) is scored
by its MC-sample predictive mean/variance rather than one stochastic
sample: `ModelOutput.E_var`/`F_var` get populated for real, the point-
estimate metrics (energy/force MAE etc.) are computed on that same mean
so the accuracy and uncertainty numbers describe one predictive
distribution, and `ml.utils.uq_metrics.UQAccumulator` scores calibration,
coverage, Spearman correlation, a risk-coverage curve, Gaussian NLL/CRPS,
and sharpness. See that module's docstring for the Normal-distribution
assumption this rests on - provisional pending Dongxiao's sign-off
(SCRUM-39) - and ml/models/blip.py's for why this makes evaluation
`num_mc_samples` times more expensive for BLIP specifically. A model
without `predict_distribution` is scored exactly as before: one forward
pass, `E_var`/`F_var` stay `None`, and `uq_metrics` is `null`.

The MC sample count is read once from the model's own `num_mc_samples`,
passed to `predict_distribution` explicitly, and recorded as
`num_mc_samples` (null for a deterministic model), so the benchmark can
require one count across every stochastic run in a table (CLAUDE.md;
SCRUM-49 sets the project value). A distribution model with no
`num_mc_samples` is still scored, with a warning and a null count - the
benchmark then refuses it.

Optionally (`predictions_out`, CLI `--save-predictions PATH`; off by
default) the per-frame predictions are written to one `.npz` for plotting:
each frame's gold index, E_ref/E_pred, F_ref/F_pred per atom, and E_var/
F_var for a stochastic model, plus a JSON `metadata` entry (checkpoint and
its sha256, scoring-data fingerprint, split, seed, num_mc_samples, batch
size, units). They are the exact tensors the accumulator was given, kept
during the one scoring pass - nothing is recomputed, so a plot made from
them cannot disagree with the scores. Off, the output is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from ml.data.datasets import MD17Dataset
from ml.models.registry import load_model
from ml.utils.contract import ModelOutput, ReferenceData, Units
from ml.utils.metrics import MetricAccumulator
from ml.utils.uq_metrics import UQAccumulator

__all__ = ["evaluate_checkpoint", "main"]

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT_DIR / "experiments" / "results"

SCHEMA_VERSION = 1

# Version of the optional per-frame predictions .npz (see the module docstring).
PREDICTIONS_SCHEMA_VERSION = 1

# Seed for the scoring pass's RNG. Only stochastic models consume it.
DEFAULT_SEED = 0

# How closely a re-score must reproduce a stored metric, relative. Scoring on
# a multithreaded CPU is not bit-reproducible: the force backward pass sums in
# a thread-dependent order (measured up to ~7e-9 relative between scorings at
# one thread count, ~1.2e-6 when the thread count changes). 1e-5 leaves room
# for another machine and is ~100x below the 4 decimals the benchmark reports.
REPRODUCIBILITY_RTOL = 1e-5

# Which of the eight metric fields the single-number benchmark row uses.
# Both are the MD17/SchNet/PaiNN reporting convention, and both are named
# explicitly in the output so no reader has to guess which one produced a
# number (see ml/utils/metrics.py).
BENCHMARK_ENERGY_FIELD = "energy_mae_total"
BENCHMARK_FORCE_FIELD = "force_mae_component"

_UQ_NOTE_DETERMINISTIC = (
    "ECE and uncertainty-error correlation are defined by Dongxiao; this model has "
    "no predict_distribution (deterministic, or Phase 2/3 hasn't wired one in yet), "
    "so there is no uncertainty to score against that definition"
)
_UQ_NOTE_PROVISIONAL = (
    "computed under ml.utils.uq_metrics's provisional definition (Normal "
    "distribution assumed from E_var/F_var) - not yet Dongxiao's confirmed "
    "definition; see that module's docstring and SCRUM-39"
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


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def scoring_data_fingerprint(dataset: MD17Dataset, gold_path: Path, split: str) -> dict[str, Any]:
    """sha256 of exactly what was scored: the split's indices, and z, R, E, F at them.

    Hashed as the dataset holds them for scoring (z int64; R, E, F float32),
    so two scorings share a fingerprint only if they saw the same frames with
    the same values. Molecule, theory and frame count alone cannot tell a
    different gold file of the same size apart; this can.
    """
    with np.load(gold_path, allow_pickle=False) as gold:
        indices = np.asarray(gold[f"{split}_idx"], dtype=np.int64)
    if len(indices) != len(dataset):
        raise ValueError(f"{split}_idx has {len(indices)} entries but the dataset holds {len(dataset)} frames")
    arrays = {"indices": indices, "z": dataset.z.numpy(), "R": dataset.R.numpy(),
              "E": dataset.E.numpy(), "F": dataset.F.numpy()}
    parts = {name: _sha256(array) for name, array in arrays.items()}
    return {
        "algorithm": "sha256",
        "sha256": hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest(),
        "parts": parts,
        "dtypes": {name: str(array.dtype) for name, array in arrays.items()},
        "shapes": {name: list(array.shape) for name, array in arrays.items()},
    }


def _collect_predictions(collected: dict[str, list[np.ndarray]], prediction: ModelOutput,
                         reference: ReferenceData, n_atoms: int) -> None:
    """Keep one batch's accumulator inputs, as they were handed to it: energies
    per frame, forces (and variances) as (frames, atoms, 3)."""
    n_frames = reference.n_frames
    # The reshape below is only right if atoms are grouped frame by frame,
    # n_atoms each, in batch order - check it rather than assume it.
    expected = torch.arange(n_frames, device=reference.batch.device).repeat_interleave(n_atoms)
    if not torch.equal(reference.batch, expected):
        raise ValueError("batch atoms are not grouped frame by frame; cannot export per-atom predictions")

    def per_frame(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    def per_atom(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().reshape(n_frames, n_atoms, 3)

    collected["E_ref"].append(per_frame(reference.E))
    collected["E_pred"].append(per_frame(prediction.E))
    collected["F_ref"].append(per_atom(reference.F))
    collected["F_pred"].append(per_atom(prediction.F))
    if prediction.E_var is not None:
        collected["E_var"].append(per_frame(prediction.E_var))
    if prediction.F_var is not None:
        collected["F_var"].append(per_atom(prediction.F_var))


def _write_predictions(path: Path, gold_index: np.ndarray, collected: dict[str, list[np.ndarray]],
                       metadata: dict[str, Any]) -> str:
    """One .npz at exactly `path`, written atomically. Returns its sha256."""
    arrays = {name: np.concatenate(chunks) for name, chunks in collected.items() if chunks}
    if any(len(array) != len(gold_index) for array in arrays.values()):
        raise ValueError(f"collected {len(arrays['E_ref'])} frames but the split has {len(gold_index)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    # A file object, not a path: np.savez would append ".npz" to a path without it.
    with open(partial, "wb") as f:
        np.savez_compressed(f, gold_index=gold_index, **arrays,
                            metadata=np.array(json.dumps(metadata, sort_keys=True)))
    os.replace(partial, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    gold_path: Optional[str | Path] = None,
    split: str = "test",
    batch_size: int = 32,
    device: str = "cpu",
    num_workers: int = 0,
    units: Optional[Units] = None,
    seed: int = DEFAULT_SEED,
    predictions_out: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Score one checkpoint and return the full result record.

    Args:
        predictions_out: if set, also write the per-frame predictions to this
            .npz (module docstring) and add `predictions` (path, sha256) to
            the result. Default None: nothing is written and the result is
            unchanged.
        seed: seeds torch's RNG for the scoring pass, inside a forked RNG
            state so the caller's RNG is left exactly as it was. A
            stochastic model (BLIP draws fresh weight noise on every
            forward pass) therefore scores identically every time it is
            scored with the same seed; a deterministic model draws nothing
            and is unaffected. Recorded in the output.
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
    # Everything that can draw from torch's RNG runs inside one forked state,
    # restored on exit: rebuilding the model (its layers initialise randomly
    # before the checkpoint's weights load), iterating the DataLoader (it
    # draws a base seed) and a stochastic forward pass. torch.manual_seed
    # reseeds the CPU generator and every CUDA device's, so all of them are
    # forked - not only the device being scored on.
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        return _score_checkpoint(
            checkpoint_path, gold_path, split, batch_size, device, num_workers, units, seed,
            predictions_out,
        )


def _score_checkpoint(
    checkpoint_path: str | Path,
    gold_path: Optional[str | Path],
    split: str,
    batch_size: int,
    device: str,
    num_workers: int,
    units: Optional[Units],
    seed: int,
    predictions_out: Optional[str | Path] = None,
) -> dict[str, Any]:
    """evaluate_checkpoint's body; always called inside its forked RNG state."""
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
    scores_distribution = model.supports_distribution
    # Resolved once and passed explicitly, so the record states the count
    # that actually ran. None for a deterministic model: nothing to count.
    num_mc_samples = None
    if scores_distribution:
        num_mc_samples = getattr(model.model, "num_mc_samples", None)
        if num_mc_samples is None:
            warnings.append(
                f"{type(model.model).__name__} scores a distribution but has no num_mc_samples, so the "
                "MC sample count used is not recorded; benchmark.py refuses this run until it is "
                "(SCRUM-49)."
            )
        else:
            num_mc_samples = int(num_mc_samples)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    # shuffle=False: evaluation is order-independent by construction (the
    # accumulator sums), but a fixed order keeps timings comparable run to run.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    accumulator = MetricAccumulator()
    uq_accumulator = UQAccumulator()
    reference_units = units or Units()
    # Per-frame export, opt-in: filled from the same objects the accumulator
    # gets, inside this one pass. None (the default) skips it entirely.
    collected: Optional[dict[str, list[np.ndarray]]] = (
        {name: [] for name in ("E_ref", "E_pred", "E_var", "F_ref", "F_pred", "F_var")}
        if predictions_out is not None else None
    )

    model_seconds = 0.0
    n_batches = 0
    wall_start = time.perf_counter()

    # Seeded after the model is rebuilt, so the scoring pass's random draws do
    # not depend on how much RNG reconstruction consumed. (The caller's state
    # is preserved by evaluate_checkpoint's fork.)
    torch.manual_seed(seed)

    # no_grad for the loop, but grad back on around the model call: forces are
    # F = -dE/dR, so that call is the one place a graph is needed. Not every
    # model re-enables grad internally - MPNN and SchNet do not, and raise
    # under no_grad - and evaluation must not depend on which ones do.
    # predict_distribution manages its own grad context internally (BLIP's
    # own @torch.no_grad() with a local enable_grad() per MC sample), so it
    # needs no outer enable_grad() here the way the point-estimate path does.
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(torch_device)

            model_start = time.perf_counter()
            if scores_distribution:
                e_mean, e_var, f_mean, f_var = model.predict_distribution(
                    batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch,
                    num_mc_samples=num_mc_samples,
                )
                energy, forces = e_mean.detach(), f_mean.detach()
                energy_var, force_var = e_var.detach(), f_var.detach()
            else:
                with torch.enable_grad():
                    energy, forces = model.predict_energy_and_forces(
                        batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch
                    )
                energy, forces = energy.detach(), forces.detach()
                energy_var = force_var = None
            if torch_device.type == "cuda":
                torch.cuda.synchronize()
            model_seconds += time.perf_counter() - model_start

            prediction = ModelOutput(
                E=energy, F=forces, E_var=energy_var, F_var=force_var, units=reference_units
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
            if prediction.is_stochastic:
                uq_accumulator.update(prediction, reference)
            if collected is not None:
                _collect_predictions(collected, prediction, reference, int(dataset.z.shape[0]))
            n_batches += 1

    wall_seconds = time.perf_counter() - wall_start
    metrics = accumulator.compute()
    uq_metrics = uq_accumulator.compute()  # None if the model never produced a distribution
    record = metrics["per_key"][0]
    n_configs = int(record["n_frames"])

    uq_record = uq_metrics["per_key"][0] if uq_metrics is not None else None
    benchmark_ece = uq_record["energy"]["ece"] if uq_record and uq_record["energy"] else None
    benchmark_correlation = (
        uq_record["energy"]["spearman_uncertainty_error"] if uq_record and uq_record["energy"] else None
    )
    fingerprint = scoring_data_fingerprint(dataset, resolved_gold, split)

    result = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(checkpoint_path),
        "phase": checkpoint["phase"],
        "model_config": checkpoint["model_config"],
        "energy_scaling": model.energy_scaling(),
        "seed": seed,
        # Eval MC sample count actually used; null for a deterministic model.
        "num_mc_samples": num_mc_samples,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "data": {
            "gold_path": str(resolved_gold),
            "split": split,
            "molecule": dataset.molecule,
            "theory": dataset.theory,
            "fingerprint": fingerprint,
            "cutoff_radius": cutoff_radius,
            "num_rbf": num_rbf,
            "n_configs": n_configs,
            "n_atoms_per_config": int(dataset.z.shape[0]),
        },
        "metrics": metrics,
        "uq_metrics": uq_metrics,
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
            "model": type(model.model).__name__,  # the architecture, not the shared wrapper
            "energy_mae": record[BENCHMARK_ENERGY_FIELD],
            "energy_mae_field": BENCHMARK_ENERGY_FIELD,
            "energy_mae_unit": record["units"]["energy"],
            "force_mae": record[BENCHMARK_FORCE_FIELD],
            "force_mae_field": BENCHMARK_FORCE_FIELD,
            "force_mae_unit": record["units"]["force"],
            "ece": benchmark_ece,
            "uncertainty_correlation": benchmark_correlation,
            "uncertainty_note": _UQ_NOTE_PROVISIONAL if uq_record else _UQ_NOTE_DETERMINISTIC,
        },
        "warnings": warnings,
    }

    if collected is not None:
        predictions_path = Path(predictions_out)
        # Scoring order is the split's own order (shuffle=False), so frame k
        # is the k-th entry of {split}_idx - the same array the fingerprint hashes.
        with np.load(resolved_gold, allow_pickle=False) as gold:
            gold_index = np.asarray(gold[f"{split}_idx"], dtype=np.int64)
        energy_unit, force_unit = record["units"]["energy"], record["units"]["force"]
        metadata = {
            "schema_version": PREDICTIONS_SCHEMA_VERSION,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "phase": checkpoint["phase"],
            "molecule": dataset.molecule,
            "theory": dataset.theory,
            "split": split,
            "seed": seed,
            "num_mc_samples": num_mc_samples,
            "batch_size": batch_size,
            "units": record["units"],
            "data_fingerprint": fingerprint,
            "arrays": {
                "gold_index": f"index into the gold file's arrays ({split}_idx), in scoring order",
                "E_ref, E_pred": f"total energy per frame, {energy_unit}",
                "E_var": f"predicted energy variance per frame, ({energy_unit})^2 - stochastic models only",
                "F_ref, F_pred": f"force per atom and component, (frames, atoms, 3), {force_unit}",
                "F_var": f"predicted force variance, (frames, atoms, 3), ({force_unit})^2 - stochastic models only",
            },
        }
        sha = _write_predictions(predictions_path, gold_index, collected, metadata)
        result["predictions"] = {"path": str(predictions_path), "sha256": sha}
    return result


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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="RNG seed for the scoring pass (matters only for stochastic models)")
    parser.add_argument(
        "--output",
        default=None,
        help="default experiments/results/<checkpoint stem>.json",
    )
    parser.add_argument("--save-predictions", metavar="PATH", default=None,
                        help="also write per-frame predictions to this .npz (off by default)")
    args = parser.parse_args(argv)

    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        gold_path=args.gold_path,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        predictions_out=args.save_predictions,
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
    if row["ece"] is not None:
        print(f"  energy ECE / Spearman(std, |err|)     : "
              f"{row['ece']:.4g} / {row['uncertainty_correlation']:.4g}"
              if row["uncertainty_correlation"] is not None
              else f"{row['ece']:.4g} / n/a")
    print(f"  written to {output}")
    if "predictions" in result:
        print(f"  per-frame predictions written to {result['predictions']['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
