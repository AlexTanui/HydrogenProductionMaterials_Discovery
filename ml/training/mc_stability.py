"""Monte Carlo sample-count stability (SCRUM-49): the smallest number of MC
samples at which a stochastic model's uncertainty metrics stop changing.

For one trained checkpoint, on a fixed stride subset of the **validation**
split - this chooses a protocol value, so the test split is never read:

1. For each seed, draw `max(ks)` stochastic passes per frame and form the
   predictive mean and variance from the first K of them, for every K in
   `ks` (`prefix_distributions`). Nested prefixes cost one run of the
   largest K instead of one run per K, and keep the comparison between K
   values from being swamped by independent sampling noise. Each prefix is
   exactly what `predict_distribution(num_mc_samples=K)` returns under the
   same seed - tests/ml/test_mc_stability.py pins that - but that method
   returns only the mean and variance, not the passes, hence the loop here.
2. Score every (seed, K) with `UQAccumulator`, the same UQ protocol v1.0
   path evaluate.py uses, for energy and force.
3. Apply the stability criterion (`stable_k`): K is stable when each judged
   metric's mean over seeds is within `tolerance` of its value at the
   largest K, and its seed-to-seed spread is within `tolerance` too. The
   answer is the smallest K from which every larger K is also stable.
   The criterion and its 0.01 default are a proposal (SCRUM-49 needs it
   agreed), so both are arguments, not constants.

    python -m ml.training.mc_stability --checkpoint experiments/checkpoints/phase2_blip_nll.pt \\
        --out reports/mc_stability/phase2_blip_nll.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from torch_geometric.loader import DataLoader

from ml.data.datasets import MD17Dataset
from ml.models.registry import load_model
from ml.training.evaluate import ROOT_DIR
from ml.training.train import stride_subset
from ml.utils.contract import ModelOutput, ReferenceData, Units
from ml.utils.uq_metrics import UQAccumulator

DEFAULT_KS = (2, 5, 10, 20, 50, 100)  # K=1 has no variance (unbiased estimator)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_FRAMES = 2000
DEFAULT_TOLERANCE = 0.01
# Judged by the criterion: bounded in [0, 1] (Spearman in [-1, 1]), so one
# absolute tolerance means the same thing for each. Sharpness and Gaussian
# NLL are reported alongside but not judged: both carry units, so a single
# tolerance doesn't fit them.
JUDGED_METRICS = ("ece", "coverage_90", "spearman")


def prefix_distributions(model, batch, ks) -> dict[int, tuple[torch.Tensor, ...]]:
    """{K: (E_mean, E_var, F_mean, F_var)} for each K in `ks`, from the first K
    of `max(ks)` stochastic passes - absolute units, as
    `ScaledEnergyModel.predict_distribution` returns them."""
    inner = model.model
    energies, forces = [], []
    for _ in range(max(ks)):  # the loop BLIP/GraphStochastic.predict_distribution run
        with torch.enable_grad():
            pos = batch.pos.detach().requires_grad_(True)
            e, f = inner.predict_energy_and_forces(batch.x, pos, batch.edge_index, batch.edge_attr,
                                                   batch=batch.batch)
        energies.append(e.detach())
        forces.append(f.detach())
    e_stack, f_stack = torch.stack(energies, dim=0), torch.stack(forces, dim=0)
    n_atoms = torch.bincount(batch.batch, minlength=e_stack.shape[1]).to(e_stack.dtype)
    scale, shift = model.scale, model.shift
    out = {}
    for k in ks:
        e, f = e_stack[:k], f_stack[:k]
        out[k] = (scale * e.mean(0) + n_atoms * shift, scale * scale * e.var(0, unbiased=True),
                  scale * f.mean(0), scale * scale * f.var(0, unbiased=True))
    return out


def _summary(block: dict) -> dict[str, Optional[float]]:
    return {"ece": block["ece"], "coverage_90": block["coverage"]["90%"],
            "spearman": block["spearman_uncertainty_error"], "sharpness": block["sharpness_mean_std"],
            "nll_gaussian": block["nll_gaussian"]}


@torch.no_grad()
def measure(model, dataset, ks, seeds, batch_size: int = 32, device: str = "cpu") -> dict:
    """{seed: {K: {"energy": {metric: value}, "force": {...}}}} over `dataset`."""
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    units = Units()
    results: dict = {}
    for seed in seeds:
        accumulators = {k: UQAccumulator() for k in ks}
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devices):  # as evaluate.py: the caller's RNG is untouched
            torch.manual_seed(seed)
            for batch in loader:
                batch = batch.to(device)
                reference = ReferenceData(E=batch.y, F=batch.force, batch=batch.batch,
                                          molecule=dataset.molecule, theory=dataset.theory,
                                          units=units, z=batch.x.squeeze(-1))
                for k, (e_mean, e_var, f_mean, f_var) in prefix_distributions(model, batch, ks).items():
                    accumulators[k].update(ModelOutput(E=e_mean, F=f_mean, E_var=e_var, F_var=f_var, units=units),
                                           reference)
        results[seed] = {}
        for k, accumulator in accumulators.items():
            record = accumulator.compute()["per_key"][0]
            results[seed][k] = {q: _summary(record[q]) for q in ("energy", "force")}
    return results


def stable_k(results: dict, tolerance: float = DEFAULT_TOLERANCE,
             metrics=JUDGED_METRICS) -> dict[str, Any]:
    """The smallest K from which every larger K is within `tolerance` of the
    largest K (mean over seeds) with seed spread (population std) within it.

    The largest K is the reference, so it can only fail on spread. An answer
    equal to it means no smaller K passed - raise the reference, don't
    adopt it as "enough"."""
    seeds = sorted(results)
    ks = sorted(results[seeds[0]])
    reference_k = ks[-1]

    def values(k, quantity, metric):
        return [results[s][k][quantity][metric] for s in seeds]

    rows = []
    for k in ks:
        worst = {"gap": 0.0, "spread": 0.0}
        undefined = []
        for quantity in ("energy", "force"):
            for metric in metrics:
                now, ref = values(k, quantity, metric), values(reference_k, quantity, metric)
                if None in now or None in ref:
                    undefined.append(f"{quantity}.{metric}")
                    continue
                worst["gap"] = max(worst["gap"], abs(statistics.fmean(now) - statistics.fmean(ref)))
                worst["spread"] = max(worst["spread"], statistics.pstdev(now))
        rows.append({"k": k, "max_gap_to_reference": worst["gap"], "max_seed_spread": worst["spread"],
                     "undefined": undefined,
                     "stable": not undefined and worst["gap"] <= tolerance and worst["spread"] <= tolerance})
    answer = None
    for row in reversed(rows):
        if not row["stable"]:
            break
        answer = row["k"]
    return {"stable_k": answer, "reference_k": reference_k, "tolerance": tolerance,
            "judged_metrics": list(metrics), "rows": rows}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SCRUM-49: MC sample count at which UQ metrics stop changing.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="val frames, a stride subset")
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if min(args.ks) < 2:
        parser.error("every K must be at least 2: one sample has no variance")

    model, checkpoint = load_model(args.checkpoint, map_location="cpu")
    if not model.supports_distribution:
        parser.error(f"{args.checkpoint}: no predict_distribution - nothing to measure")
    data = checkpoint["data"]
    gold = Path(data["gold_path"])
    gold = gold if gold.is_absolute() else ROOT_DIR / gold
    dataset = MD17Dataset(gold, split="val", cutoff_radius=data["cutoff_radius"], num_rbf=data["num_rbf"])
    positions, stride, offset = stride_subset(len(dataset), min(args.frames, len(dataset)), seed=42)
    subset = torch.utils.data.Subset(dataset, positions.tolist())
    subset.molecule, subset.theory = dataset.molecule, dataset.theory

    results = measure(model, subset, sorted(args.ks), args.seeds, args.batch_size, args.device)
    verdict = stable_k(results, args.tolerance)
    print(f"{args.checkpoint}: {len(positions)} val frames (stride {stride}, offset {offset}), "
          f"seeds {args.seeds}, reference K={verdict['reference_k']}, tolerance {args.tolerance}")
    for row in verdict["rows"]:
        k = row["k"]
        cells = "  ".join(
            f"{q[0].upper()}.{m} {statistics.fmean(results[s][k][q][m] for s in args.seeds):.4f}"
            for q in ("energy", "force") for m in ("ece", "coverage_90", "spearman")
            if None not in [results[s][k][q][m] for s in args.seeds]
        )
        print(f"K={k:>4}  gap {row['max_gap_to_reference']:.4f}  spread {row['max_seed_spread']:.4f}  "
              f"{'stable' if row['stable'] else '      '}  {cells}")
    print(f"smallest stable K: {verdict['stable_k']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "checkpoint": str(args.checkpoint), "split": "val", "frames": len(positions),
            "stride": stride, "offset": offset, "ks": sorted(args.ks), "seeds": args.seeds,
            "results": {str(s): {str(k): v for k, v in r.items()} for s, r in results.items()},
            "verdict": verdict,
        }, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
