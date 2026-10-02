"""THROWAWAY smoke-training script for PaiNN. NOT the bake-off training path.

    +---------------------------------------------------------------------+
    | This script exists for verification and compute profiling ONLY.      |
    |                                                                     |
    | Its purpose is to answer two questions: does the model actually      |
    | learn (does the loss go down), and what does a training step cost    |
    | in wall-clock and memory (SCRUM-49 compute profiling)?               |
    |                                                                     |
    | Nothing it prints is an official result. Numbers from this loop are  |
    | NOT comparable with the other four bake-off models, because those    |
    | models will be trained by the shared loop in ml/training/train.py    |
    | (SCRUM-51) under a shared schedule, optimiser and budget. Quoting a  |
    | MAE from here in a benchmark table would defeat the purpose of the   |
    | bake-off.                                                           |
    |                                                                     |
    | It writes no checkpoint to experiments/checkpoints/ and no JSON to   |
    | experiments/results/. Delete this file once SCRUM-51 lands.          |
    +---------------------------------------------------------------------+

**It reads the TRAIN split and nothing else.** The val and test splits are
never constructed here - not for early stopping, not for monitoring, not
for a sanity check. `_assert_train_only` enforces that structurally rather
than by convention.

Run:

    python scripts/smoke_train_painn.py --config experiments/configs/phase1_painn.yaml
    python scripts/smoke_train_painn.py --steps 100 --n-train 256    # quicker
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch_geometric.loader import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ml.data.datasets import MD17Dataset  # noqa: E402
from ml.models.painn import PaiNN  # noqa: E402

DEFAULT_CONFIG = ROOT_DIR / "experiments" / "configs" / "phase1_painn.yaml"

BANNER = (
    "=" * 78
    + "\n  SMOKE TRAINING - verification and compute profiling only."
    + "\n  Not the bake-off training path; these numbers are NOT official results."
    + "\n  Reads the TRAIN split only; val/test are never touched.\n"
    + "=" * 78
)


def _assert_train_only(split: str) -> str:
    """Hard guard. The test split is read only by ml/training/evaluate.py."""
    if split != "train":
        raise SystemExit(
            f"refusing to read the {split!r} split. This script profiles training "
            "only; the test split is scored exactly once, by "
            "ml/training/evaluate.py, and the val split belongs to the real "
            "training loop (SCRUM-51)."
        )
    return split


def _peak_rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / (1024 * 1024)


def smoke_train(
    config: dict[str, Any],
    steps: int,
    n_train: int,
    log_every: int,
    device_name: str,
) -> dict[str, Any]:
    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]

    torch.manual_seed(int(data_config.get("seed", 42)))
    device = torch.device(device_name)

    gold_path = ROOT_DIR / data_config["gold_path"]
    if not gold_path.exists():
        raise SystemExit(
            f"no gold data at {gold_path}. Run `python -m ml.data.preprocessing` "
            "first - do not invent stand-in data (CLAUDE.md)."
        )

    dataset = MD17Dataset(
        gold_path,
        split=_assert_train_only("train"),
        cutoff_radius=float(data_config["cutoff_radius"]),
        num_rbf=int(data_config["num_rbf"]),
    )
    # A contiguous head of the train split - still training frames only.
    n_train = min(n_train, len(dataset))
    subset = torch.utils.data.Subset(dataset, range(n_train))
    loader = DataLoader(subset, batch_size=int(train_config["batch_size"]), shuffle=True)

    model = PaiNN(
        hidden_channels=int(model_config["hidden_channels"]),
        num_layers=int(model_config["num_layers"]),
        num_rbf=int(model_config["num_rbf"]),
        cutoff_radius=float(model_config["cutoff_radius"]),
        max_atomic_number=int(model_config["max_atomic_number"]),
        eps=float(model_config["eps"]),
    ).to(device)

    # Centre the energy scale on the training subset, exactly as the real
    # loop must: a raw readout cannot reach ethanol's ~-97,000 kcal/mol, and
    # without this the loss is dominated by a constant offset the model
    # spends its whole budget chasing.
    energies = dataset.E[:n_train]
    n_atoms = int(dataset.z.shape[0])
    model.set_energy_statistics(
        shift=float(energies.mean()) / n_atoms,
        scale=float(energies.std().clamp(min=1e-6)) / n_atoms,
    )

    optimiser = torch.optim.Adam(model.parameters(), lr=float(train_config["lr"]))
    energy_weight = float(train_config["energy_loss_weight"])
    force_weight = float(train_config["force_loss_weight"])

    model.train()
    history: list[dict[str, float]] = []
    step_times: list[float] = []
    rss_before = _peak_rss_mb()
    peak_rss = rss_before
    wall_start = time.perf_counter()

    step = 0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            batch = batch.to(device)
            step_start = time.perf_counter()

            optimiser.zero_grad(set_to_none=True)
            energy, forces = model.predict_energy_and_forces(
                batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch
            )
            energy_loss = torch.nn.functional.mse_loss(energy, batch.y)
            force_loss = torch.nn.functional.mse_loss(forces, batch.force)
            loss = energy_weight * energy_loss + force_weight * force_loss
            # Backward through the force term needs the double-backward graph,
            # which model.train() enables via create_graph=True.
            loss.backward()
            optimiser.step()

            step_times.append(time.perf_counter() - step_start)
            peak_rss = max(peak_rss, _peak_rss_mb())
            # .detach() before float(): these tensors still carry the
            # double-backward graph, and converting one to a Python scalar
            # otherwise warns (and would pin the graph alive).
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "energy_mse": float(energy_loss.detach()),
                "force_mse": float(force_loss.detach()),
                "energy_mae": float((energy - batch.y).abs().mean().detach()),
                "force_mae": float((forces - batch.force).abs().mean().detach()),
            }
            history.append(record)
            if step % log_every == 0 or step == steps - 1:
                print(
                    f"  step {record['step']:4d}  loss {record['loss']:12.4f}   "
                    f"E MAE {record['energy_mae']:10.4f}   F MAE {record['force_mae']:8.4f}"
                )
            step += 1

    wall_seconds = time.perf_counter() - wall_start
    window = max(1, len(history) // 10)
    first, last = history[:window], history[-window:]

    return {
        "not_official_results": (
            "smoke run for verification and compute profiling only; not "
            "comparable with other bake-off models"
        ),
        "steps": len(history),
        "n_train_frames": n_train,
        "batch_size": int(train_config["batch_size"]),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "learning": {
            "first_window_mean_loss": sum(r["loss"] for r in first) / len(first),
            "last_window_mean_loss": sum(r["loss"] for r in last) / len(last),
            "first_window_force_mae": sum(r["force_mae"] for r in first) / len(first),
            "last_window_force_mae": sum(r["force_mae"] for r in last) / len(last),
            "window_size": window,
        },
        "compute": {
            "wall_clock_s": wall_seconds,
            "mean_step_s": sum(step_times) / len(step_times),
            "median_step_s": sorted(step_times)[len(step_times) // 2],
            "frames_per_s": (len(history) * int(train_config["batch_size"])) / wall_seconds,
            "rss_before_mb": rss_before,
            "peak_rss_mb": peak_rss,
            "rss_growth_mb": peak_rss - rss_before,
        },
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "history": history,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--steps", type=int, default=300, help="a few hundred is plenty")
    parser.add_argument("--n-train", type=int, default=2048, help="frames from the TRAIN split")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--report", default=None, help="optional JSON path (not under experiments/)"
    )
    args = parser.parse_args(argv)

    print(BANNER)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = smoke_train(
        config=config,
        steps=args.steps,
        n_train=args.n_train,
        log_every=args.log_every,
        device_name=args.device,
    )

    learning = result["learning"]
    compute = result["compute"]
    improved = learning["last_window_mean_loss"] < learning["first_window_mean_loss"]

    print("\n" + "-" * 78)
    print(f"  loss      {learning['first_window_mean_loss']:.4f} -> "
          f"{learning['last_window_mean_loss']:.4f}"
          f"   ({'DECREASING - model learns' if improved else 'NOT DECREASING'})")
    print(f"  force MAE {learning['first_window_force_mae']:.4f} -> "
          f"{learning['last_window_force_mae']:.4f}")
    print(f"  compute   {compute['mean_step_s'] * 1e3:.1f} ms/step, "
          f"{compute['frames_per_s']:.0f} frames/s, "
          f"peak RSS {compute['peak_rss_mb']:.0f} MB "
          f"(+{compute['rss_growth_mb']:.0f} MB)")
    print(f"  {result['steps']} steps in {compute['wall_clock_s']:.1f} s "
          f"on {result['environment']['device']}")
    print("-" * 78)
    print("  Reminder: not official results, not comparable across bake-off models.")

    if args.report:
        report_path = Path(args.report)
        if "experiments" in report_path.parts:
            raise SystemExit(
                "refusing to write a smoke-run report under experiments/ - that "
                "directory holds official results only"
            )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  profile written to {report_path}")

    return 0 if improved else 1


if __name__ == "__main__":
    raise SystemExit(main())
