"""Provisional standalone PaiNN trainer for the Phase 1 bake-off (SCRUM-52).

    +---------------------------------------------------------------------+
    | PROVISIONAL. The shared loop in ml/training/train.py cannot train    |
    | PaiNN yet (docs/reviews/scrum-52-painn-review.md, H1-H3), so this    |
    | trains it now under the same loop, with every setting read from      |
    | experiments/configs/phase1_painn.yaml. If the team's final shared    |
    | settings differ from that config, PaiNN is re-trained through        |
    | train.py and this checkpoint is discarded. Delete this file once     |
    | train.py supports PaiNN.                                             |
    +---------------------------------------------------------------------+

Identical to train.py: the per-epoch loop *is* train.py's `run_epoch`,
imported rather than copied so the two cannot drift - energy/force-weighted
MSE, gradient clipping at 10, eval-mode val pass. Around it, as in
train.py's `main`: Adam (weight_decay 1e-6), ReduceLROnPlateau (factor 0.5,
patience 5), early stopping on val loss, a checkpoint on every improvement,
torch seed from `data.seed`.

Deliberately different from train.py:

- **Energy.** The model owns the shift. PaiNN's per-atom `energy_shift` /
  `energy_scale` buffers are set from the training subset and the loss is
  on absolute energy (`run_epoch` is called with `energy_shift=0.0`).
  train.py subtracts a mean outside the model, which evaluate.py never adds
  back (review H3); doing both would apply the shift twice.
- **Training budget.** `train.train_subset` frames from `train_idx`, see
  `stride_subset`. Val is used whole.
- **Checkpoint format.** `registry.save_checkpoint`, the shape evaluate.py
  reads (review H2), written atomically.
- **The test split is never constructed.** Score once afterwards with
  ml/training/evaluate.py.

Run (from the repo root):

    python scripts/train_painn.py --smoke     # 2 epochs x 3 batches, not a result
    python scripts/train_painn.py             # full run
    python -m ml.training.evaluate --checkpoint experiments/checkpoints/phase1_painn.pt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ml.data.datasets import MD17Dataset  # noqa: E402
from ml.models.registry import build_model, save_checkpoint  # noqa: E402
from ml.training.train import run_epoch  # noqa: E402

DEFAULT_CONFIG = ROOT_DIR / "experiments" / "configs" / "phase1_painn.yaml"

# train.py hardcodes these (train.py:169-170, and the clip at :98 inside
# run_epoch) rather than reading them from the config. Mirrored here so this
# run matches the shared loop, and recorded in the checkpoint. GRAD_CLIP is
# a record only - run_epoch applies its own.
WEIGHT_DECAY = 1e-6
LR_FACTOR = 0.5
LR_PATIENCE = 5
GRAD_CLIP = 10.0

# The same budget as train.py --smoke-test.
SMOKE_EPOCHS = 2
SMOKE_PATIENCE = 2
SMOKE_BATCHES = 3

HISTORY_FIELDS = [
    "epoch", "lr",
    "train_loss", "train_energy_mse", "train_force_mse",
    "val_loss", "val_energy_mse", "val_force_mse",
    "train_s", "val_s", "epoch_s", "elapsed_s", "improved",
]


def open_split(gold_path: Path, split: str, data_config: dict[str, Any]) -> MD17Dataset:
    """The only place a split is opened. Test is refused: evaluate.py scores it, once."""
    if split not in ("train", "val"):
        raise SystemExit(
            f"refusing to open the {split!r} split. Training reads train and val "
            "only; the test split is scored exactly once, afterwards, by "
            "ml/training/evaluate.py."
        )
    return MD17Dataset(
        gold_path,
        split=split,
        cutoff_radius=float(data_config["cutoff_radius"]),
        num_rbf=int(data_config["num_rbf"]),
    )


def stride_subset(n_train: int, size: Optional[int], seed: int) -> tuple[np.ndarray, int, int]:
    """Positions within `train_idx` for the training subset: a fixed stride, seeded offset.

    The config calls the subset "a deterministic stride over train_idx at
    seed 42". A stride alone needs no seed, so the seed picks the starting
    offset within the first stride window (systematic sampling):

        stride    = n_train // size
        offset    = default_rng(seed).integers(stride)
        positions = offset + stride * arange(size)

    The frames are spread evenly over the whole train block rather than
    taken from its head, and never leave it: the largest position is below
    `stride * size <= n_train`. The positions are recorded in the checkpoint
    so the other bake-off models can reuse exactly these frames.

    Returns (positions, stride, offset). `size=None` means the whole split.
    """
    if size is None:
        return np.arange(n_train), 1, 0
    if not 0 < size <= n_train:
        raise ValueError(f"train_subset must be in 1..{n_train}, got {size}")
    stride = n_train // size
    offset = int(np.random.default_rng(seed).integers(stride))
    return offset + stride * np.arange(size), stride, offset


def git_state() -> dict[str, Any]:
    def git(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", *args], cwd=ROOT_DIR, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = git("status", "--porcelain")
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status)}


def save_atomic(path: Path, model: torch.nn.Module, phase: str, data: dict, train: dict) -> None:
    """save_checkpoint to a temp file, then rename it over the old one.

    A run killed mid-write therefore still leaves the previous best
    checkpoint intact, instead of a truncated file in its place.
    """
    tmp = path.with_name(path.name + ".tmp")
    save_checkpoint(tmp, model, phase, data=data, train=train)
    os.replace(tmp, path)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)


def train(config: dict[str, Any], config_path: Path, smoke: bool, overwrite: bool) -> int:
    data_cfg, model_cfg, train_cfg = config["data"], config["model"], config["train"]

    phase = model_cfg["phase"]
    if phase != "painn":
        raise SystemExit(f"this script trains PaiNN only; {config_path.name} has model.phase={phase!r}")
    if float(data_cfg["cutoff_radius"]) != float(model_cfg["cutoff_radius"]):
        raise SystemExit(
            f"data.cutoff_radius={data_cfg['cutoff_radius']} != model.cutoff_radius="
            f"{model_cfg['cutoff_radius']}; the config requires them equal"
        )
    device = torch.device(train_cfg["device"])
    if device.type != "cpu":
        raise SystemExit(
            f"train.device={train_cfg['device']!r}: train.py's run_epoch never moves "
            "batches off the CPU, so this script runs on cpu only"
        )

    checkpoint_path = ROOT_DIR / config["output"]["checkpoint"]
    if smoke:
        # Never at the real path, so a smoke checkpoint cannot be scored as a result.
        checkpoint_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_smoke.pt")
    history_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_history.csv")
    if checkpoint_path.exists() and not (smoke or overwrite):
        raise SystemExit(f"{_repo_relative(checkpoint_path)} already exists; pass --overwrite to replace it")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    gold_path = ROOT_DIR / data_cfg["gold_path"]
    if not gold_path.exists():
        raise SystemExit(
            f"no gold data at {gold_path}. Run `python -m ml.data.preprocessing` "
            "first - do not invent stand-in data (CLAUDE.md)."
        )

    git = git_state()
    if git["git_dirty"]:
        print("WARNING: the working tree has uncommitted changes; the recorded "
              "git commit does not fully describe this run.", flush=True)

    torch.manual_seed(int(data_cfg["seed"]))
    setup_start = time.perf_counter()

    train_ds = open_split(gold_path, "train", data_cfg)
    val_ds = open_split(gold_path, "val", data_cfg)
    for field in ("molecule", "theory"):
        if str(getattr(train_ds, field)) != str(data_cfg[field]):
            raise SystemExit(
                f"config says {field}={data_cfg[field]!r} but {gold_path.name} holds "
                f"{getattr(train_ds, field)!r}"
            )

    subset_size = train_cfg.get("train_subset")
    subset_seed = int(train_cfg["train_subset_seed"])
    positions, stride, offset = stride_subset(
        len(train_ds), None if subset_size is None else int(subset_size), subset_seed
    )
    with np.load(gold_path, allow_pickle=True) as gold:
        gold_indices = gold["train_idx"][positions]

    batch_size = int(train_cfg["batch_size"])
    train_loader = DataLoader(
        torch.utils.data.Subset(train_ds, positions.tolist()), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Per-atom statistics from the training subset only. float64: the mean of
    # ~-97,000 kcal/mol over 10,000 frames loses digits in float32.
    energies = train_ds.E[torch.as_tensor(positions)].double()
    n_atoms = int(train_ds.z.shape[0])
    energy_mean, energy_std = float(energies.mean()), float(energies.std())
    shift, scale = energy_mean / n_atoms, energy_std / n_atoms

    model = build_model(
        phase,
        hidden_channels=int(model_cfg["hidden_channels"]),
        num_layers=int(model_cfg["num_layers"]),
        num_rbf=int(model_cfg["num_rbf"]),
        cutoff_radius=float(model_cfg["cutoff_radius"]),
        max_atomic_number=int(model_cfg["max_atomic_number"]),
        eps=float(model_cfg["eps"]),
    ).to(device)
    model.set_energy_statistics(shift=shift, scale=scale)

    lr = float(train_cfg["lr"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE
    )
    energy_w = float(train_cfg["energy_loss_weight"])
    force_w = float(train_cfg["force_loss_weight"])

    epochs = SMOKE_EPOCHS if smoke else int(train_cfg["epochs"])
    patience = SMOKE_PATIENCE if smoke else int(train_cfg["patience"])
    max_batches = SMOKE_BATCHES if smoke else None
    setup_s = time.perf_counter() - setup_start

    data_record = {
        "gold_path": data_cfg["gold_path"],
        "molecule": data_cfg["molecule"],
        "theory": data_cfg["theory"],
        "cutoff_radius": float(data_cfg["cutoff_radius"]),
        "num_rbf": int(data_cfg["num_rbf"]),
    }
    static_record = {
        "script": "scripts/train_painn.py",
        "provisional": True,
        "smoke": smoke,
        **git,
        "config_path": _repo_relative(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "config": config,
        "seed": int(data_cfg["seed"]),
        "train_subset": {
            "size": int(len(positions)),
            "seed": subset_seed,
            "stride": stride,
            "offset": offset,
            # Positions into the gold file's train_idx (what the loader indexes),
            # and the gold frame indices they map to.
            "positions": positions.tolist(),
            "gold_indices": gold_indices.tolist(),
        },
        "n_train_frames": int(len(positions)),
        "n_val_frames": len(val_ds),
        "energy_statistics": {
            "computed_on": "train subset",
            "energy_mean": energy_mean,
            "energy_std": energy_std,
            "n_atoms": n_atoms,
            "shift_per_atom": shift,
            "scale_per_atom": scale,
            "unit": "kcal/mol",
        },
        "optimizer": {"name": "Adam", "lr": lr, "weight_decay": WEIGHT_DECAY},
        "scheduler": {"name": "ReduceLROnPlateau", "factor": LR_FACTOR, "patience": LR_PATIENCE},
        "grad_clip_norm": GRAD_CLIP,
        "loss": {"name": "mse", "energy_weight": energy_w, "force_weight": force_w},
        "batch_size": batch_size,
        "max_epochs": epochs,
        "patience": patience,
        "max_batches_per_epoch": max_batches,
        "history_csv": _repo_relative(history_path),
        "setup_s": setup_s,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }

    print(f"{'SMOKE RUN - not a result. ' if smoke else ''}PaiNN on "
          f"{data_cfg['molecule']}/{data_cfg['theory']}, git {git['git_commit']}", flush=True)
    print(f"train: {len(positions)} of {len(train_ds)} frames (stride {stride}, offset {offset}, "
          f"seed {subset_seed}); val: {len(val_ds)} frames, whole; test: not opened", flush=True)
    print(f"energy: mean {energy_mean:.3f}, std {energy_std:.3f} kcal/mol -> per-atom "
          f"shift {shift:.4f}, scale {scale:.4f}", flush=True)
    print(f"{'epoch':>5} {'train_loss':>12} {'train_E':>10} {'train_F':>10} {'val_loss':>12} "
          f"{'val_E':>10} {'val_F':>10} {'lr':>10} {'epoch_s':>8} {'elapsed_s':>10}", flush=True)

    best_val_loss = float("inf")
    best_epoch: Optional[int] = None
    best_state: Optional[dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0
    epochs_done = 0
    stop_reason = "max_epochs"
    last_row: dict[str, Any] = {}

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        history = csv.DictWriter(history_file, fieldnames=HISTORY_FIELDS)
        history.writeheader()
        history_file.flush()

        train_start = time.perf_counter()
        try:
            for epoch in range(1, epochs + 1):
                epoch_lr = optimizer.param_groups[0]["lr"]
                epoch_start = time.perf_counter()
                train_loss, train_e, train_f = run_epoch(
                    model, train_loader, 0.0, energy_w, force_w,
                    optimizer=optimizer, max_batches=max_batches,
                )
                train_s = time.perf_counter() - epoch_start
                val_loss, val_e, val_f = run_epoch(
                    model, val_loader, 0.0, energy_w, force_w,
                    optimizer=None, max_batches=max_batches,
                )
                epoch_s = time.perf_counter() - epoch_start
                scheduler.step(val_loss)
                elapsed = time.perf_counter() - train_start
                epochs_done = epoch

                improved = val_loss < best_val_loss
                print(f"{epoch:>5} {train_loss:>12.4f} {train_e:>10.4f} {train_f:>10.4f} "
                      f"{val_loss:>12.4f} {val_e:>10.4f} {val_f:>10.4f} {epoch_lr:>10.2e} "
                      f"{epoch_s:>8.1f} {elapsed:>10.1f}", flush=True)
                if improved:
                    best_val_loss, best_epoch = val_loss, epoch
                    epochs_without_improvement = 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    # On every improvement, so an interrupted run still leaves a
                    # usable checkpoint (marked complete=False).
                    save_atomic(checkpoint_path, model, phase, data_record, {
                        **static_record,
                        "complete": False,
                        "stop_reason": None,
                        "epochs_run": epoch,
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val_loss,
                        "wall_clock_s": elapsed,
                    })
                    print(f"  -> checkpoint updated (val_loss {best_val_loss:.4f}) at "
                          f"{_repo_relative(checkpoint_path)}", flush=True)
                else:
                    epochs_without_improvement += 1

                last_row = {
                    "epoch": epoch, "lr": epoch_lr,
                    "train_loss": train_loss, "train_energy_mse": train_e, "train_force_mse": train_f,
                    "val_loss": val_loss, "val_energy_mse": val_e, "val_force_mse": val_f,
                    "train_s": train_s, "val_s": epoch_s - train_s, "epoch_s": epoch_s,
                    "elapsed_s": elapsed, "improved": int(improved),
                }
                history.writerow(last_row)
                history_file.flush()

                if epochs_without_improvement >= patience:
                    stop_reason = "early_stopping"
                    print(f"Early stopping at epoch {epoch} (no val improvement for "
                          f"{patience} epochs)", flush=True)
                    break
        except KeyboardInterrupt:
            stop_reason = "interrupted"
            print(f"\nInterrupted during epoch {epochs_done + 1}; keeping the best "
                  "checkpoint so far.", flush=True)

        wall_clock_s = time.perf_counter() - train_start

    if best_state is None:
        print("No epoch completed, so no checkpoint was written.", flush=True)
        return 130 if stop_reason == "interrupted" else 1

    model.load_state_dict(best_state)
    save_atomic(checkpoint_path, model, phase, data_record, {
        **static_record,
        "complete": stop_reason != "interrupted",
        "stop_reason": stop_reason,
        "epochs_run": epochs_done,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "wall_clock_s": wall_clock_s,
    })
    print(f"Best epoch {best_epoch} (val_loss {best_val_loss:.4f}); {epochs_done} epochs "
          f"in {wall_clock_s:.1f} s, stopped by {stop_reason}.")
    print(f"Checkpoint: {_repo_relative(checkpoint_path)}")
    print(f"History:    {_repo_relative(history_path)}")

    if smoke and last_row:
        # Per-batch cost from the last (warm) smoke epoch, scaled to a full epoch.
        train_batch_s = last_row["train_s"] / min(max_batches, len(train_loader))
        val_batch_s = last_row["val_s"] / min(max_batches, len(val_loader))
        full_epoch_s = train_batch_s * len(train_loader) + val_batch_s * len(val_loader)
        full_epochs = int(train_cfg["epochs"])
        print(f"Projection: {train_batch_s:.3f} s/train batch x {len(train_loader)} + "
              f"{val_batch_s:.3f} s/val batch x {len(val_loader)} = {full_epoch_s / 60:.1f} "
              f"min/epoch; {full_epochs} epochs = {full_epoch_s * full_epochs / 3600:.1f} h "
              f"(upper bound - early stopping can end it sooner)")
        print("SMOKE RUN - not a result.")

    return 130 if stop_reason == "interrupted" else 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--smoke", action="store_true",
        help=f"{SMOKE_EPOCHS} epochs x {SMOKE_BATCHES} batches, written to "
             "<checkpoint>_smoke.pt - checks nothing is broken, not a result",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing full-run checkpoint"
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return train(config, config_path, smoke=args.smoke, overwrite=args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
