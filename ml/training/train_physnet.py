"""Standalone trainer for the PhysNet bake-off entry (SCRUM-50).

**This is scaffolding, not the shared training loop.** `ml/training/train.py`
is Ruturaj's and is still empty; this exists so the PhysNet deliverable
isn't blocked on it. It depends only on things that have actually landed:
`ml/data/datasets.py`, `ml/models/physnet.py`, and Fazin's
`ml/utils/metrics.py` + `ml/utils/contract.py` (SCRUM-37). When `train.py`
lands, the comparison should be re-run through it so every model in the
bake-off is trained by identical code - and this file deleted rather than
left around as a second way to do the same thing.

    python -m ml.training.train_physnet --config experiments/configs/phase1_physnet.yaml

    # quick end-to-end check on a slice, before committing to a real run
    python -m ml.training.train_physnet --config experiments/configs/phase1_physnet.yaml \\
        --limit-train 2000 --limit-val 500 --epochs 2
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader

from ml.data.datasets import MD17Dataset
from ml.models.physnet import PhysNet
from ml.utils.contract import ModelOutput, ReferenceData
from ml.utils.metrics import MetricAccumulator

ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_loaders(cfg, limit_train=None, limit_val=None):
    gold = ROOT / cfg["data"]["gold_path"]
    if not gold.exists():
        raise FileNotFoundError(
            f"{gold} not found. Generate it first:\n"
            "  python -m ml.data.preprocessing --dataset md17"
        )
    kw = dict(cutoff_radius=cfg["data"]["cutoff_radius"],
              num_rbf=cfg["model"].get("num_rbf", 16))
    train = MD17Dataset(gold, split="train", **kw)
    val = MD17Dataset(gold, split="val", **kw)

    # Subsetting takes a contiguous prefix, never a random sample: the
    # splits are contiguous trajectory blocks precisely so that adjacent
    # near-identical frames stay on one side of the boundary, and a random
    # subset would undo that reasoning.
    if limit_train:
        train = torch.utils.data.Subset(train, range(min(limit_train, len(train))))
    if limit_val:
        val = torch.utils.data.Subset(val, range(min(limit_val, len(val))))

    bs = cfg["train"]["batch_size"]
    return (DataLoader(train, batch_size=bs, shuffle=True),
            DataLoader(val, batch_size=bs, shuffle=False))


def save_checkpoint_compat(path, model, cfg, epoch, val_loss):
    """Writes the checkpoint shape ml/models/registry.py defines.

    That contract - phase / model_config / state_dict / data / train - is
    what ml/training/evaluate.py reads, so a checkpoint in any other shape
    cannot be scored by the shared harness, which is the only thing that
    produces comparable bake-off numbers.

    `registry.py` lives on the `fazin` branch (SCRUM-52) and has not
    merged to main yet, so it is imported opportunistically: use
    `save_checkpoint` when present, and otherwise write the identical
    dict by hand. Once it merges, delete the fallback and always call it.
    """
    payload = {
        "phase": "physnet",
        "model_config": model.config(),
        "state_dict": model.state_dict(),
        "data": {
            "gold_path": cfg["data"]["gold_path"],
            "molecule": cfg["data"]["molecule"],
            "theory": cfg["data"]["theory"],
            "cutoff_radius": cfg["data"]["cutoff_radius"],
            "num_rbf": cfg["model"].get("num_rbf", 16),
        },
        "train": {"epoch": epoch, "val_loss": val_loss, "config_name": cfg["name"],
                  "trained_by": "ml/training/train_physnet.py (scaffolding, not train.py)"},
    }
    try:
        from ml.models.registry import MODEL_REGISTRY, save_checkpoint
        if "physnet" in MODEL_REGISTRY:
            return save_checkpoint(path, model, "physnet", payload["data"], payload["train"])
        print("NOTE: registry.py is present but PhysNet is not in MODEL_REGISTRY - "
              "add `\"physnet\": PhysNet` there so evaluate.py can score this checkpoint.")
    except ImportError:
        pass
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def batch_forward(model, data, device):
    z = data.x.reshape(-1).long().to(device)
    pos = data.pos.to(device)
    edge_index = data.edge_index.to(device)
    batch = data.batch.to(device)
    energy, forces = model.predict_energy_and_forces(z, pos, edge_index, None, batch)
    return energy, forces, z, batch


def run_epoch(model, loader, cfg, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    w_e = cfg["train"]["energy_loss_weight"]
    w_f = cfg["train"]["force_loss_weight"]
    total, n_batches = 0.0, 0

    for data in loader:
        energy, forces, _z, _b = batch_forward(model, data, device)
        target_e = data.y.reshape(-1).to(device)
        target_f = data.force.to(device)

        loss = w_e * (energy - target_e).abs().mean() + w_f * (forces - target_f).abs().mean()

        if training:
            optimizer.zero_grad()
            loss.backward()
            # MD17 energies are ~1e5 and forces are weighted 100x, so early
            # gradients can be enormous. Clipping keeps the first epochs
            # from diverging before the per-element shift has settled.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        total += float(loss.detach())
        n_batches += 1

    return total / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    """Scores a split with Fazin's accumulator, so the numbers in the
    results JSON are produced by the same code as every other model's."""
    model.eval()
    acc = MetricAccumulator()
    for data in loader:
        energy, forces, z, batch = batch_forward(model, data, device)
        acc.update(
            ModelOutput(E=energy.detach().cpu(), F=forces.detach().cpu()),
            ReferenceData(E=data.y.reshape(-1).cpu(), F=data.force.cpu(),
                          batch=data.batch.cpu(), molecule=cfg["data"]["molecule"],
                          theory=cfg["data"]["theory"], z=z.cpu()),
        )
    return acc.compute()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="overrides the config")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs

    torch.manual_seed(cfg["data"]["seed"])
    requested = cfg["train"].get("device", "cpu")
    device = torch.device(requested if (requested != "cuda" or torch.cuda.is_available()) else "cpu")
    if requested == "cuda" and device.type == "cpu":
        print("WARNING: config asks for cuda but no GPU is visible - falling back to cpu.")
        print("         Full-dataset training on cpu is not realistic; use --limit-train.")

    train_loader, val_loader = make_loaders(cfg, args.limit_train, args.limit_val)

    model = PhysNet(
        hidden_channels=cfg["model"]["hidden_channels"],
        num_modules=cfg["model"].get("num_modules", 3),
        num_layers=cfg["model"].get("num_layers", 2),
        num_rbf=cfg["model"].get("num_rbf", 16),
        cutoff=cfg["data"]["cutoff_radius"],
    ).to(device)

    # Anchor the output near the data's magnitude before step one, or the
    # optimiser spends its first epochs travelling from 0 to -1e5.
    first = next(iter(train_loader))
    model.init_shift_from_energies(
        first.x.reshape(-1)[: int((first.batch == 0).sum())],
        first.y.reshape(-1),
    )
    print(model)
    print(f"device={device}  train_batches={len(train_loader)}  val_batches={len(val_loader)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    ckpt_dir = ROOT / "experiments" / "checkpoints"
    res_dir = ROOT / "experiments" / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cfg['name']}.pt"

    best, bad_epochs, t0 = float("inf"), 0, time.time()
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, cfg, device, optimizer)
        val_loss = run_epoch(model, val_loader, cfg, device, None)
        scheduler.step(val_loss)

        marker = ""
        if val_loss < best:
            best, bad_epochs, marker = val_loss, 0, "  *best"
            save_checkpoint_compat(ckpt_path, model, cfg, epoch, val_loss)
        else:
            bad_epochs += 1
        print(f"epoch {epoch:>4}  train {train_loss:>12.4f}  val {val_loss:>12.4f}{marker}")

        if bad_epochs >= cfg["train"]["patience"]:
            print(f"early stop: no improvement for {bad_epochs} epochs")
            break
    train_seconds = time.time() - t0

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"])

    t1 = time.time()
    val_metrics = evaluate(model, val_loader, cfg, device)
    infer_seconds = time.time() - t1
    n_val = sum(int(d.y.reshape(-1).shape[0]) for d in val_loader)

    results = {
        "name": cfg["name"],
        "model": "physnet",
        "note": "PhysNet short-range term only; see docs/model_cards/physnet.md",
        "split_evaluated": "val",  # test stays untouched until final evaluation
        "metrics": val_metrics,
        "training_time_seconds": round(train_seconds, 2),
        "inference_time_seconds": round(infer_seconds, 4),
        "inference_ms_per_config": round(1000 * infer_seconds / max(n_val, 1), 4),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "epochs_run": epoch,
        "best_val_loss": best,
        "device": str(device),
        "hardware": platform.processor() or platform.machine(),
        "torch_version": torch.__version__,
        "subset": {"train": args.limit_train, "val": args.limit_val},
        "config": cfg,
    }
    res_path = res_dir / f"{cfg['name']}.json"
    res_path.write_text(json.dumps(results, indent=2))

    print(f"\ncheckpoint -> {ckpt_path}")
    print(f"results    -> {res_path}")
    print(json.dumps(val_metrics, indent=2)[:1200])
    if args.limit_train:
        print("\nNOTE: this was a SUBSET run - not a bake-off result. "
              "Re-run without --limit-train for numbers that count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
