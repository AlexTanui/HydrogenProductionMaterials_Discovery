"""Standalone SchNet train + eval script for SCRUM-53 (Phase 1 bake-off).

Temporary stand-in for `ml/training/train.py` (Ruturaj) and
`ml/training/evaluate.py` (Fazin) - both still empty placeholders. Retire
this file once either lands: swap the config loading below for
`ExperimentConfig.from_yaml()` and the loops for whatever the shared
train/eval scripts turn out to be. Until then this is SchNet-only and
touches nothing anyone else owns.

Reads every hyperparameter from `experiments/configs/phase1_schnet.yaml`
(no ml.config.py yet, but nothing here is hardcoded either, per CLAUDE.md's
config-driven convention).

Run:
    python -m ml.training.train_schnet
    python -m ml.training.train_schnet --smoke-test   # a few batches, fast sanity check
    python -m ml.training.train_schnet --config path/to/other.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as Fnn
import yaml
from torch_geometric.loader import DataLoader

from ml.data.datasets import MD17Dataset
from ml.models.schnet import SchNet
from ml.utils.contract import ModelOutput, ReferenceData
from ml.utils.metrics import MetricAccumulator

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = ROOT_DIR / "experiments" / "configs" / "phase1_schnet.yaml"


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_datasets(cfg: dict) -> tuple[MD17Dataset, MD17Dataset, MD17Dataset]:
    gold_path = ROOT_DIR / cfg["data"]["gold_path"]
    kwargs = dict(cutoff_radius=cfg["data"]["cutoff_radius"], num_rbf=cfg["model"]["num_rbf"])
    return (
        MD17Dataset(gold_path, split="train", **kwargs),
        MD17Dataset(gold_path, split="val", **kwargs),
        MD17Dataset(gold_path, split="test", **kwargs),
    )


def batch_loss(model: SchNet, batch, energy_shift: float, energy_w: float, force_w: float):
    """Trains the model against *shifted* energy (see module docs on why),
    forces are unshifted (already ~zero-mean, no scale issue)."""
    pred_e, pred_f = model.predict_energy_and_forces(
        batch.x.view(-1), batch.pos, batch.edge_index, batch.edge_attr, batch=batch.batch,
    )
    target_e = batch.y.view(-1) - energy_shift
    energy_loss = Fnn.mse_loss(pred_e, target_e)
    force_loss = Fnn.mse_loss(pred_f, batch.force)
    loss = energy_w * energy_loss + force_w * force_loss
    return loss, float(energy_loss.detach()), float(force_loss.detach())


def run_epoch(model, loader, energy_shift, energy_w, force_w, optimizer=None, max_batches=None):
    """optimizer=None -> eval mode, no weight update (forces still need
    autograd internally - never wrap this in torch.no_grad())."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = total_e = total_f = 0.0
    n = 0
    for batch in loader:
        if is_train:
            optimizer.zero_grad()
        loss, e_loss, f_loss = batch_loss(model, batch, energy_shift, energy_w, force_w)
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
        total_loss += float(loss.detach())
        total_e += e_loss
        total_f += f_loss
        n += 1
        if max_batches is not None and n >= max_batches:
            break
    return total_loss / n, total_e / n, total_f / n


def evaluate_test_set(model: SchNet, loader, dataset: MD17Dataset, energy_shift: float, max_batches=None) -> dict:
    """Real, physical-unit metrics via Fazin's harness - energy_shift is
    added back here so this number is comparable to every other bake-off
    architecture, none of which need to know this model trains on a
    shifted target internally."""
    model.eval()
    acc = MetricAccumulator()
    n_frames = 0
    n = 0
    start = time.perf_counter()
    for batch in loader:
        pred_e, pred_f = model.predict_energy_and_forces(
            batch.x.view(-1), batch.pos, batch.edge_index, batch.edge_attr, batch=batch.batch,
        )
        pred = ModelOutput(E=(pred_e + energy_shift).detach(), F=pred_f.detach())
        ref = ReferenceData(
            E=batch.y.view(-1), F=batch.force, batch=batch.batch,
            molecule=dataset.molecule, theory=dataset.theory, z=batch.x.view(-1),
        )
        acc.update(pred, ref)
        n_frames += ref.n_frames
        n += 1
        if max_batches is not None and n >= max_batches:
            break
    elapsed = time.perf_counter() - start
    results = acc.compute()
    results["inference_time_s_total"] = elapsed
    results["inference_time_ms_per_frame"] = 1000 * elapsed / n_frames if n_frames else None
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke-test", action="store_true",
                         help="Tiny run (a few batches, a few epochs) to check nothing's broken - not a real result.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["data"]["seed"])

    device = torch.device(cfg["train"]["device"])
    train_ds, val_ds, test_ds = make_datasets(cfg)

    batch_size = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Energy centering: this dataset's total energies sit at ~-97196
    # kcal/mol with a std of only ~4.2 - the physically meaningful part is
    # a tiny wobble on a huge constant. Training the model to output that
    # wobble (target = E - shift) instead of the raw value converges far
    # faster than making the readout bias climb to -97196 on its own.
    # energy_shift is saved in the checkpoint and added back at eval time
    # (see evaluate_test_set) so reported metrics stay in real kcal/mol,
    # comparable to every other architecture in the bake-off.
    energy_shift = float(train_ds.E.mean())

    model = SchNet(
        hidden_channels=cfg["model"]["hidden_channels"],
        num_layers=cfg["model"]["num_layers"],
        num_rbf=cfg["model"]["num_rbf"],
        cutoff=cfg["data"]["cutoff_radius"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    energy_w = cfg["train"]["energy_loss_weight"]
    force_w = cfg["train"]["force_loss_weight"]

    epochs = 2 if args.smoke_test else cfg["train"]["epochs"]
    patience = 2 if args.smoke_test else cfg["train"]["patience"]
    max_batches = 3 if args.smoke_test else None

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    checkpoints_dir = ROOT_DIR / "experiments" / "checkpoints"
    results_dir = ROOT_DIR / "experiments" / "results"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / f"{cfg['name']}.pt"
    results_path = results_dir / f"{cfg['name']}.json"

    print(f"energy_shift = {energy_shift:.3f} kcal/mol", flush=True)
    print(f"{'epoch':>5} {'train_loss':>12} {'train_E':>10} {'train_F':>10} {'val_loss':>12} {'val_E':>10} {'val_F':>10} {'lr':>10} {'epoch_s':>8} {'elapsed_s':>10}",
          flush=True)

    train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_e, train_f = run_epoch(
            model, train_loader, energy_shift, energy_w, force_w, optimizer=optimizer, max_batches=max_batches,
        )
        val_loss, val_e, val_f = run_epoch(
            model, val_loader, energy_shift, energy_w, force_w, optimizer=None, max_batches=max_batches,
        )
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.perf_counter() - epoch_start
        elapsed = time.perf_counter() - train_start
        print(f"{epoch:>5} {train_loss:>12.4f} {train_e:>10.4f} {train_f:>10.4f} "
              f"{val_loss:>12.4f} {val_e:>10.4f} {val_f:>10.4f} {lr:>10.2e} {epoch_time:>8.1f} {elapsed:>10.1f}",
              flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # Saved on every improvement, not just at the end, so a killed/
            # interrupted run still leaves a usable, non-stale checkpoint.
            torch.save({
                "model_state_dict": best_state,
                "energy_shift": energy_shift,
                "config": cfg,
                "epoch": epoch,
                "val_loss": best_val_loss,
                "training_time_s_so_far": elapsed,
            }, checkpoint_path)
            print(f"  -> checkpoint updated (val_loss {best_val_loss:.4f}) at {checkpoint_path}", flush=True)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs)", flush=True)
                break

    training_time_s = time.perf_counter() - train_start

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({
        "model_state_dict": model.state_dict(),
        "energy_shift": energy_shift,
        "config": cfg,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    test_results = evaluate_test_set(model, test_loader, test_ds, energy_shift, max_batches=max_batches)
    test_results["training_time_s"] = training_time_s
    test_results["best_val_loss"] = best_val_loss
    test_results["config"] = cfg

    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"Results saved: {results_path}")
    print(json.dumps(test_results["overall"], indent=2))


if __name__ == "__main__":
    main()
