"""Shared training entrypoint for Phase 1/2/3 experiment configs (SCRUM-51).

Reads `ml.config.ExperimentConfig` from a YAML file (glossary.md §6) and
dispatches to the model for `config.model.phase`, so no phase choice is ever
hardcoded here - see CLAUDE.md's three-phase-plan note. "mpnn" (Phase 1,
`ml/models/mpnn.py`) and "blip" (Phase 2, `ml/models/blip.py`) are
implemented; "graph_stochastic" raises until `ml/models/graph_stochastic.py`
lands.

Note for "blip": this loop trains/evaluates it with one stochastic weight
sample per forward call, same as any deterministic model - it is a valid
point-estimate run, but it is *not* yet how Phase 2's actual uncertainty
deliverable gets produced. That needs multiple MC samples per input
(`BLIP.predict_distribution`), which this shared loop does not call. See
`ml/models/blip.py`'s module docstring for the full list of what's wired up
and what's still a documented gap (the KL regularisation term, MC-aware
evaluation, and the UQ metrics to score it with).

The train/eval loop itself (energy+force MSE, early stopping on val loss,
checkpoint-on-improvement, Fazin's `MetricAccumulator` for the held-out test
report) mirrors `ml/training/train_schnet.py`, written as a temporary
stand-in for this exact file during the Phase 1 bake-off - see that module's
docstring. `train_schnet.py` stays as-is (SchNet isn't wired into
`ml.config`'s phase dispatch), but this is now the real shared path.

Run:
    python -m ml.training.train experiments/configs/phase1_baseline.yaml
    python -m ml.training.train experiments/configs/phase1_baseline.yaml --smoke-test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from ml.config import ExperimentConfig
from ml.data.datasets import MD17Dataset
from ml.models import registry
from ml.models.scaling import ScaledEnergyModel
from ml.utils.contract import ModelOutput, ReferenceData
from ml.utils.metrics import MetricAccumulator

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def build_model(cfg: ExperimentConfig) -> torch.nn.Module:
    """The registered model for `model.phase`, built from the other `model:` keys."""
    return registry.build_model(cfg.model.phase, **cfg.model.kwargs)


def make_datasets(cfg: ExperimentConfig) -> tuple[MD17Dataset, MD17Dataset, MD17Dataset]:
    gold_path = ROOT_DIR / cfg.data.gold_path
    kwargs = dict(cutoff_radius=cfg.data.cutoff_radius, num_rbf=cfg.data.num_rbf)
    return (
        MD17Dataset(gold_path, split="train", **kwargs),
        MD17Dataset(gold_path, split="val", **kwargs),
        MD17Dataset(gold_path, split="test", **kwargs),
    )


def batch_loss(model: torch.nn.Module, batch, energy_w: float, force_w: float):
    """MSE on absolute energy (kcal/mol) and forces. The model is a
    ScaledEnergyModel, which carries the energy shift itself."""
    pred_e, pred_f = model.predict_energy_and_forces(
        batch.x.view(-1), batch.pos, batch.edge_index, batch.edge_attr, batch=batch.batch,
    )
    target_e = batch.y.view(-1)
    energy_loss = F.mse_loss(pred_e, target_e)
    force_loss = F.mse_loss(pred_f, batch.force)
    loss = energy_w * energy_loss + force_w * force_loss
    return loss, float(energy_loss.detach()), float(force_loss.detach())


def run_epoch(model, loader, energy_w, force_w, optimizer=None, max_batches=None):
    """optimizer=None -> eval mode, no weight update (forces still need
    autograd internally - never wrap this in torch.no_grad())."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = total_e = total_f = 0.0
    n = 0
    for batch in loader:
        if is_train:
            optimizer.zero_grad()
        loss, e_loss, f_loss = batch_loss(model, batch, energy_w, force_w)
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


def evaluate_test_set(model, loader, dataset: MD17Dataset, max_batches=None) -> dict:
    """Real, physical-unit metrics via Fazin's harness. The wrapped model
    already predicts absolute energy, so nothing is added back."""
    model.eval()
    acc = MetricAccumulator()
    n_frames = 0
    n = 0
    start = time.perf_counter()
    for batch in loader:
        pred_e, pred_f = model.predict_energy_and_forces(
            batch.x.view(-1), batch.pos, batch.edge_index, batch.edge_attr, batch=batch.batch,
        )
        pred = ModelOutput(E=pred_e.detach(), F=pred_f.detach())
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
    parser.add_argument("config", help="Path to an experiments/configs/*.yaml file")
    parser.add_argument("--smoke-test", action="store_true",
                         help="Tiny run (a few batches, a few epochs) to check nothing's broken - not a real result.")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    torch.manual_seed(cfg.data.seed)

    device = torch.device(cfg.train.device)
    train_ds, val_ds, test_ds = make_datasets(cfg)

    batch_size = cfg.train.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # One energy scale-and-shift for every model (ml/models/scaling.py):
    # per-atom statistics from the training data only, held in the model's
    # own buffers, so the loss is on absolute energy and the checkpoint
    # carries the shift - nothing is subtracted here or added back later.
    n_atoms = int(train_ds.z.shape[0])
    shift, scale = ScaledEnergyModel.statistics_from_energies(train_ds.E, n_atoms)
    model = ScaledEnergyModel(build_model(cfg), shift=shift, scale=scale).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    energy_w = cfg.train.energy_loss_weight
    force_w = cfg.train.force_loss_weight

    epochs = 2 if args.smoke_test else cfg.train.epochs
    patience = 2 if args.smoke_test else cfg.train.patience
    max_batches = 3 if args.smoke_test else None

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    checkpoint_path = cfg.checkpoint_path(ROOT_DIR / "experiments" / "checkpoints")
    results_path = cfg.results_path(ROOT_DIR / "experiments" / "results")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    cfg_dict = cfg.to_dict()

    print(f"energy per atom: shift {shift:.4f}, scale {scale:.4f} kcal/mol", flush=True)
    print(f"{'epoch':>5} {'train_loss':>12} {'train_E':>10} {'train_F':>10} {'val_loss':>12} {'val_E':>10} {'val_F':>10} {'lr':>10} {'epoch_s':>8} {'elapsed_s':>10}",
          flush=True)

    train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_e, train_f = run_epoch(
            model, train_loader, energy_w, force_w, optimizer=optimizer, max_batches=max_batches,
        )
        val_loss, val_e, val_f = run_epoch(
            model, val_loader, energy_w, force_w, optimizer=None, max_batches=max_batches,
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
                "energy_scaling": model.energy_scaling(),
                "config": cfg_dict,
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
        "energy_scaling": model.energy_scaling(),
        "config": cfg_dict,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    test_results = evaluate_test_set(model, test_loader, test_ds, max_batches=max_batches)
    test_results["training_time_s"] = training_time_s
    test_results["best_val_loss"] = best_val_loss
    test_results["config"] = cfg_dict

    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"Results saved: {results_path}")
    print(json.dumps(test_results["overall"], indent=2))


if __name__ == "__main__":
    main()
