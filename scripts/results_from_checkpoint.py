"""Produce the results JSON from an already-saved checkpoint.

`ml/training/train_physnet.py` writes its results only after the training
loop finishes, so a run stopped with Ctrl+C leaves a perfectly good
checkpoint and no numbers. This scores that checkpoint on its own, using
the same MetricAccumulator, so an interrupted run still yields a complete
deliverable.

    python -m scripts.results_from_checkpoint --config experiments/configs/phase1_physnet.yaml

Use it when a run has to be cut short. A run that finished normally has
already written this file and does not need it.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from ml.models.physnet import PhysNet
from ml.training.train_physnet import ROOT, evaluate, load_config, make_loaders


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--limit-val", type=int, default=2000,
                   help="validation configs to score (keep it matched to the training run)")
    p.add_argument("--train-seconds", type=float, default=None,
                   help="wall-clock training time to record, if known")
    p.add_argument("--epochs-run", type=int, default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device("cpu")

    _train_loader, val_loader = make_loaders(cfg, limit_train=1, limit_val=args.limit_val)

    ckpt_path = ROOT / "experiments" / "checkpoints" / f"{cfg['name']}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}; nothing to score")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Rebuild from the checkpoint's own model_config, never from the YAML:
    # the YAML may have been edited since the run, and a model rebuilt with
    # different widths silently loads nothing useful.
    model = PhysNet(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(model)

    t0 = time.time()
    metrics = evaluate(model, val_loader, cfg, device)
    infer_seconds = time.time() - t0
    n_val = sum(int(d.y.reshape(-1).shape[0]) for d in val_loader)

    results = {
        "name": cfg["name"],
        "model": "physnet",
        "note": ("PhysNet short-range term only; see docs/model_cards/physnet.md. "
                 "Scored from a saved checkpoint by scripts/results_from_checkpoint.py "
                 "- the training run was stopped before it wrote its own results."),
        "split_evaluated": "val",
        "metrics": metrics,
        "training_time_seconds": args.train_seconds,
        "inference_time_seconds": round(infer_seconds, 4),
        "inference_ms_per_config": round(1000 * infer_seconds / max(n_val, 1), 4),
        "n_parameters": sum(p_.numel() for p_ in model.parameters()),
        "epochs_run": args.epochs_run if args.epochs_run is not None else ckpt.get("train", {}).get("epoch"),
        "best_val_loss": ckpt.get("train", {}).get("val_loss"),
        "device": "cpu",
        "hardware": platform.processor() or platform.machine(),
        "torch_version": torch.__version__,
        "subset": {"val": args.limit_val},
        "config": cfg,
    }

    out = ROOT / "experiments" / "results" / f"{cfg['name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nresults -> {out}")

    m = metrics["per_key"][0]
    print(f"\nE MAE   {m['energy_mae_total']}")
    print(f"E RMSE  {m['energy_rmse_total']}")
    print(f"F MAE   {m['force_mae_component']}")
    print(f"F RMSE  {m['force_rmse_component']}")
    print(f"infer   {results['inference_ms_per_config']} ms/config")
    print(f"epochs  {results['epochs_run']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
