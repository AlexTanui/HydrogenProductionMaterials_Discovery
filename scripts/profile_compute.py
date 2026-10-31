"""Compute profile for SCRUM-49: training and inference cost per phase, and
how inference time and memory scale with the Monte Carlo sample count.

Measures, on the machine it runs on, through the harness's own pieces
(registry model -> ScaledEnergyModel, train.run_epoch for training steps,
predict_energy_and_forces / predict_distribution for inference):

- a training step, per model config, at the protocol's batch size, and the
  epoch time that implies for the protocol's 10,000 train / 5,000 val frames;
- one deterministic inference pass (Phase 1);
- `predict_distribution` at each MC sample count in `--mc`, for every
  stochastic config, with peak memory.

Weights are untrained: cost doesn't depend on their values. Frames come from
the real val split (never test). Each config runs in its own process so that
peak memory is per config; within a process MC counts run in ascending order,
so a CPU peak (which can't be reset) reads as the peak up to that count.

    python scripts/profile_compute.py --device cpu --threads 4 24 --out reports/compute/NAME.json
    python scripts/profile_compute.py --device cuda --out reports/compute/NAME.json
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.config import ExperimentConfig  # noqa: E402
from ml.data.datasets import MD17Dataset  # noqa: E402
from ml.models.scaling import ScaledEnergyModel  # noqa: E402
from ml.training.train import build_model, git_state, run_epoch  # noqa: E402

CONFIGS = ROOT / "experiments" / "configs"
TRAIN_CONFIGS = ("phase1_baseline", "phase2_blip", "phase2_blip_nll", "phase3_graph_stochastic")
MC_CONFIGS = ("phase2_blip", "phase3_graph_stochastic")
DEFAULT_MC = (1, 2, 5, 10, 20, 50, 100)
N_FRAMES = 256   # val frames per timed pass: 8 batches of 32


def _peak_rss_mb() -> float:
    import psutil
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)  # Windows
    if peak is None:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak *= 1 if sys.platform == "darwin" else 1024  # bytes on macOS, KiB on Linux
    return peak / 2**20


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _model_and_loader(name: str, device: torch.device):
    cfg = ExperimentConfig.from_yaml(CONFIGS / f"{name}.yaml")
    torch.manual_seed(0)
    inner = build_model(cfg)
    dataset = MD17Dataset(ROOT / cfg.data.gold_path, split="val",
                          cutoff_radius=cfg.data.cutoff_radius, num_rbf=cfg.data.num_rbf)
    positions = np.linspace(0, len(dataset), N_FRAMES, endpoint=False).astype(int).tolist()
    subset = torch.utils.data.Subset(dataset, positions)
    shift, scale = ScaledEnergyModel.statistics_from_energies(dataset.E[positions], int(dataset.z.shape[0]))
    model = ScaledEnergyModel(inner, shift=shift, scale=scale).to(device)
    batches = [b.to(device) for b in DataLoader(subset, batch_size=cfg.train.batch_size, shuffle=False)]
    return cfg, model, batches


def _profile_config(name: str, device_name: str, threads: int, mc: list[int]) -> dict[str, Any]:
    """Runs in its own process (see main)."""
    torch.set_num_threads(threads)
    device = torch.device(device_name)
    cfg, model, batches = _model_and_loader(name, device)
    n_frames = sum(int(b.num_graphs) for b in batches)
    result: dict[str, Any] = {"config": name, "device": device_name, "threads": threads,
                              "n_parameters": int(sum(p.numel() for p in model.parameters())),
                              "frames_timed": n_frames, "batch_size": cfg.train.batch_size,
                              "peak_rss_mb_after_setup": _peak_rss_mb()}

    # Inference first, so the CPU peak (which can't be reset) describes it
    # before training's larger graphs raise it.
    model.eval()
    if model.supports_distribution:
        result["mc"] = []
        for k in sorted(mc):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            model.predict_distribution(batches[0].x, batches[0].pos, batches[0].edge_index,
                                       batches[0].edge_attr, batches[0].batch, num_mc_samples=k)  # warm-up
            _sync(device)
            start = time.perf_counter()
            for batch in batches:
                model.predict_distribution(batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch,
                                           num_mc_samples=k)
            _sync(device)
            seconds = time.perf_counter() - start
            row = {"num_mc_samples": k, "ms_per_frame": seconds / n_frames * 1e3,
                   "ms_per_frame_per_sample": seconds / n_frames / k * 1e3,
                   "peak_rss_mb": _peak_rss_mb()}
            if device.type == "cuda":
                row["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
            result["mc"].append(row)
    else:
        for timed in (False, True):  # one warm-up pass, then the timed one
            start = time.perf_counter()
            for batch in batches:
                with torch.enable_grad():
                    model.predict_energy_and_forces(batch.x, batch.pos, batch.edge_index, batch.edge_attr,
                                                    batch.batch)
            _sync(device)
        result["single_pass_ms_per_frame"] = (time.perf_counter() - start) / n_frames * 1e3
        result["peak_rss_mb_after_inference"] = _peak_rss_mb()

    # Training step: the harness's own run_epoch over the timed batches (one
    # warm-up pass first). epoch=None puts BLIP-NLL straight on its NLL branch.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    w_e, w_f = cfg.train.energy_loss_weight, cfg.train.force_loss_weight
    run_epoch(model, batches[:2], w_e, w_f, device, optimizer=optimizer)
    _sync(device)
    start = time.perf_counter()
    run_epoch(model, batches, w_e, w_f, device, optimizer=optimizer)
    _sync(device)
    step_s = (time.perf_counter() - start) / len(batches)
    start = time.perf_counter()
    run_epoch(model, batches, w_e, w_f, device)
    _sync(device)
    val_batch_s = (time.perf_counter() - start) / len(batches)
    train_steps = math.ceil(cfg.train.train_subset / cfg.train.batch_size)
    val_steps = math.ceil(cfg.train.val_subset / cfg.train.batch_size)
    result["train_step_ms"] = step_s * 1e3
    result["val_batch_ms"] = val_batch_s * 1e3
    result["protocol_epoch_s"] = step_s * train_steps + val_batch_s * val_steps
    result["protocol_200_epochs_h"] = result["protocol_epoch_s"] * cfg.train.epochs / 3600
    result["peak_rss_mb_after_training"] = _peak_rss_mb()
    if device.type == "cuda":
        result["peak_cuda_allocated_mb_training"] = torch.cuda.max_memory_allocated() / 2**20
    return result


def _environment(device_name: str) -> dict[str, Any]:
    import psutil
    env = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "logical_cpus": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "python": platform.python_version(),
        "torch": torch.__version__,
        **git_state(ROOT),
    }
    if device_name == "cuda":
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        env["cuda"] = torch.version.cuda
    return env


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--threads", type=int, nargs="+", default=[torch.get_num_threads()])
    parser.add_argument("--mc", type=int, nargs="+", default=list(DEFAULT_MC))
    parser.add_argument("--configs", nargs="+", default=list(TRAIN_CONFIGS))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--one", nargs=2, metavar=("CONFIG", "THREADS"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.one:  # child process: one config, one thread count
        print(json.dumps(_profile_config(args.one[0], args.device, int(args.one[1]), args.mc)))
        return 0

    environment = _environment(args.device)  # before anything is written, so git_dirty is honest
    runs = []
    for threads in args.threads:
        for name in args.configs:
            out = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--device", args.device,
                 "--one", name, str(threads), "--mc", *map(str, args.mc)],
                cwd=ROOT, check=True, capture_output=True, text=True,
            ).stdout.strip().splitlines()[-1]
            run = json.loads(out)
            runs.append(run)
            line = (f"{args.device} {threads:>2} thr  {name:24} step {run['train_step_ms']:7.1f} ms  "
                    f"epoch {run['protocol_epoch_s']:6.1f} s  200 ep {run['protocol_200_epochs_h']:5.2f} h  "
                    f"RSS setup {run['peak_rss_mb_after_setup']:.0f} MB, after training {run['peak_rss_mb_after_training']:.0f} MB")
            if "single_pass_ms_per_frame" in run:
                line += f"  1 pass {run['single_pass_ms_per_frame']:.3f} ms/frame"
            print(line, flush=True)
            for row in run.get("mc", []):
                mem = f"  peak cuda {row['peak_cuda_allocated_mb']:7.1f} MB" if "peak_cuda_allocated_mb" in row else ""
                print(f"{'':36}K={row['num_mc_samples']:>3}: {row['ms_per_frame']:8.3f} ms/frame "
                      f"({row['ms_per_frame_per_sample']:.3f} per sample)  peak RSS {row['peak_rss_mb']:7.1f} MB{mem}",
                      flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"environment": environment, "runs": runs}, indent=2) + "\n",
                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
