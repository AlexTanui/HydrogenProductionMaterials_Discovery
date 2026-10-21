"""Shared training entrypoint: one locked protocol for every model (SCRUM-51).

Trains any model in ml/models/registry.py - the five Phase 1 bake-off
backbones (mpnn, schnet, painn, physnet, dimenet), Phase 2's blip, and
Phase 3 once it registers - from an experiment YAML (glossary.md section
6). `model.phase` picks the model, so no phase choice is hardcoded here
(CLAUDE.md's three-phase-plan note), and nothing below branches on it.

The protocol every model gets, so that a benchmark row differs from
another only in its `model:` block:

- **Data.** Gold only. `train.train_subset` frames from `train_idx` and
  `train.val_subset` frames from `val_idx`, each a fixed stride with a
  seeded offset (`stride_subset`), recorded in the run record. The test
  split is never constructed here.
- **Energy.** Every model is wrapped in `ScaledEnergyModel`
  (ml/models/scaling.py) with per-atom shift/scale from the training
  subset; the loss is on absolute energies in kcal/mol.
- **Loss and optimiser.** energy_w * MSE(E) + force_w * MSE(F); Adam with
  weight decay 1e-6; ReduceLROnPlateau(factor 0.5, patience 5) on val
  loss; gradient-norm clipping at 10; early stopping after
  `train.patience` epochs without val improvement. Val (the subset) is the
  only thing early stopping and checkpoint selection see.
- **Batch order.** The train loader shuffles with its own generator seeded
  from `data.seed`, so every model sees the same batches in the same
  order however much of the global RNG its initialisation - or, for BLIP,
  its forward pass - consumes.
- **Checkpoints.** `registry.save_checkpoint` on every val improvement,
  written atomically, so an interrupted run leaves a usable checkpoint.
- **Scoring.** Once, at the end, by `ml.training.evaluate.evaluate_checkpoint`
  on the test split. It is the only scoring path in the repo.

A real run writes:

    experiments/checkpoints/<name>.pt      checkpoint (gitignored)
    experiments/runs/<name>/               run record, tracked in git:
        run.json          resolved config, git state, energy statistics,
                          environment, parameter count, epochs, timing
        subsets.json      the train/val frames used, as gold indices
        history.csv       one row per epoch
        test_metrics.json evaluate_checkpoint's output
    experiments/results/<name>.json        same as test_metrics.json, for the backend

and refuses to overwrite an existing checkpoint or run record without
`--overwrite`.

BLIP is trained and scored here with one stochastic weight sample per
forward call. That is a valid point-estimate run, not Phase 2's
uncertainty deliverable, which needs `BLIP.predict_distribution`'s MC
samples - see ml/models/blip.py's module docstring.

Run:
    python -m ml.training.train experiments/configs/phase1_painn.yaml
    python -m ml.training.train experiments/configs/phase1_painn.yaml --smoke-test
    python -m ml.training.train experiments/configs/phase1_painn.yaml --epochs 2 --output-root DIR

`--smoke-test` (2 epochs x 3 batches) and `--epochs N` are development
runs: they write to a fresh temporary directory unless `--output-root` is
given, never score the test split, and are stamped `"reportable": false`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader

from ml.config import ExperimentConfig
from ml.data.datasets import MD17Dataset
from ml.models import registry
from ml.models.scaling import ScaledEnergyModel
from ml.training.evaluate import evaluate_checkpoint

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS_DIR = ROOT_DIR / "experiments"

# Fixed across every model; recorded in each run record. Changing one is a
# protocol change for the whole comparison, not a per-model tweak.
WEIGHT_DECAY = 1e-6
LR_FACTOR = 0.5
LR_PATIENCE = 5
GRAD_CLIP = 10.0

SMOKE_EPOCHS = 2
SMOKE_BATCHES = 3

HISTORY_FIELDS = [
    "epoch", "lr",
    "train_loss", "train_energy_mse", "train_force_mse", "train_kl",
    "val_loss", "val_energy_mse", "val_force_mse", "val_kl",
    "train_s", "val_s", "epoch_s", "elapsed_s", "improved",
]


# --------------------------------------------------------------------------
# Model and data
# --------------------------------------------------------------------------

def build_model(cfg: ExperimentConfig) -> torch.nn.Module:
    """The registered model for `model.phase`, built from the other `model:` keys."""
    return registry.build_model(cfg.model.phase, **cfg.model.kwargs)


def resolve_gold_path(cfg: ExperimentConfig) -> Path:
    path = Path(cfg.data.gold_path)
    return path if path.is_absolute() else ROOT_DIR / path


def open_split(cfg: ExperimentConfig, split: str) -> MD17Dataset:
    """The only place training opens a split. Test is refused: evaluate.py scores it, once."""
    if split not in ("train", "val"):
        raise ValueError(
            f"refusing to open the {split!r} split. Training reads train and val only; "
            "the test split is scored exactly once, afterwards, by ml/training/evaluate.py."
        )
    dataset = MD17Dataset(
        resolve_gold_path(cfg), split=split,
        cutoff_radius=cfg.data.cutoff_radius, num_rbf=cfg.data.num_rbf,
    )
    for field in ("molecule", "theory"):
        if str(getattr(dataset, field)) != str(getattr(cfg.data, field)):
            raise ValueError(
                f"config says data.{field}={getattr(cfg.data, field)!r} but "
                f"{resolve_gold_path(cfg).name} holds {getattr(dataset, field)!r}"
            )
    return dataset


def stride_subset(n: int, size: Optional[int], seed: Optional[int]) -> tuple[np.ndarray, int, int]:
    """Positions within a split for a subset: a fixed stride with a seeded offset.

        stride    = n // size
        offset    = default_rng(seed).integers(stride)
        positions = offset + stride * arange(size)

    The frames are spread evenly over the whole block rather than taken
    from its head, and never leave it: the largest position is below
    `stride * size <= n`. A stride alone needs no seed; the seed picks the
    starting offset within the first stride window (systematic sampling).
    Same algorithm as scripts/train_painn.py, which introduced it.

    Returns (positions, stride, offset). `size=None` means the whole split.
    """
    if size is None:
        return np.arange(n), 1, 0
    if not 0 < size <= n:
        raise ValueError(f"subset size must be in 1..{n}, got {size}")
    stride = n // size
    offset = int(np.random.default_rng(seed).integers(stride))
    return offset + stride * np.arange(size), stride, offset


def select_subsets(cfg: ExperimentConfig) -> dict[str, dict[str, Any]]:
    """The train and val frames this config trains and selects on.

    `positions` index into the split (what the loader uses); `gold_indices`
    are the frame numbers in the gold file they map to. Val reuses
    `train_subset_seed` - there is no separate val seed.
    """
    seed = cfg.train.train_subset_seed
    with np.load(resolve_gold_path(cfg), allow_pickle=False) as gold:
        split_idx = {split: gold[f"{split}_idx"] for split in ("train", "val")}
    selected = {}
    for split, size in (("train", cfg.train.train_subset), ("val", cfg.train.val_subset)):
        positions, stride, offset = stride_subset(len(split_idx[split]), size, seed)
        selected[split] = {
            "size": int(len(positions)),
            "of": int(len(split_idx[split])),
            "seed": seed,
            "stride": stride,
            "offset": offset,
            "positions": positions,
            "gold_indices": split_idx[split][positions],
        }
    return selected


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

# Absolute-last-resort variance floor - not the real fix, see
# `_gaussian_nll_loss` below. 1e-6, not uq_metrics.py's 1e-12: that module
# scores a trained checkpoint in float64; this runs every training step
# in float32, where 1e-12 is close enough to the precision limit that
# log(variance) and 1/variance can both go unstable.
_NLL_VAR_EPS = 1e-6


def _gaussian_nll_loss(mean: torch.Tensor, var: torch.Tensor, target: torch.Tensor,
                        min_var: float = _NLL_VAR_EPS) -> torch.Tensor:
    """Mean of -log N(target; mean, var) = 0.5*log(2*pi*var) + (target-mean)^2 / (2*var).

    Unlike MSE, this has a gradient through `var` as well as `mean`: it
    rewards a larger predicted variance exactly where the squared error
    is large, and penalises a large variance where the error is small.
    That is the one piece plain MSE can never provide, no matter how
    `kl_weight` is tuned - see ml/models/blip.py's module docstring.

    `min_var` is ADDED to `var`, never `clamp_min`-ed in: `var.clamp_min(x)`
    has zero gradient everywhere `var < x`, and right after BLIP's warmup
    the predicted variance is tiny almost everywhere (Fazin's review of
    SCRUM-61 - noise_gate has no KL term and plain MSE actively shrinks it
    during warmup) - a clamp there would silence the one gradient this
    loss exists to provide, freezing noise_gate exactly when it most
    needs to move. Addition keeps that gradient flowing at every scale of
    `var` while still guaranteeing the denominator below is never smaller
    than `min_var`. Train calls this with `min_var` set from the model's
    own warmup-era validation error (`batch_loss`/`train()`); a bare call
    (e.g. a unit test) falls back to the numerical-safety default above,
    which does not protect against the real-world blowup - the caller
    choosing a real `min_var` is what does that.
    """
    var = var + min_var
    return (0.5 * torch.log(2 * math.pi * var) + (target - mean) ** 2 / (2 * var)).mean()


def batch_loss(model: torch.nn.Module, batch, energy_w: float, force_w: float,
                n_batches: int = 1, epoch: Optional[int] = None):
    """Energy/force loss, weighted, plus BLIP's KL complexity cost when the
    wrapped model defines one.

    Two loss forms, chosen per-model by a generic hook (no model-name
    branch, same spirit as the `kl_weight` hook below):

    - **Default (every model, and BLIP at its old `train_num_mc_samples=1`
      default):** MSE on one stochastic sample, exactly as before.
    - **BLIP with `train_num_mc_samples > 1`, past its warmup:** Gaussian
      NLL (`_gaussian_nll_loss` above) on the K-sample mean/variance from
      `predict_training_distribution`. Costs roughly K forward+backward
      passes instead of one.

    **Why a warmup, why `min_var` matters more than the warmup itself,
    and why `epoch` (not a step count):** NLL training from BLIP's actual
    state after plain-MSE warmup diverges on its own. Per Fazin's review
    of SCRUM-61: `StochasticLinear.noise_gate` carries no KL term at all
    (only `weight_mu`/`weight_rho` do), so a single-sample MSE warmup
    actively shrinks it - any weight-noise is just unwanted variance from
    MSE's point of view, and nothing resists that. By the time warmup
    ends, the K-sample variance is often tiny while the model's squared
    error is not; `error^2 / (2*var)` was observed to reach ~1e8-3e8 in
    both a smoke test and a hand-checked estimate from real training data
    before `min_var` (below) existed. `nll_warmup_epochs` (a `model:` key,
    read via `getattr(inner, "nll_warmup_epochs", 0)` - same hook pattern
    as `kl_weight`) buys the mean prediction time to stop being
    catastrophically wrong before NLL turns on, but a warmup alone does
    not fix the collapsed-variance side: nothing during MSE-only warmup
    ever grows it back up. That is `train()`'s job, once per run, right
    at the boundary: measure that epoch's real validation error and use
    it to set `inner.nll_min_energy_var`/`nll_min_force_var`, the `min_var`
    `_gaussian_nll_loss` adds in (see that function for why addition, not
    a clamp). `epoch` is None when a caller (e.g. a unit test) wants the
    NLL branch immediately regardless of warmup; `train()` always passes
    the real epoch number.

    `n_batches` is the loader's length for the epoch being run (not
    `max_batches`'s dev-mode truncation): the minibatch-ELBO recipe
    distributes one full-dataset KL term evenly across every batch in an
    epoch (kl_weight * kl_divergence() / n_batches), so a shortened
    smoke-test epoch should not change that scaling. A model with no
    `kl_weight` (every Phase 1 model, and BLIP before this is set >0)
    gets `getattr(..., 0.0)` and the branch is skipped - zero added cost
    for models this was never meant to touch.
    """
    inner = getattr(model, "model", model)
    train_k = int(getattr(inner, "train_num_mc_samples", 1))
    warmup = int(getattr(inner, "nll_warmup_epochs", 0))
    past_warmup = epoch is None or epoch > warmup

    if train_k > 1 and past_warmup and getattr(model, "supports_training_distribution", False):
        pred_e, e_var, pred_f, f_var = model.predict_training_distribution(
            batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch,
            num_mc_samples=train_k,
        )
        # Set once, at the warmup/NLL boundary, from that epoch's own
        # measured validation error (train()'s loop) - not a guessed
        # constant. Falls back to the tiny numerical default only for a
        # caller that never went through that boundary (e.g. a unit test
        # requesting the NLL branch directly via epoch=None).
        e_min_var = float(getattr(inner, "nll_min_energy_var", _NLL_VAR_EPS))
        f_min_var = float(getattr(inner, "nll_min_force_var", _NLL_VAR_EPS))
        energy_loss = _gaussian_nll_loss(pred_e, e_var, batch.y.view(-1), e_min_var)
        force_loss = _gaussian_nll_loss(pred_f, f_var, batch.force, f_min_var)
    else:
        pred_e, pred_f = model.predict_energy_and_forces(
            batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch,
        )
        energy_loss = F.mse_loss(pred_e, batch.y.view(-1))
        force_loss = F.mse_loss(pred_f, batch.force)

    loss = energy_w * energy_loss + force_w * force_loss

    kl_weight = float(getattr(inner, "kl_weight", 0.0))
    kl_term = 0.0
    if kl_weight > 0 and hasattr(inner, "kl_divergence"):
        kl = inner.kl_divergence()
        loss = loss + (kl_weight * kl) / n_batches
        kl_term = float(kl.detach()) * kl_weight / n_batches

    return loss, float(energy_loss.detach()), float(force_loss.detach()), kl_term


def run_epoch(model, loader, energy_w, force_w, device, optimizer=None, max_batches=None, epoch=None):
    """One pass over `loader`; returns mean (loss, energy MSE, force MSE, KL term) over batches.

    optimizer=None -> eval mode, no weight update. Forces still need
    autograd, so this is never wrapped in torch.no_grad().

    `epoch` just passes through to `batch_loss`'s NLL-warmup check - see
    that docstring. `train()` passes the real epoch number for both the
    train and val passes, so warmup applies identically to both.
    """
    is_train = optimizer is not None
    model.train(is_train)
    n_batches = len(loader)
    total_loss = total_e = total_f = total_kl = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        if is_train:
            optimizer.zero_grad()
        loss, e_loss, f_loss, kl_term = batch_loss(model, batch, energy_w, force_w, n_batches, epoch=epoch)
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
        total_loss += float(loss.detach())
        total_e += e_loss
        total_f += f_loss
        total_kl += kl_term
        n += 1
        if max_batches is not None and n >= max_batches:
            break
    return total_loss / n, total_e / n, total_f / n, total_kl / n


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

# Run records are this loop's own output, not a change to what is being run.
# Excluded from the dirty check so that, when configs run back to back, one
# run's record does not mark the next run as dirty. Everything else
# uncommitted - tracked or untracked - still counts.
DIRTY_CHECK_EXCLUDES = ("experiments/runs/",)


def git_state(root: Path = ROOT_DIR) -> dict[str, Any]:
    def git(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    pathspec = [".", *(f":(exclude){path}" for path in DIRTY_CHECK_EXCLUDES)]
    status = git("status", "--porcelain", "--", *pathspec)
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "git_dirty_ignores": list(DIRTY_CHECK_EXCLUDES),
    }


def environment(device: torch.device) -> dict[str, Any]:
    return {
        "device": str(device),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "numpy": np.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)


def _sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.int64).tobytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _subset_summary(info: dict[str, Any]) -> dict[str, Any]:
    return {
        **{k: v for k, v in info.items() if k not in ("positions", "gold_indices")},
        "gold_indices_sha256": _sha256(info["gold_indices"]),
        "first_gold_index": int(info["gold_indices"][0]),
        "last_gold_index": int(info["gold_indices"][-1]),
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def train(
    cfg: ExperimentConfig,
    *,
    output_root: Path = EXPERIMENTS_DIR,
    config_path: Optional[Path] = None,
    epochs: Optional[int] = None,
    max_batches: Optional[int] = None,
    score_test: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train `cfg` under the shared protocol. Returns the run record.

    `epochs` / `max_batches` override the config for development runs;
    any override, or `score_test=False`, stamps the record non-reportable.
    """
    overrides = {k: v for k, v in (("epochs", epochs), ("max_batches", max_batches)) if v is not None}
    reportable = score_test and not overrides
    output_root = Path(output_root)
    checkpoint_path = cfg.checkpoint_path(output_root / "checkpoints")
    run_dir = output_root / "runs" / cfg.name
    results_path = cfg.results_path(output_root / "results")
    for existing in (checkpoint_path, run_dir / "run.json"):
        if existing.exists() and not overwrite:
            raise FileExistsError(
                f"{_display_path(existing)} already exists; pass --overwrite to replace it"
            )

    device = torch.device(cfg.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("train.device is cuda but torch.cuda.is_available() is False")
    gold_path = resolve_gold_path(cfg)
    if not gold_path.exists():
        raise FileNotFoundError(
            f"no gold data at {gold_path}. Run `python -m ml.data.preprocessing` first - "
            "do not invent stand-in data (CLAUDE.md)."
        )

    git = git_state()
    if git["git_dirty"]:
        print("WARNING: the working tree has uncommitted changes; the recorded git "
              "commit does not fully describe this run.", flush=True)

    setup_start = time.perf_counter()
    # Seeded before the model is built, so initialisation is reproducible.
    torch.manual_seed(cfg.data.seed)
    inner = build_model(cfg)  # fails on a bad model: key before any data is read

    train_ds, val_ds = open_split(cfg, "train"), open_split(cfg, "val")
    subsets = select_subsets(cfg)
    train_positions = subsets["train"]["positions"]
    val_positions = subsets["val"]["positions"]

    # Per-atom statistics from the training subset only.
    n_atoms = int(train_ds.z.shape[0])
    train_energies = train_ds.E[torch.as_tensor(train_positions)]
    shift, scale = ScaledEnergyModel.statistics_from_energies(train_energies, n_atoms)
    model = ScaledEnergyModel(inner, shift=shift, scale=scale).to(device)
    if model.cutoff_radius != float(cfg.data.cutoff_radius):
        raise ValueError(
            f"the model's own cutoff is {model.cutoff_radius} but data.cutoff_radius is "
            f"{cfg.data.cutoff_radius}; they must match or messages are silently dropped"
        )
    n_parameters = int(sum(p.numel() for p in model.parameters()))

    batch_size = cfg.train.batch_size
    shuffle_generator = torch.Generator().manual_seed(cfg.data.seed)
    train_loader = DataLoader(
        torch.utils.data.Subset(train_ds, train_positions.tolist()),
        batch_size=batch_size, shuffle=True, generator=shuffle_generator,
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(val_ds, val_positions.tolist()), batch_size=batch_size, shuffle=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE
    )
    energy_w, force_w = cfg.train.energy_loss_weight, cfg.train.force_loss_weight
    max_epochs = epochs if epochs is not None else cfg.train.epochs
    patience = cfg.train.patience
    # BLIP's NLL path (0 for every other model, and for BLIP at its
    # default train_num_mc_samples=1): see batch_loss's docstring for why
    # both of these need handling here, not just in the loss function.
    nll_train_k = int(getattr(inner, "train_num_mc_samples", 1))
    nll_warmup = int(getattr(inner, "nll_warmup_epochs", 0))
    # batch_loss's own NLL condition, at run level: some epoch in
    # nll_warmup + 1 .. max_epochs actually takes the NLL branch.
    nll_active = (nll_train_k > 1 and nll_warmup < max_epochs
                  and getattr(model, "supports_training_distribution", False))
    setup_s = time.perf_counter() - setup_start

    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.csv"
    _write_json(run_dir / "subsets.json", {
        split: {**_subset_summary(info), "gold_indices": info["gold_indices"].tolist()}
        for split, info in subsets.items()
    })

    data_record = {
        "gold_path": cfg.data.gold_path,
        "molecule": cfg.data.molecule,
        "theory": cfg.data.theory,
        "cutoff_radius": float(cfg.data.cutoff_radius),
        "num_rbf": int(cfg.data.num_rbf),
    }
    record: dict[str, Any] = {
        "name": cfg.name,
        "phase": cfg.model.phase,
        "model": type(inner).__name__,
        "reportable": reportable,
        "overrides": overrides,
        "status": "running",
        **git,
        "config_path": _display_path(config_path) if config_path else None,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path else None,
        "config": cfg.to_dict(),
        "model_config": inner.config(),
        "n_parameters": n_parameters,
        "subsets": {split: _subset_summary(info) for split, info in subsets.items()},
        "energy_statistics": {
            "computed_on": "train subset",
            "n_frames": int(len(train_positions)),
            "energy_mean": shift * n_atoms,
            "energy_std": scale * n_atoms,
            "n_atoms": n_atoms,
            **model.energy_scaling(),
        },
        "protocol": {
            # The shared base loss: one text for every model, so benchmark.py
            # can hold it must-match (SCRUM-62). Byte-identical to every
            # record written before loss_override existed. A model-specific
            # replacement goes in loss_override, never in here.
            "loss": (
                "energy_loss_weight * MSE(E) + force_loss_weight * MSE(F), absolute E, plus kl_weight * "
                "model.kl_divergence() / n_batches when the model defines kl_weight (BLIP only; see "
                "ml/models/blip.py)"
            ),
            "loss_override": (
                "after nll_warmup_epochs epochs of the base loss, Gaussian NLL(E) + Gaussian NLL(F) on "
                "the train_num_mc_samples-sample mean/variance from predict_training_distribution "
                "replace the MSE terms, weighted by energy_loss_weight/force_loss_weight; variance floor "
                "(added, not clamped) = the last warmup epoch's validation MSE, or 1e-6 with no warmup; "
                "best-val selection and the LR scheduler reset at the first NLL epoch"
                if nll_active else None
            ),
            "energy_loss_weight": energy_w,
            "force_loss_weight": force_w,
            "kl_weight": float(getattr(inner, "kl_weight", 0.0)),
            "train_num_mc_samples": int(getattr(inner, "train_num_mc_samples", 1)),
            "nll_warmup_epochs": int(getattr(inner, "nll_warmup_epochs", 0)),
            "optimizer": {"name": "Adam", "lr": cfg.train.lr, "weight_decay": WEIGHT_DECAY},
            "scheduler": {"name": "ReduceLROnPlateau", "factor": LR_FACTOR, "patience": LR_PATIENCE},
            "grad_clip_norm": GRAD_CLIP,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "max_batches_per_epoch": max_batches,
            "shuffle": "torch.Generator seeded with data.seed",
            "selection": "lowest val loss on the val subset",
        },
        "environment": environment(device),
        "setup_s": setup_s,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": _display_path(checkpoint_path),
        "history_csv": _display_path(history_path),
        "epochs_run": 0,
        "best_epoch": None,
        "best_val_loss": None,
        "stop_reason": None,
        "wall_clock_s": None,
    }
    _write_json(run_dir / "run.json", record)

    def checkpoint_train_record(complete: bool) -> dict[str, Any]:
        keys = ("reportable", "overrides", "git_commit", "git_dirty", "config_sha256",
                "n_parameters", "epochs_run", "best_epoch", "best_val_loss", "stop_reason",
                "wall_clock_s", "subsets")
        return {**{k: record[k] for k in keys}, "config_name": cfg.name, "complete": complete,
                "run_record": _display_path(run_dir)}

    label = "" if reportable else "DEVELOPMENT RUN - not a result. "
    print(f"{label}{cfg.name}: {type(inner).__name__} ({n_parameters:,} parameters) on "
          f"{cfg.data.molecule}/{cfg.data.theory}, device {device}, git {git['git_commit']}", flush=True)
    print(f"train: {len(train_positions)} of {len(train_ds)} frames (stride "
          f"{subsets['train']['stride']}, offset {subsets['train']['offset']}); val: "
          f"{len(val_positions)} of {len(val_ds)} (stride {subsets['val']['stride']}, offset "
          f"{subsets['val']['offset']}); test: not opened", flush=True)
    print(f"energy per atom: shift {shift:.4f}, scale {scale:.4f} kcal/mol", flush=True)
    print(f"{'epoch':>5} {'train_loss':>12} {'train_E':>10} {'train_F':>10} {'train_KL':>10} "
          f"{'val_loss':>12} {'val_E':>10} {'val_F':>10} {'val_KL':>10} {'lr':>10} "
          f"{'train_s':>8} {'val_s':>8}", flush=True)

    best_val_loss = float("inf")
    best_state: Optional[dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0
    stop_reason = "max_epochs"

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        history = csv.DictWriter(history_file, fieldnames=HISTORY_FIELDS)
        history.writeheader()
        history_file.flush()

        train_start = time.perf_counter()
        try:
            for epoch in range(1, max_epochs + 1):
                epoch_lr = optimizer.param_groups[0]["lr"]
                epoch_start = time.perf_counter()
                train_loss, train_e, train_f, train_kl = run_epoch(
                    model, train_loader, energy_w, force_w, device,
                    optimizer=optimizer, max_batches=max_batches, epoch=epoch,
                )
                train_s = time.perf_counter() - epoch_start
                val_loss, val_e, val_f, val_kl = run_epoch(
                    model, val_loader, energy_w, force_w, device, max_batches=max_batches, epoch=epoch,
                )
                epoch_s = time.perf_counter() - epoch_start

                if nll_train_k > 1:
                    if epoch == nll_warmup:
                        # Last MSE-warmup epoch: calibrate the NLL
                        # variance floor to THIS epoch's real validation
                        # error, not a guessed constant - batch_loss's
                        # docstring. Read by next epoch's batch_loss call.
                        inner.nll_min_energy_var = max(val_e, 1e-6)
                        inner.nll_min_force_var = max(val_f, 1e-6)
                    if epoch == nll_warmup + 1:
                        # First NLL epoch: val_loss just changed what it
                        # means (MSE -> NLL). Comparing it against a
                        # best_val_loss set under the old loss, or
                        # carrying the scheduler's internal "best" across
                        # the switch, silently freezes the checkpoint at
                        # the last warmup epoch and reports an ordinary
                        # MSE model as the NLL result (Fazin's review,
                        # SCRUM-61) - reset both so the new regime is
                        # judged on its own terms from its own first epoch.
                        best_val_loss = float("inf")
                        epochs_without_improvement = 0
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE
                        )

                scheduler.step(val_loss)
                elapsed = time.perf_counter() - train_start

                improved = val_loss < best_val_loss
                record["epochs_run"] = epoch
                record["wall_clock_s"] = elapsed
                print(f"{epoch:>5} {train_loss:>12.4f} {train_e:>10.4f} {train_f:>10.4f} "
                      f"{train_kl:>10.4f} {val_loss:>12.4f} {val_e:>10.4f} {val_f:>10.4f} "
                      f"{val_kl:>10.4f} {epoch_lr:>10.2e} {train_s:>8.1f} "
                      f"{epoch_s - train_s:>8.1f}", flush=True)
                if improved:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    record["best_epoch"], record["best_val_loss"] = epoch, best_val_loss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    registry.save_checkpoint(
                        checkpoint_path, model, cfg.model.phase, data_record,
                        checkpoint_train_record(complete=False),
                    )
                else:
                    epochs_without_improvement += 1

                history.writerow({
                    "epoch": epoch, "lr": epoch_lr,
                    "train_loss": train_loss, "train_energy_mse": train_e, "train_force_mse": train_f,
                    "train_kl": train_kl,
                    "val_loss": val_loss, "val_energy_mse": val_e, "val_force_mse": val_f,
                    "val_kl": val_kl,
                    "train_s": train_s, "val_s": epoch_s - train_s, "epoch_s": epoch_s,
                    "elapsed_s": elapsed, "improved": int(improved),
                })
                history_file.flush()
                _write_json(run_dir / "run.json", record)

                if epochs_without_improvement >= patience:
                    stop_reason = "early_stopping"
                    print(f"Early stopping at epoch {epoch} (no val improvement for "
                          f"{patience} epochs)", flush=True)
                    break
        except KeyboardInterrupt:
            stop_reason = "interrupted"
            print(f"\nInterrupted during epoch {record['epochs_run'] + 1}; keeping the best "
                  "checkpoint so far.", flush=True)
        wall_clock_s = time.perf_counter() - train_start

    record["stop_reason"] = stop_reason
    record["wall_clock_s"] = wall_clock_s
    record["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if best_state is None:
        record["status"] = "failed: no epoch completed, no checkpoint written"
        _write_json(run_dir / "run.json", record)
        return record

    # Re-save the best weights with the final metadata, marked complete.
    model.load_state_dict(best_state)
    registry.save_checkpoint(
        checkpoint_path, model, cfg.model.phase, data_record,
        checkpoint_train_record(complete=stop_reason != "interrupted"),
    )
    record["checkpoint_sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    print(f"Best epoch {record['best_epoch']} (val_loss {best_val_loss:.4f}); "
          f"{record['epochs_run']} epochs in {wall_clock_s:.1f} s, stopped by {stop_reason}.", flush=True)
    print(f"Checkpoint: {_display_path(checkpoint_path)}", flush=True)

    if score_test and stop_reason != "interrupted":
        result = evaluate_checkpoint(
            checkpoint_path, split="test", batch_size=batch_size, device=cfg.train.device,
        )
        _write_json(run_dir / "test_metrics.json", result)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(results_path, result)
        record["test"] = {"metrics_json": "test_metrics.json", "benchmark_row": result["benchmark_row"]}
        row = result["benchmark_row"]
        print(f"Test: energy MAE {row['energy_mae']:.4f} {row['energy_mae_unit']}, "
              f"force MAE {row['force_mae']:.4f} {row['force_mae_unit']}", flush=True)
    record["status"] = "complete" if stop_reason != "interrupted" else "interrupted"
    _write_json(run_dir / "run.json", record)
    print(f"Run record: {_display_path(run_dir)}", flush=True)
    return record


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train one experiment config under the shared protocol.")
    parser.add_argument("config", help="Path to an experiments/configs/*.yaml file")
    parser.add_argument("--smoke-test", action="store_true",
                        help=f"{SMOKE_EPOCHS} epochs x {SMOKE_BATCHES} batches to check nothing is "
                             "broken - not a result")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override train.epochs (a development run - not a result)")
    parser.add_argument("--output-root", default=None,
                        help="where checkpoints/, runs/ and results/ go; default experiments/ for a "
                             "real run, a fresh temporary directory for a development run")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing checkpoint and run record")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    cfg = ExperimentConfig.from_yaml(config_path)
    epochs = SMOKE_EPOCHS if args.smoke_test else args.epochs
    max_batches = SMOKE_BATCHES if args.smoke_test else None
    development = epochs is not None
    if args.output_root:
        output_root = Path(args.output_root)
    elif development:
        output_root = Path(tempfile.mkdtemp(prefix=f"{cfg.name}_dev_"))
        print(f"Development run: outputs go to {output_root}", flush=True)
    else:
        output_root = EXPERIMENTS_DIR

    record = train(
        cfg, output_root=output_root, config_path=config_path, epochs=epochs,
        max_batches=max_batches, score_test=not development, overwrite=args.overwrite,
    )
    if record["status"] == "interrupted":
        return 130
    return 0 if record["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
