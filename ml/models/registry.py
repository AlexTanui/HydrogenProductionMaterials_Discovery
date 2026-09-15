"""Model name -> class lookup, so evaluation never needs to know which model it scores.

`ml/training/evaluate.py` must stay model-agnostic: every entry in the
Phase 1 bake-off gets its final numbers from that one file, and a single
`if phase == "painn"` branch there is how a benchmark table quietly stops
comparing like with like. This module is the one place that maps a config
name to a class.

**Checkpoint contract.** A checkpoint is a dict carrying everything needed
to rebuild the model that wrote it - evaluation reconstructs from the file
alone and never guesses a hyperparameter:

    {
        "phase":        "painn",           # a key of MODEL_REGISTRY
        "model_config": {...},             # constructor kwargs, from model.config()
        "state_dict":   {...},             # weights *and* buffers
        "data":         {                  # which data this was trained on
            "gold_path":     "data/gold/md17/ethanol_dft.npz",
            "molecule":      "ethanol",
            "theory":        "dft",
            "cutoff_radius": 5.0,
            "num_rbf":       16,
        },
        "train":        {...},             # optional: wall-clock, epochs, config name
    }

`save_checkpoint` / `load_model` below are the only sanctioned way to
write and read that shape. The training loop (SCRUM-51) should call
`save_checkpoint`; nothing else needs to know the layout.

Adding a model: import it and add one line to `MODEL_REGISTRY`. The class
must expose `predict_energy(z, pos, edge_index, edge_attr, batch)`,
`predict_energy_and_forces(...)` returning `(E, F)`, and `config()`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from ml.models.painn import PaiNN

__all__ = [
    "MODEL_REGISTRY",
    "REQUIRED_INTERFACE",
    "build_model",
    "save_checkpoint",
    "load_checkpoint",
    "load_model",
]

# Keyed by `model.phase` in the experiment YAML (glossary.md section 6).
#
# Note that glossary.md's documented enum is `mpnn | blip | graph_stochastic`
# - the three *research* phases. The SCRUM-50 bake-off adds candidate Phase 1
# backbones alongside those, so `painn` is an extension of that enum and
# glossary.md section 6 needs updating to match once the bake-off is agreed.
# The other four bake-off entries register themselves here as they land.
MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "painn": PaiNN,
}

# Every registered model must expose these. Checked at build time rather
# than at the first forward pass, so a mis-registered class fails where the
# cause is obvious instead of deep inside an eval loop.
REQUIRED_INTERFACE = ("predict_energy", "predict_energy_and_forces", "config")


class UnknownModel(KeyError):
    """A checkpoint or config names a model that is not registered."""


def _resolve(phase: str) -> type[nn.Module]:
    try:
        return MODEL_REGISTRY[phase]
    except KeyError:
        raise UnknownModel(
            f"unknown model phase {phase!r}; registered: {sorted(MODEL_REGISTRY)}. "
            "Add it to MODEL_REGISTRY in ml/models/registry.py rather than "
            "special-casing it in the training or evaluation loop."
        ) from None


def build_model(phase: str, **model_config: Any) -> nn.Module:
    """Construct a registered model from its config kwargs."""
    cls = _resolve(phase)
    missing = [name for name in REQUIRED_INTERFACE if not hasattr(cls, name)]
    if missing:
        raise TypeError(
            f"{cls.__name__} is registered as {phase!r} but does not implement "
            f"{missing}. Every bake-off model needs the same interface or the "
            "comparison is not like-for-like."
        )
    return cls(**model_config)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    phase: str,
    data: dict[str, Any],
    train: Optional[dict[str, Any]] = None,
) -> Path:
    """Write a checkpoint in the contract above. For the training loop to call."""
    if phase not in MODEL_REGISTRY:
        raise UnknownModel(
            f"refusing to save a checkpoint under unregistered phase {phase!r}; "
            f"registered: {sorted(MODEL_REGISTRY)}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "phase": phase,
            "model_config": model.config(),
            "state_dict": model.state_dict(),
            "data": dict(data),
            "train": dict(train or {}),
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Read a checkpoint and validate it carries the contract's fields."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no checkpoint at {path}. Train one first - the bake-off's "
            "training loop is SCRUM-51 (ml/training/train.py)."
        )
    # weights_only=False: the contract stores plain dicts of config alongside
    # the tensors. The file is produced by this repo's own training loop.
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"{path.name} is a {type(checkpoint).__name__}, not a checkpoint dict. "
            "A bare state_dict carries no model config, so evaluation cannot "
            "rebuild the model - see the contract in ml/models/registry.py."
        )
    missing = [k for k in ("phase", "model_config", "state_dict") if k not in checkpoint]
    if missing:
        raise ValueError(
            f"{path.name} is missing checkpoint field(s) {missing}; has "
            f"{sorted(checkpoint)}. See the contract in ml/models/registry.py."
        )
    return checkpoint


def load_model(path: str | Path, map_location: str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model a checkpoint describes. Returns (model in eval mode, checkpoint).

    `strict=True` on purpose: a silently-ignored missing key would load a
    partially-initialised model that still emits plausible energies.
    `energy_shift` / `energy_scale` are buffers precisely so they travel
    here and evaluation reproduces the training scale exactly.
    """
    checkpoint = load_checkpoint(path, map_location=map_location)
    model = build_model(checkpoint["phase"], **checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint
