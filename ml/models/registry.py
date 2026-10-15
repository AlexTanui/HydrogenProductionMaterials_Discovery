"""Model name -> class lookup, so evaluation never needs to know which model it scores.

`ml/training/evaluate.py` must stay model-agnostic: every entry in the
Phase 1 bake-off gets its final numbers from that one file, and a single
`if phase == "painn"` branch there is how a benchmark table quietly stops
comparing like with like. This module is the one place that maps a config
name to a class.

**Checkpoint contract (version 2).** A checkpoint is a dict carrying
everything needed to rebuild the model that wrote it - evaluation
reconstructs from the file alone and never guesses a hyperparameter:

    {
        "checkpoint_version": 2,
        "phase":        "painn",           # a key of MODEL_REGISTRY
        "model_config": {...},             # constructor kwargs, from model.config()
        "energy_scaling": {...},           # shift/scale as numbers - for reading only
        "state_dict":   {...},             # ScaledEnergyModel's: "model.*" weights and
                                           # buffers, plus the "shift"/"scale" buffers
        "data":         {                  # which data this was trained on
            "gold_path":     "data/gold/md17/ethanol_dft.npz",
            "molecule":      "ethanol",
            "theory":        "dft",
            "cutoff_radius": 5.0,
            "num_rbf":       16,
        },
        "train":        {...},             # optional: wall-clock, epochs, config name
    }

Every checkpoint holds a model wrapped in `ml.models.scaling.ScaledEnergyModel`,
the one energy scale-and-shift the harness applies to every model, so
`load_model` returns a model that predicts absolute energies with no
per-model special case. Version 1 checkpoints (bare models, written by the
superseded per-model trainers) are refused: their energy convention differs
from the harness's, and scoring them as if it did is how a benchmark row
ends up ~97,000 kcal/mol off.

`save_checkpoint` / `load_model` below are the only sanctioned way to
write and read that shape. `ml/training/train.py` calls `save_checkpoint`;
nothing else needs to know the layout.

Adding a model: import it and add one line to `MODEL_REGISTRY`. The class
must expose `predict_energy`, `predict_energy_and_forces(z, pos,
edge_index, edge_attr, batch)` returning `(E, F)`, and `config()`. The
harness only ever calls `predict_energy_and_forces`; `predict_energy`'s
signature still differs between models (PaiNN and DimeNet++ need `pos`).
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from ml.models.blip import BLIP
from ml.models.dimenet import GoldDimeNetPlusPlus
from ml.models.mpnn import MPNN
from ml.models.painn import PaiNN
from ml.models.physnet import PhysNet
from ml.models.scaling import ScaledEnergyModel
from ml.models.schnet import SchNet

__all__ = [
    "CHECKPOINT_VERSION",
    "MODEL_REGISTRY",
    "REQUIRED_INTERFACE",
    "build_model",
    "save_checkpoint",
    "load_checkpoint",
    "load_model",
]

# Keyed by `model.phase` in the experiment YAML (glossary.md section 6): the
# five Phase 1 bake-off backbones plus Phase 2's BLIP. Phase 3
# (`graph_stochastic`) registers here when ml/models/graph_stochastic.py lands.
MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "mpnn": MPNN,
    "schnet": SchNet,
    "painn": PaiNN,
    "physnet": PhysNet,
    "dimenet": GoldDimeNetPlusPlus,
    "blip": BLIP,
}

CHECKPOINT_VERSION = 2

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
    """Construct a registered model from its config kwargs.

    Unknown kwargs raise rather than being dropped: a misspelt config key
    that silently falls back to a constructor default trains a different
    model from the one the config describes.
    """
    cls = _resolve(phase)
    missing = [name for name in REQUIRED_INTERFACE if not hasattr(cls, name)]
    if missing:
        raise TypeError(
            f"{cls.__name__} is registered as {phase!r} but does not implement "
            f"{missing}. Every bake-off model needs the same interface or the "
            "comparison is not like-for-like."
        )
    parameters = inspect.signature(cls.__init__).parameters
    takes_var_kwargs = any(p.kind is p.VAR_KEYWORD for p in parameters.values())
    if not takes_var_kwargs:
        # A class whose __init__ takes **kwargs (DimeNet++) checks its own.
        accepted = [name for name in parameters if name != "self"]
        unknown = sorted(set(model_config) - set(accepted))
        if unknown:
            raise TypeError(
                f"model.phase={phase!r} ({cls.__name__}) does not accept "
                f"{unknown}; accepted model keys: {accepted}"
            )
    return cls(**model_config)


def save_checkpoint(
    path: str | Path,
    model: ScaledEnergyModel,
    phase: str,
    data: dict[str, Any],
    train: Optional[dict[str, Any]] = None,
) -> Path:
    """Write a checkpoint in the contract above. For the training loop to call.

    Written atomically: to a temporary file first, then renamed over `path`.
    A run killed mid-write therefore still leaves the previous checkpoint
    intact instead of a truncated file in its place.
    """
    if phase not in MODEL_REGISTRY:
        raise UnknownModel(
            f"refusing to save a checkpoint under unregistered phase {phase!r}; "
            f"registered: {sorted(MODEL_REGISTRY)}"
        )
    if not isinstance(model, ScaledEnergyModel):
        raise TypeError(
            f"save_checkpoint takes the ScaledEnergyModel the harness trained, not a bare "
            f"{type(model).__name__}: a bare model carries no energy scale-and-shift, so "
            "evaluation would score it about 97,000 kcal/mol off. See ml/models/scaling.py."
        )
    if not isinstance(model.model, MODEL_REGISTRY[phase]):
        raise TypeError(
            f"phase {phase!r} is {MODEL_REGISTRY[phase].__name__}, but the wrapped model "
            f"is a {type(model.model).__name__}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "phase": phase,
            "model_config": model.model.config(),
            "energy_scaling": model.energy_scaling(),
            "state_dict": model.state_dict(),
            "data": dict(data),
            "train": dict(train or {}),
        },
        tmp,
    )
    os.replace(tmp, path)
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
    version = checkpoint.get("checkpoint_version", 1)
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"{path.name} is checkpoint version {version}; this harness reads version "
            f"{CHECKPOINT_VERSION}, where every model carries the shared energy "
            "scale-and-shift (ml/models/scaling.py). Version 1 checkpoints come from the "
            "superseded per-model trainers and use a different energy convention - "
            "retrain through ml/training/train.py rather than scoring this one."
        )
    return checkpoint


def load_model(
    path: str | Path, map_location: str = "cpu"
) -> tuple[ScaledEnergyModel, dict[str, Any]]:
    """Rebuild the model a checkpoint describes. Returns (wrapped model in eval mode, checkpoint).

    The registered model is rebuilt from `model_config`, wrapped in
    `ScaledEnergyModel`, and the whole state_dict - including the `shift`
    and `scale` buffers - is loaded, so evaluation reproduces training's
    absolute energies exactly. `strict=True` on purpose: a silently-ignored
    missing key would load a partially-initialised model that still emits
    plausible energies.
    """
    checkpoint = load_checkpoint(path, map_location=map_location)
    model = ScaledEnergyModel(build_model(checkpoint["phase"], **checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint
