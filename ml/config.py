"""Experiment config schema (glossary.md section 6).

Loaded from experiments/configs/*.yaml by `ExperimentConfig.from_yaml`.
Unknown keys fail loudly - at the top level and in `data:` / `train:`
here, and in `model:` when ml/models/registry.py builds the model, where
the accepted keys are that model's constructor arguments. A misspelt key
that silently falls back to a default runs a different experiment from
the one the file describes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DataConfig:
    dataset: str = "md17"
    molecule: str = "ethanol"
    theory: str = "dft"
    gold_path: str = ""
    cutoff_radius: float = 5.0
    # Record of how the gold split was produced. The split itself is baked
    # into the gold file (train_idx/val_idx/test_idx) and never re-derived.
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    use_literature_split: bool = False
    # Width of MD17Dataset's Gaussian-RBF `edge_attr`. No registered model
    # reads it - each recomputes distances from `pos` with its own radial
    # basis, sized by its own `model:` keys - so this is the same for every
    # config and is not a model hyperparameter.
    num_rbf: int = 16
    seed: int = 42

    def __post_init__(self) -> None:
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train/val/test splits must sum to 1.0, got {total}")


@dataclass
class ModelConfig:
    """`phase` is a key of ml/models/registry.py's MODEL_REGISTRY; every
    other key under `model:` is passed to that model's constructor as-is."""

    phase: str = "mpnn"
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelConfig":
        raw = dict(raw)
        if "phase" not in raw:
            raise ValueError("model.phase is required: it names the model in ml/models/registry.py")
        phase = raw.pop("phase")
        if not isinstance(phase, str) or not phase:
            raise ValueError(f"model.phase must be a registry key (a non-empty string), got {phase!r}")
        return cls(phase=phase, kwargs=raw)

    def to_dict(self) -> dict[str, Any]:
        return {"phase": self.phase, **self.kwargs}


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 32
    lr: float = 5e-4
    energy_loss_weight: float = 1.0
    force_loss_weight: float = 100.0
    patience: int = 20
    device: str = "cpu"
    # Training budget. `train_subset` frames from train_idx and `val_subset`
    # frames from val_idx, each a fixed stride with an offset drawn from
    # `train_subset_seed` (ml/training/train.py::stride_subset). None means
    # the whole split. The test split is never subset - or read - here.
    train_subset: Optional[int] = None
    train_subset_seed: Optional[int] = None
    val_subset: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("train_subset", "val_subset"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"train.{name} must be a positive integer or omitted, got {value!r}")
        subsetting = self.train_subset is not None or self.val_subset is not None
        if subsetting and self.train_subset_seed is None:
            raise ValueError(
                "train.train_subset_seed is required when train.train_subset or "
                "train.val_subset is set: it fixes which frames every model sees"
            )


def _section(cls, raw: Optional[dict[str, Any]], section: str):
    raw = dict(raw or {})
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in `{section}:`; known keys: {sorted(known)}")
    return cls(**raw)


@dataclass
class ExperimentConfig:
    name: str = "unnamed_experiment"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        unknown = sorted(set(raw) - {"name", "data", "model", "train"})
        if unknown:
            raise ValueError(
                f"{path.name}: unknown top-level key(s) {unknown}; expected name, data, model, train"
            )
        return cls(
            name=raw.get("name", path.stem),
            data=_section(DataConfig, raw.get("data"), "data"),
            model=ModelConfig.from_dict(raw.get("model") or {}),
            train=_section(TrainConfig, raw.get("train"), "train"),
        )

    def to_dict(self) -> dict[str, Any]:
        """The resolved config, in the YAML's own shape."""
        return {
            "name": self.name,
            "data": asdict(self.data),
            "model": self.model.to_dict(),
            "train": asdict(self.train),
        }

    def checkpoint_path(self, checkpoints_dir: str | Path = "experiments/checkpoints") -> Path:
        return Path(checkpoints_dir) / f"{self.name}.pt"

    def results_path(self, results_dir: str | Path = "experiments/results") -> Path:
        return Path(results_dir) / f"{self.name}.json"
