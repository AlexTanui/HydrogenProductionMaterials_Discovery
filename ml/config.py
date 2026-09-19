from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    dataset: str = "md17"
    molecule: str = "ethanol"
    theory: str = "dft"
    gold_path: str = ""
    cutoff_radius: float = 5.0
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    use_literature_split: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train/val/test splits must sum to 1.0, got {total}")


@dataclass
class ModelConfig:
    phase: str = "mpnn"
    hidden_channels: int = 128
    num_layers: int = 4
    num_rbf: int = 50
    num_mc_samples: int = 20
    perturbation_strength: float = 0.1

    def __post_init__(self) -> None:
        valid_phases = ("mpnn", "blip", "graph_stochastic")
        if self.phase not in valid_phases:
            raise ValueError(f"model.phase must be one of {valid_phases}, got {self.phase!r}")


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 32
    lr: float = 5e-4
    energy_loss_weight: float = 1.0
    force_loss_weight: float = 100.0
    patience: int = 20
    device: str = "cpu"


@dataclass
class ExperimentConfig:
    name: str = "unnamed_experiment"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        return cls(
            name=raw.get("name", path.stem),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    def checkpoint_path(self, checkpoints_dir: str | Path = "experiments/checkpoints") -> Path:
        return Path(checkpoints_dir) / f"{self.name}.pt"

    def results_path(self, results_dir: str | Path = "experiments/results") -> Path:
        return Path(results_dir) / f"{self.name}.json"
