"""PyTorch Geometric Dataset wrapper over gold-stage MD17 data.

Reads only from data/gold/ (see glossary.md SS3) - never bronze/silver
directly. Migrated from rajcleaning.ipynb; see git history for changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from ml.data.md17 import build_graph


class MD17Dataset(Dataset):
    def __init__(self, gold_path: str | Path, split: str = "train", cutoff_radius: float = 5.0, num_rbf: int = 16):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        data = np.load(gold_path, allow_pickle=True)
        self.molecule = str(data["molecule"])
        self.theory = str(data["theory"])
        self.cutoff_radius = cutoff_radius
        self.num_rbf = num_rbf
        self.split = split
        idx = data[f"{split}_idx"]
        self.z = torch.as_tensor(data["z"], dtype=torch.long)
        self.R = torch.as_tensor(data["R"][idx], dtype=torch.float32)
        self.E = torch.as_tensor(data["E"][idx], dtype=torch.float32)
        self.F = torch.as_tensor(data["F"][idx], dtype=torch.float32)

    def len(self) -> int:
        return self.R.shape[0]

    def get(self, idx: int) -> Data:
        pos = self.R[idx]
        edge_index, edge_attr = build_graph(pos.numpy(), cutoff=self.cutoff_radius, num_rbf=self.num_rbf)
        return Data(x=self.z.view(-1, 1), pos=pos, edge_index=edge_index, edge_attr=edge_attr,
                    y=self.E[idx].view(1), force=self.F[idx])

    def __repr__(self) -> str:
        return (f"MD17Dataset(molecule={self.molecule!r}, theory={self.theory!r}, "
                f"split={self.split!r}, n_configs={len(self)}, n_atoms={self.z.shape[0]})")
