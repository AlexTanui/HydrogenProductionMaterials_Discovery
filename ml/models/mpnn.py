from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

MAX_ATOMIC_NUMBER = 100


class GaussianRBF(nn.Module):

    def __init__(self, num_rbf: int = 50, cutoff: float = 5.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_rbf)
        width = centers[1] - centers[0] if num_rbf > 1 else torch.tensor(cutoff)
        self.register_buffer("centers", centers)
        self.width = width

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        d = distances.view(-1, 1)
        return torch.exp(-((d - self.centers.view(1, -1)) ** 2) / (2 * self.width ** 2))


class MPNNLayer(nn.Module):

    def __init__(self, hidden_channels: int, num_rbf: int):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_channels + num_rbf, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_rbf: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        messages = self.message_mlp(torch.cat([h[col], edge_rbf], dim=-1))

        aggregated = torch.zeros_like(h)
        aggregated.index_add_(0, row, messages)

        return h + self.update_mlp(torch.cat([h, aggregated], dim=-1))


class MPNN(nn.Module):
    def __init__(self, hidden_channels: int = 128, num_layers: int = 4,
                 num_rbf: int = 50, cutoff: float = 5.0):
        super().__init__()
        self.embedding = nn.Embedding(MAX_ATOMIC_NUMBER + 1, hidden_channels)
        self.rbf = GaussianRBF(num_rbf=num_rbf, cutoff=cutoff)
        self.layers = nn.ModuleList([MPNNLayer(hidden_channels, num_rbf) for _ in range(num_layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def predict_energy(self, z: torch.Tensor, edge_index: torch.Tensor,
                        edge_attr: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`edge_attr` must already be RBF-expanded distance, shape
        [n_edges, num_rbf] - `ml.data.datasets.MD17Dataset` (via
        `ml.data.md17.build_graph`) hands it out in exactly that shape."""
        h = self.embedding(z.view(-1))

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        atomic_energies = self.readout(h).view(-1)

        if batch is None:
            return atomic_energies.sum().view(1)

        n_graphs = int(batch.max().item()) + 1
        total_energy = torch.zeros(n_graphs, dtype=atomic_energies.dtype, device=atomic_energies.device)
        total_energy.index_add_(0, batch, atomic_energies)
        return total_energy

    def predict_energy_and_forces(self, z: torch.Tensor, pos: torch.Tensor,
                                   edge_index: torch.Tensor, edge_attr: torch.Tensor,
                                   batch: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """`edge_attr` is accepted only to match the shared interface - the
        actual energy pass uses RBF features recomputed from `pos` (static
        `edge_index` topology reused as-is) so that `dE/dR` is a real
        gradient instead of disconnected from the graph - see
        `ml/models/schnet.py`'s module docstring for why reusing the
        precomputed `edge_attr` here would break forces."""
        pos = pos.requires_grad_(True)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        live_edge_attr = self.rbf(dist)

        energy = self.predict_energy(z, edge_index, live_edge_attr, batch=batch)

        grad_outputs = torch.ones_like(energy)
        (dE_dR,) = torch.autograd.grad(
            outputs=energy, inputs=pos, grad_outputs=grad_outputs,
            create_graph=self.training, retain_graph=True,
        )
        forces = -dE_dR
        return energy, forces
