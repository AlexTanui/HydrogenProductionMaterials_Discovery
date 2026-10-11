"""SchNet (Schutt et al. 2017) - Phase 1 bake-off baseline, SCRUM-53.

Continuous-filter convolutions over Gaussian-expanded interatomic
distances. Exposes the same two methods as the MPNN prototype
(raj_mpnn/MPNN.ipynb, see ml/models/mpnn.py / SCRUM-51) so it's a drop-in
for the shared training loop once it lands:

    predict_energy(z, edge_index, edge_attr, batch)
    predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)

`ml.data.datasets.MD17Dataset` (via ml/data/md17.py:build_graph) already
hands out `edge_attr` as Gaussian-RBF-expanded distance, shape
[n_edges, num_rbf] - not a raw scalar distance. SchNet's filter network is
built to consume exactly that, so `num_rbf` here must match the Dataset's
`num_rbf`.

**Why predict_energy_and_forces recomputes edge features from `pos`
instead of reusing the passed-in `edge_attr`:** the MPNN prototype in
raj_mpnn/MPNN.ipynb calls `pos.requires_grad_(True)` but then computes
energy from the precomputed `edge_attr` tensor, which has no graph
connection back to `pos` at all - `torch.autograd.grad(energy, pos)` on
that would raise (or silently return None), because nothing in the energy
computation actually depends on `pos`. Forces need a real computational
path from positions to energy, so here the RBF features are rebuilt from
`pos` inside this method (static `edge_index` topology reused as-is; only
the distances need to be live). Worth checking ml/models/mpnn.py doesn't
repeat the same bug when it lands.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

MAX_ATOMIC_NUMBER = 100


class ShiftedSoftplus(nn.Module):
    """Softplus shifted to equal 0 at x=0 - SchNet's activation (Schutt et al. 2017)."""

    def __init__(self) -> None:
        super().__init__()
        self.shift = float(torch.log(torch.tensor(2.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x) - self.shift


class GaussianRBF(nn.Module):
    """Expands scalar distances into a Gaussian radial basis.

    Only used inside predict_energy_and_forces, to rebuild RBF features
    from a live `pos` so gradients flow back to atomic positions - see the
    module docstring. Kept identical in form to
    ml.data.md17._gaussian_rbf so a model trained on the Dataset's
    precomputed edge_attr and one recomputed here from pos see the same
    feature distribution.
    """

    def __init__(self, num_rbf: int = 16, cutoff: float = 5.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        self.width = float(centers[1] - centers[0]) if num_rbf > 1 else cutoff

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((dist.unsqueeze(-1) - self.centers) ** 2) / (2 * self.width**2))


class CFConv(nn.Module):
    """Continuous-filter convolution (one SchNet interaction block).

    A filter network maps RBF-expanded edge distance to a per-channel
    filter; the source atom's features are passed through a linear layer,
    modulated (elementwise) by that filter, then summed into the target
    atom - the mechanism that makes the convolution continuous in
    interatomic distance rather than discretized into edge types.
    """

    def __init__(self, hidden_channels: int, num_rbf: int):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.lin_in = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.lin_out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_rbf: torch.Tensor) -> torch.Tensor:
        row, col = edge_index  # row = target node, col = source/neighbor node
        w = self.filter_net(edge_rbf)                # [n_edges, hidden]
        messages = self.lin_in(h[col]) * w            # [n_edges, hidden]

        aggregated = torch.zeros_like(h)
        aggregated.index_add_(0, row, messages)       # sum incoming messages per node

        return h + self.lin_out(aggregated)            # residual update


class SchNet(nn.Module):
    def __init__(self, hidden_channels: int = 128, num_layers: int = 4,
                 num_rbf: int = 16, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff
        self.embedding = nn.Embedding(MAX_ATOMIC_NUMBER + 1, hidden_channels)
        self.rbf = GaussianRBF(num_rbf=num_rbf, cutoff=cutoff)
        self.interactions = nn.ModuleList(
            [CFConv(hidden_channels, num_rbf) for _ in range(num_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def predict_energy(self, z: torch.Tensor, edge_index: torch.Tensor,
                        edge_attr: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`edge_attr` must already be RBF-expanded distance, shape
        [n_edges, num_rbf] - see the module docstring."""
        h = self.embedding(z.view(-1))

        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_attr)

        atomic_energies = self.readout(h).view(-1)  # [n_atoms]

        if batch is None:
            return atomic_energies.sum().view(1)

        n_graphs = int(batch.max().item()) + 1
        total_energy = torch.zeros(n_graphs, dtype=atomic_energies.dtype, device=atomic_energies.device)
        total_energy.index_add_(0, batch, atomic_energies)
        return total_energy

    def predict_energy_and_forces(self, z: torch.Tensor, pos: torch.Tensor,
                                   edge_index: torch.Tensor, edge_attr: torch.Tensor,
                                   batch: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """`edge_attr` is accepted only to match the shared interface -
        the actual energy pass uses RBF features recomputed from `pos`
        (see module docstring) so forces come out as a real gradient,
        not zeros."""
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
