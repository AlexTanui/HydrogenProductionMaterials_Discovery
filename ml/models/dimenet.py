"""DimeNet++ on existing MD17 graphs; no compiled graph extensions required.

Uses PyG's DimeNetPlusPlus layers and forward equations. Triplets are
enumerated with PyTorch for small MD17 batches (quadratic in edge count).
This adapter deliberately does not patch PyG or shared project modules.

The shared harness (`ml/models/registry.py`, `ml/training/train.py`,
`ml/training/evaluate.py`) calls every model through
`predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)`.
`predict_energy` / `predict_energy_and_forces` below are thin adapters that
pack those tensors into the object `forward` already reads; `forward` and
`energy_forces` are unchanged. Geometry comes from `pos` only - the
dataset's `edge_attr` is accepted for the interface and ignored, as in
every other model.
"""
import inspect
from types import SimpleNamespace

import torch
from torch_geometric.nn import DimeNetPlusPlus


class GoldDimeNetPlusPlus(DimeNetPlusPlus):
    def __init__(self, **kwargs):
        # Bind against PyG's own signature, so an unknown or missing
        # argument raises TypeError, and the resolved arguments (defaults
        # included) are what config() hands the registry.
        bound = inspect.signature(DimeNetPlusPlus.__init__).bind(None, **kwargs)
        bound.apply_defaults()
        super().__init__(**kwargs)
        self._config = {k: v for k, v in bound.arguments.items() if k != 'self'}

    def config(self):
        """Constructor kwargs, for embedding in a checkpoint. See ml/models/registry.py."""
        return dict(self._config)

    def forward(self, data):
        z, pos = data.x.reshape(-1).long(), data.pos
        j, i = data.edge_index
        # Edge a is j->i; edge b is k->j. Exclude returning to i.
        idx_ji, idx_kj = torch.where((j[:, None] == i[None, :]) &
                                     (i[:, None] != j[None, :]))
        idx_i, idx_j, idx_k = i[idx_ji], j[idx_ji], j[idx_kj]
        dist = (pos[i] - pos[j]).square().sum(-1).sqrt()
        pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
        angle = torch.atan2(torch.linalg.cross(pos_ij, pos_jk).norm(dim=-1),
                            (pos_ij * pos_jk).sum(-1))
        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)
        x = self.emb(z, rbf, i, j)
        out = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
        for interaction, output in zip(self.interaction_blocks, self.output_blocks[1:]):
            x = interaction(x, rbf, sbf, idx_kj, idx_ji)
            out = out + output(x, rbf, i, num_nodes=pos.size(0))
        batch = data.batch
        result = out.new_zeros((data.num_graphs, 1))
        return result.index_add(0, batch, out).flatten()

    def predict_energy(self, z, pos, edge_index, edge_attr=None, batch=None):
        """Total energy per molecule, shape (B,), from `forward`. `edge_attr` is ignored."""
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        data = SimpleNamespace(x=z, pos=pos, edge_index=edge_index, batch=batch,
                               num_graphs=int(batch.max()) + 1)
        return self(data)

    def predict_energy_and_forces(self, z, pos, edge_index, edge_attr=None, batch=None):
        """Energy (B,) and forces (N, 3) as -dE/dpos, the same maths as `energy_forces`.

        create_graph only while training, so evaluation does not retain a
        double-backward graph. Grad is re-enabled locally, so this also
        works under an enclosing torch.no_grad().
        """
        with torch.enable_grad():
            if not pos.requires_grad:
                pos = pos.detach().requires_grad_(True)
            energy = self.predict_energy(z, pos, edge_index, edge_attr, batch)
            (grad,) = torch.autograd.grad(energy.sum(), pos, create_graph=self.training)
        return energy, -grad


def energy_forces(model, batch, create_graph=False):
    batch.pos = batch.pos.detach().clone().requires_grad_(True)
    energy = model(batch)
    force = -torch.autograd.grad(energy.sum(), batch.pos, create_graph=create_graph)[0]
    return energy, force
