"""DimeNet++ on existing MD17 graphs; no compiled graph extensions required.

Uses PyG's DimeNetPlusPlus layers and forward equations. Triplets are
enumerated with PyTorch for small MD17 batches (quadratic in edge count).
This adapter deliberately does not patch PyG or shared project modules.
"""
import torch
from torch_geometric.nn import DimeNetPlusPlus


class GoldDimeNetPlusPlus(DimeNetPlusPlus):
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


def energy_forces(model, batch, create_graph=False):
    batch.pos = batch.pos.detach().clone().requires_grad_(True)
    energy = model(batch)
    force = -torch.autograd.grad(energy.sum(), batch.pos, create_graph=create_graph)[0]
    return energy, force
