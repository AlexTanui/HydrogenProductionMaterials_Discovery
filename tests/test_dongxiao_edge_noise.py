"""Coupled-noise physics checks using nonzero output weights."""
import unittest
import torch
from torch_geometric.data import Batch
from ml.models.dimenet import GoldDimeNetPlusPlus, energy_forces
from ml.models.dongxiao_edge_noise import EdgeNoiseDimeNet
from ml.training.train_dimenet import ROOT, load_gold
import yaml


class EdgeNoiseChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cfg = yaml.safe_load((ROOT/'experiments/configs/dimenet_smoke.yaml').read_text())
        cls.ds = load_gold(cfg)[0]['train']
        mc = dict(cfg['model']); mc.pop('name')
        cls.base = GoldDimeNetPlusPlus(**mc, output_initializer='glorot_orthogonal').double()
        cls.model = EdgeNoiseDimeNet(**mc).double()
        cls.model.load_state_dict(cls.base.state_dict(), strict=True)

    def batch(self):
        b = Batch.from_data_list([self.ds[0]])
        b = b.apply(lambda x: x.double() if x.is_floating_point() else x)
        b.edge_noise = torch.linspace(-1, 1, b.edge_index.shape[1], dtype=torch.double)[:, None]
        return b

    def test_zero_noise_identity_and_mc_variance(self):
        self.model.noise_std = 0
        e, f = energy_forces(self.model, self.batch())
        eb, fb = energy_forces(self.base, self.batch())
        torch.testing.assert_close(e, eb, atol=0, rtol=0)
        torch.testing.assert_close(f, fb, atol=0, rtol=0)
        self.model.noise_std = .05
        b = self.batch(); del b.edge_noise
        samples = torch.stack([self.model(b).detach() for _ in range(8)])
        self.assertGreater(samples.var().item(), 0)

    def test_fixed_noise_derivative_and_parameter_gradient(self):
        self.model.noise_std = .05
        e, f = energy_forces(self.model, self.batch(), create_graph=True)
        plus, minus = self.batch(), self.batch()
        plus.pos[0, 0] += 1e-5; minus.pos[0, 0] -= 1e-5
        fd = -(self.model(plus) - self.model(minus)) / 2e-5
        torch.testing.assert_close(f[0, 0], fd[0], atol=1e-5, rtol=1e-4)
        self.model.zero_grad()
        (e.square().mean()+f.square().mean()).backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))

    def test_coupled_rotation_translation_and_permutation(self):
        self.model.noise_std = .05
        b = self.batch(); e, f = energy_forces(self.model, b)
        q, _ = torch.linalg.qr(torch.tensor([[1.,2.,3.],[3.,-1.,1.],[2.,1.,-2.]], dtype=torch.double))
        moved = self.batch(); moved.pos = moved.pos @ q + 2.5
        er, fr = energy_forces(self.model, moved)
        torch.testing.assert_close(e, er, atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(f @ q, fr, atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(f.sum(0), torch.zeros(3, dtype=f.dtype), atol=1e-8, rtol=0)
        p = torch.arange(b.num_nodes-1, -1, -1)
        inverse = torch.argsort(p)
        permuted = self.batch(); permuted.x = b.x[p]; permuted.pos = b.pos.detach()[p]
        # Preserve edge identity and corresponding noise while relabelling nodes.
        permuted.edge_index = inverse[b.edge_index]
        ep, fp = energy_forces(self.model, permuted)
        torch.testing.assert_close(ep, e, atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(fp, f[p], atol=1e-8, rtol=1e-6)

    def test_batch_isolation_with_coupled_edge_noise(self):
        self.model.noise_std = .05
        single = self.batch(); e, f = energy_forces(self.model, single)
        combined = Batch.from_data_list([self.ds[0], self.ds[1]])
        combined = combined.apply(lambda x: x.double() if x.is_floating_point() else x)
        n = single.edge_index.shape[1]
        combined.edge_noise = torch.cat([single.edge_noise, torch.zeros(combined.edge_index.shape[1]-n, 1, dtype=torch.double)])
        ec, fc = energy_forces(self.model, combined)
        torch.testing.assert_close(ec[0], e[0], atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(fc[:single.num_nodes], f, atol=1e-8, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
