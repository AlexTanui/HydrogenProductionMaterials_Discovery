"""Physics and Gold boundary checks for Dongxiao's independent adapter."""
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.data import Batch

from ml.models.dimenet import GoldDimeNetPlusPlus, energy_forces
from ml.training.train_dimenet import ROOT, load_gold


class DimeNetChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.config = yaml.safe_load((ROOT/'experiments/configs/dimenet_smoke.yaml').read_text())
        cls.datasets, cls.mean, cls.selected, _ = load_gold(cls.config)
        mc = dict(cls.config['model'])
        mc.pop('name')
        # Nonzero output initialization makes derivative checks meaningful.
        cls.model = GoldDimeNetPlusPlus(**mc, output_initializer='glorot_orthogonal').double()

    def batch(self, indices=(0,)):
        return Batch.from_data_list([self.datasets['train'][i] for i in indices]).apply(lambda x: x.double() if x.is_floating_point() else x)

    def test_energy_force_and_parameter_derivatives(self):
        batch = self.batch()
        energy, force = energy_forces(self.model, batch, create_graph=True)
        h = 1e-5
        plus, minus = self.batch(), self.batch()
        plus.pos[0, 0] += h
        minus.pos[0, 0] -= h
        finite_difference = -(self.model(plus) - self.model(minus)) / (2*h)
        torch.testing.assert_close(force[0, 0], finite_difference[0], atol=1e-5, rtol=1e-4)
        self.model.zero_grad()
        (energy.square().mean() + force.square().mean()).backward()
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(g.abs().sum() > 0 for g in gradients))

    def test_batch_isolation_rotation_and_translation(self):
        e, f = energy_forces(self.model, self.batch())
        combined = self.model(self.batch((0, 1)))
        torch.testing.assert_close(e[0], combined[0], atol=1e-9, rtol=1e-7)
        q, _ = torch.linalg.qr(torch.tensor([[1.,2.,3.],[3.,-1.,1.],[2.,1.,-2.]],dtype=torch.float64))
        transformed = self.batch()
        transformed.pos = transformed.pos @ q + 2.5
        er, fr = energy_forces(self.model, transformed)
        torch.testing.assert_close(er, e, atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(fr, f @ q, atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(f.sum(0), torch.zeros(3,dtype=f.dtype),atol=1e-8,rtol=0)

    def test_saved_gold_indices_and_train_only_center(self):
        with np.load(ROOT/self.config['data']['gold_path']) as gold:
            np.testing.assert_array_equal(self.selected['train'],gold['train_idx'][:32])
            np.testing.assert_array_equal(self.selected['val'],gold['val_idx'][:16])
            self.assertAlmostEqual(self.mean,float(gold['E'][gold['train_idx'][:32]].astype(np.float64).mean()))
        bad = dict(self.config, data=dict(self.config['data'], gold_path='data/bronze/md17/md17_ethanol.npz'))
        with self.assertRaisesRegex(ValueError, 'requires a file under'):
            load_gold(bad)


if __name__ == '__main__':
    unittest.main()

