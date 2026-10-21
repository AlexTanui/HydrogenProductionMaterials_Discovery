"""Analytic checks for MC axes, undefined deterministic UQ and rejection ranking."""
import unittest
import numpy as np
from ml.training.dongxiao_edge_experiment import metrics, uq_metrics


class UQChecks(unittest.TestCase):
    def test_structure_axis_and_perfect_uncertainty_order(self):
        target_e = np.zeros(4)
        target_f = np.zeros((4, 2, 3))
        mean = np.arange(1., 5.)
        e = np.stack([mean*.9, mean*1.1])
        f = np.broadcast_to(e[:, :, None, None], (2, 4, 2, 3)).copy()
        settings = dict(random_seed=1, random_repetitions=100,
                        reject_fractions=[0., .5], coverage_levels=[.5, .8])
        result = uq_metrics(e, f, target_e, target_f, settings)
        self.assertAlmostEqual(result['force_mae'], 2.5)
        self.assertAlmostEqual(result['energy_spearman'], 1.)
        self.assertAlmostEqual(result['force_spearman'], 1.)
        self.assertAlmostEqual(result['rejection'][1]['force_mae'], 1.5)
        self.assertEqual(result['rejection'][1]['retained_count'], 2)
        self.assertEqual(result['raw_energy_sample_interval_coverage'][0]['observed'], 0.)

    def test_deterministic_uncertainty_is_unavailable(self):
        e, f = np.zeros((1, 4)), np.zeros((1, 4, 2, 3))
        result = uq_metrics(e, f, e[0], f[0], {})
        self.assertFalse(result['uncertainty_available'])
        self.assertNotIn('force_spearman', result)
        self.assertEqual(metrics(e, f, e[0], f[0])['force_mae'], 0.)


if __name__ == '__main__':
    unittest.main()
