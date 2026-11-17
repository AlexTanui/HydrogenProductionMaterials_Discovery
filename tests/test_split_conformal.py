"""Tests use artificial arrays only; they are not molecular experiment results."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from ml.utils.split_conformal import absolute_residuals, conformal_quantile, prediction_intervals, evaluate
from ml.training.evaluate_split_conformal import run, sha256, validate_manifest


def fixture(root):
    rng=np.random.default_rng(42)
    for role,start,n in (("calibration",0,199),("test",1000,100)):
        prediction=rng.normal(0,2,n)
        np.savez(root/(role+".npz"),gold_index=np.arange(start,start+n),
                 E_pred=prediction,E_ref=prediction+rng.normal(0,.5,n))
    m=dict(dataset_sha256="a"*64,checkpoint_sha256="b"*64,model_identity="SYNTHETIC TEST FIXTURE",
        energy_unit="kcal/mol",energy_kind="total_absolute",illustrative=True,
        calibration_unused_for_fit_selection_tuning=True,test_unused_for_method_selection=True,
        independence_evidence="Independent seeded synthetic arrays; no trained model.",
        trajectory_limitations="No MD trajectories; not scientific evidence.",
        fit_selection_tuning_ids=[],final_test_ids=list(range(1000,1100)))
    for role in ("calibration","test"):
        m[role]=role+".npz"; m[role+"_sha256"]=sha256(root/m[role])
    (root/"manifest.json").write_text(json.dumps(m),encoding="utf-8")
    return m


class ConformalChecks(unittest.TestCase):
    def test_exact_ranks(self):
        scores=np.arange(1,20)
        self.assertEqual([conformal_quantile(scores,c) for c in (.8,.9,.95)],[16,18,19])

    def test_small_sample_is_unbounded(self):
        self.assertTrue(np.isinf(conformal_quantile([1,2,3],.95)))
        lo,hi=prediction_intervals([1,2],float("inf"))
        self.assertTrue(np.isneginf(lo).all() and np.isposinf(hi).all())

    def test_ties_and_inclusive_boundary(self):
        r=evaluate([1]*19,[0]*19,[1,-1,2],[0,0,0])
        self.assertEqual(r[1]["empirical_coverage"],2/3)
        self.assertEqual(r[1]["average_interval_width"],2)

    def test_invalid_input(self):
        for bad in ([],[np.nan],[np.inf],[[1],[2]]):
            with self.assertRaises(ValueError): conformal_quantile(bad)
        for level in (0,1,-1,np.nan):
            with self.assertRaises(ValueError): conformal_quantile([1],level)
        with self.assertRaises(ValueError): conformal_quantile([-1])
        with self.assertRaises(ValueError): absolute_residuals([1,2],[1])

    def test_finite_sample_rank_guarantee(self):
        # Exhaustive held-out ranks of 20 distinct exchangeable scores.
        for confidence in (.8,.9,.95):
            covered=sum(v<=conformal_quantile(np.delete(np.arange(20),v),confidence) for v in range(20))
            self.assertEqual(covered/20,confidence)

    def test_metrics_and_no_point_prediction_change(self):
        prediction=np.array([0.,0.])
        r=evaluate([1]*19,[0]*19,[3,4],prediction)[0]
        self.assertAlmostEqual(r["energy_mae"],3.5)
        self.assertAlmostEqual(r["energy_rmse"],np.sqrt(12.5))
        np.testing.assert_array_equal(prediction,[0,0])

    def test_manifest_rejects_leakage_and_invalid_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); m=fixture(root)
            with np.load(root/"calibration.npz") as f: c=dict(f)
            with np.load(root/"test.npz") as f: t=dict(f)
            validate_manifest(m,c,t)
            for change in ({"fit_selection_tuning_ids":[0]},
                           {"fit_selection_tuning_ids":[1000]},
                           {"calibration_unused_for_fit_selection_tuning":False},
                           {"test_unused_for_method_selection":False},
                           {"energy_kind":"per_atom"}):
                with self.assertRaises(ValueError): validate_manifest(dict(m,**change),c,t)
            c["gold_index"][1]=c["gold_index"][0]
            with self.assertRaises(ValueError): validate_manifest(m,c,t)

    def test_cli_outputs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); fixture(root)
            result=run(root/"manifest.json",root/"output")
            self.assertEqual(len(result),3)
            for name in ("comparison.csv","results.json","coverage.png","energy_intervals.png"):
                self.assertGreater((root/"output"/name).stat().st_size,0)
            self.assertEqual(json.loads((root/"output/results.json").read_text())["status"],
                             "ILLUSTRATIVE_SYNTHETIC_ONLY")
            with self.assertRaises(FileExistsError): run(root/"manifest.json",root/"output")
            with (root/"test.npz").open("ab") as f: f.write(b"changed")
            with self.assertRaisesRegex(ValueError,"hash mismatch"): run(root/"manifest.json",root/"other")


if __name__=="__main__":
    unittest.main()
