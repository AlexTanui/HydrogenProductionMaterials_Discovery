# Final frozen test evaluation: 200-epoch DimeNet++ checkpoint

Date: October 1, 2026. Ethanol CCSD(T), existing Gold test split.

## Evaluation protocol

The completed 200-epoch checkpoint was frozen before evaluation. It was selected at epoch 200 using the validation criterion `energy MSE + 100 × force component MSE`. The test set contains 1,000 held-out configurations with 9 atoms each.

The original test set had been inspected during the earlier 100-epoch baseline study. This is therefore a disclosed follow-up evaluation of the same test set, not a new untouched benchmark. No test result was used for tuning after this evaluation.

Energy offset calibration was decided before this test evaluation. The offset, 1.749300800 kcal/mol, was fitted from the 889 training configurations only. It was then applied once to the frozen 200-epoch predictions. Forces were not changed.

## Results

| Result | Energy total MAE | Energy total RMSE | Signed energy bias | Force component MAE | Force component RMSE |
|---|---:|---:|---:|---:|---:|
| Previous 100-epoch baseline, original | 1.607 | 1.736 | +1.576 | 1.530 | 2.140 |
| 200-epoch checkpoint, original energy | 1.805 | 1.873 | +1.804 | 1.165 | 1.617 |
| 200-epoch checkpoint, train-fitted offset | **0.391** | **0.508** | **+0.055** | 1.165 | 1.617 |

The 200-epoch checkpoint improves the force component metrics relative to the previous baseline. Its raw absolute energy error is higher, while the training-fitted offset removes most of the systematic energy shift on this test evaluation. The offset is post-processing, not a new trained model, and must remain associated with this checkpoint and data composition.

Additional 200-epoch metrics: energy per-atom MAE 0.0435 kcal/mol; force atom-norm MAE 2.330 kcal/mol/angstrom. The test set was not used to fit the offset.

## Interpretation and limits

The results support the earlier diagnosis of a nearly constant positive energy bias. A position-independent energy constant has zero position derivative, so it does not affect force predictions. This explains why energy calibration can change energy metrics while leaving forces unchanged.

This remains one molecule, one seed, one small architecture, and one fixed Gold composition. The shared 100- and 200-epoch runs are dependent through their first 100 epochs. The calibrated score should not be compared with another model unless the same training-only calibration protocol is used. Multi-seed validation and an agreed team-level evaluation protocol remain future work.

## Reproduction and files

- Raw and calibrated predictions: `dimenet_200_test_predictions.npz`
- Metrics: `dimenet_200_final_test.json`
- Residual plot: `dimenet_200_test_residuals.png`
- Evaluation script: `evaluate_dimenet_200_test.py`
- 200-epoch checkpoint: `experiments/results/dongxiao_dimenet_ethanol_lr1e4_200/best.pt`
- Gold SHA-256: `8e6571e1c0f0ac5a4fbab8a3236c26abff0095f54772866ad6ffa7f525deb441`
- Checkpoint SHA-256: `861ce3d11d8cf24fe79feb2e393c3738a37f9bff5bb206c5180674f72c221ba0`

No training data or shared source files were modified.
