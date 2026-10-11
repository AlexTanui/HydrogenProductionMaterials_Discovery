# Dongxiao DimeNet++: completed 200-epoch budget comparison

Finalized: October 1, 2026. Training completed September 24, 2026.

## Scope and selection

Ethanol CCSD(T), existing Gold data and saved indices: 889 training / 111 validation configurations. No new test evaluation. Same architecture, seed 42, learning rate 0.0001, batch size 4, and energy/force loss weights 1/100 as the 100-epoch baseline. The changed factor is the epoch budget. The new run starts from the same initialization; it does not resume optimizer state. All first-100-epoch metrics exactly match the original run.

Selected run: `dongxiao_dimenet_ethanol_lr1e4_200`, epoch 200. Selection uses minimum validation energy MSE + 100 times force component MSE. Verified directly from all 200 history records.

## Validation results

| Metric | 100-epoch selected model | 200-epoch selected model |
|---|---:|---:|
| Energy MAE, kcal/mol | 1.634226 | 1.815057 |
| Energy RMSE, kcal/mol | 1.749992 | 1.880484 |
| Force component MAE, kcal/mol/angstrom | 1.563769 | 1.148505 |
| Force component RMSE, kcal/mol/angstrom | 2.178289 | 1.581583 |
| Weighted validation selection score | 477.556854 | 253.676836 |

Energy MAE increased 11.07%; force component MAE decreased 26.56%. The original weighted selection criterion favors the 200-epoch model despite higher energy error. This does not establish an improvement in every target. Epoch 200 is the budget boundary; convergence is not established.

## Verification and artifacts

The checkpoint loads strictly. Recomputed validation metrics match the saved selected-epoch metrics exactly. Gold, model, training runner, and shared dataset source hashes match the pre-experiment record.

- Checkpoint SHA-256: `861ce3d11d8cf24fe79feb2e393c3738a37f9bff5bb206c5180674f72c221ba0`
- Gold SHA-256: `8e6571e1c0f0ac5a4fbab8a3236c26abff0095f54772866ad6ffa7f525deb441`
- Local checkpoint: `experiments/results/dongxiao_dimenet_ethanol_lr1e4_200/best.pt`
- Analysis: `notebooks/dongxiao_dimenet/performance_analysis/final_200/Training_Budget_Analysis.ipynb`
- Verification: `notebooks/dongxiao_dimenet/performance_analysis/final_200/checkpoint_verification.json`
- Report: `notebooks/dongxiao_dimenet/performance_analysis/Performance_Review.html`

The Notebook reanalyzes saved logs; it does not claim to rerun training. Logs for both budgets accompany it. The historical runner's purpose field says smoke test; the full Gold train/validation split and actual budget are recorded here. The 144-epoch report remains an explicitly historical snapshot.

## Limits and outstanding work

One molecule and one seed. The two runs are dependent, sharing their first 100 epochs. Validation is used for selection. The prior 100-epoch test result remains attached to the old model; it is not a test result for the new model. That test set was already inspected for error diagnostics, so future reuse must be disclosed as follow-up evaluation.

Energy-bias diagnosis on train/validation, new loss-weight experiments, additional seeds, and any further test evaluation remain separate work. No Git commit, push, or merge was performed as part of this closeout.
