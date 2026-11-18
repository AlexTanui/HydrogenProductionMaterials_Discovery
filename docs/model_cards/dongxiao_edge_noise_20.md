# Dongxiao: fixed edge-noise DimeNet++ pilot

Compared with matched deterministic fine-tuning, edge noise changed raw validation energy MAE from 1.128 to 0.460 kcal/mol and force-component MAE from 1.131 to 1.270 kcal/mol/Angstrom. Force uncertainty/error Spearman was 0.347. Energy MAE improved at the selected checkpoints, while force MAE worsened; the joint objective is dominated by force MSE. This does not demonstrate overall superiority. Removing the nominal highest-uncertainty 30% of structures reduced force MAE within the noisy model to 1.191; this selected subset is not directly comparable to the control on all structures. The nominal 50% raw energy interval covered 92.8% of validation targets, suggesting over-wide intervals in this sample. A completed experiment does not imply an accuracy improvement or calibrated uncertainty. This is one molecule, one training seed and one fixed noise strength, with validation-based selection and no new test evaluation. Previously inspected test results remain historical. Next steps are replication and a team-agreed shared-backbone comparison; do not tune parameters on the historical test set.

## Protocol

See `experiments/configs/dongxiao_edge_noise_20.yaml`. Gold only; saved 889/111 training/validation split. Both arms start from the same 200-epoch checkpoint with fresh Adam state; 20 epochs each. No node-noise model, learned noise scale, Bayesian objective, BLIP implementation, or team-wide benchmark is claimed.

## deterministic

Selected additional epoch: 18. Noise standard deviation: 0.0.

Checkpoint: `experiments/results/dongxiao_edge_noise_20/deterministic_best.pt` (ignored by Git).

SHA256: `ae5027a03ff60ea46469b1857c74bab0906d77168accf96f6ed8f533995fc323`.

## edge_gaussian

Selected additional epoch: 19. Noise standard deviation: 0.05.

Checkpoint: `experiments/results/dongxiao_edge_noise_20/edge_gaussian_best.pt` (ignored by Git).

SHA256: `8565d7349f9768cde9ee9abe199eddbaf52bb3d7da00e9a9a1cbf7346d89f0fb`.

## Evaluation definitions

Force MAE averages Cartesian components. Spearman uses one MSE and one mean component variance per structure. Empirical energy-interval coverage is uncalibrated and descriptive. Validation is reused for selection and reporting; estimates are not an independent holdout. All energy metrics in this experiment are raw, without the historical train-fitted additive correction.
