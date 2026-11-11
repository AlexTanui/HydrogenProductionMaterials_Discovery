# DimeNet++ training-budget experiment

## Completed experiment

The 200-epoch training-budget experiment is complete. Under the original joint validation selection rule, epoch 200 is selected. Energy MAE increased by 11.07%, while force component MAE decreased by 26.56%. The result is a trade-off, not an improvement in every target. This report supersedes the 144-epoch interim snapshot.

## Controlled protocol

- Ethanol CCSD(T), unchanged Gold data: 889 training and 111 validation configurations.
- Seed 42; learning rate 0.0001; energy/force loss weights 1/100; unchanged architecture and batch size.
- One changed training factor: epoch budget, 100 to 200.
- Started from the same seeded initialization because the old checkpoint has no optimizer state. This is not an exact optimizer resume and is not an independent replicate.
- Checkpoint selection: minimum validation energy RMSE squared plus 100 times force RMSE squared.
- New selected epoch: 200. Budget completion alone does not establish convergence.
- Maximum metric difference over the first 100 epochs: 0.
- No test evaluation or test-derived correction was performed for this experiment.

## Validation results

Energy units: kcal/mol. Force units: kcal/mol/angstrom; Cartesian component metrics.
Positive reduction means a lower error; negative reduction means a higher error.

| Metric | Previous selected checkpoint | New selected checkpoint | Relative reduction |
|---|---:|---:|---:|
| energy_mae | 1.634226 | 1.815057 | -11.07% |
| energy_rmse | 1.749992 | 1.880484 | -7.46% |
| force_mae | 1.563769 | 1.148505 | +26.56% |
| force_rmse | 2.178289 | 1.581583 | +27.39% |

Weighted selection score: 477.556854 to 253.676836. Individual metrics need not all improve under a joint selection rule.

## Limitations and next step

This is one seed and one molecule. Shared-prefix runs are dependent. Validation was used for model selection, so these scores are not a fresh generalization estimate. Earlier test-error analysis informed possible follow-up work; do not present the old test set as an untouched new assessment. Repeat a fixed protocol with additional seeds and agree on a final evaluation protocol with the team.

## Reproduction

Training: `python -B -m ml.training.train_dimenet --config experiments/configs/dimenet_ethanol_lr1e4_200.yaml`

Run this only on dongxiao. The training runner deliberately refuses to overwrite an existing result directory. This command starts 200 epochs from scratch.

Analysis: `python build_budget_comparison.py --repo REPOSITORY --out OUTPUT_DIRECTORY`

The experiment reuses the existing runner, whose saved purpose field still says smoke test; this report describes the actual full-split training-budget comparison. Model weights stay local.

