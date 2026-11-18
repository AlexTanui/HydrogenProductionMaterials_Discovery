# DimeNet++ edge-noise replication study

At validation-selected checkpoints, edge noise improved raw energy MAE in 2/3 fine-tuning seeds and force MAE in 0/3 seeds. These are paired runs from one pretrained checkpoint, not three independent end-to-end training initializations. No test evaluation was performed.

## Protocol

Fine-tuning seed sensitivity from ONE shared pretrained checkpoint; not independent pretraining seeds.

Gold splits, checkpoint, architecture, noise sigma 0.05, 20 epochs per arm, optimizer, validation MC streams and selection criterion.

Training batch shuffle seed and training noise seed only. Evaluation streams remain fixed across seeds.

Seed 42 is the earlier pilot; seeds 43 and 44 were specified before running the two replications. All runs are included. Fresh Adam optimizers and matched batch order within each pair. Same 889/111 Gold split. Checkpoint selection uses energy MSE + 100 times force-component MSE; no energy offset correction. Selection uses 8 noise samples; reporting uses 32 samples with a separate fixed random stream.

## Per-seed validation results

| Seed | Arm | Selected extra epoch | Energy MAE (kcal/mol) | Force MAE (kcal/mol/Angstrom) |
|---|---|---|---|---|
| 42 | deterministic | 18 | 1.1283 | 1.1309 |
| 42 | edge_gaussian | 19 | 0.4600 | 1.2704 |
| 43 | deterministic | 18 | 0.8374 | 1.1332 |
| 43 | edge_gaussian | 19 | 1.5507 | 1.3040 |
| 44 | deterministic | 19 | 0.7145 | 1.0961 |
| 44 | edge_gaussian | 17 | 0.4262 | 1.2767 |

## Mean and sample standard deviation across fine-tuning seeds

- deterministic: energy MAE 0.8934 +/- 0.2125; force MAE 1.1201 +/- 0.0208 (units as above).
- edge_gaussian: energy MAE 0.8123 +/- 0.6397; force MAE 1.2837 +/- 0.0179 (units as above).

Standard deviations describe these three runs; they are not confidence intervals or proof of significance.

## Uncertainty and matched retained subsets

Force Spearman compares structure-level mean MC variance with structure-level mean squared error. Both models below are evaluated on the SAME 77 structures retained after ranking by noisy-model uncertainty. Nominal rejection is 30%; rounding retains 77 of 111 structures.

| Seed | Force Spearman | Retained control force MAE | Retained noisy force MAE | Observed coverage of nominal 50% energy interval |
|---|---|---|---|---|
| 42 | 0.347 | 1.0539 | 1.1913 | 92.8% |
| 43 | 0.355 | 1.0565 | 1.2168 | 16.2% |
| 44 | 0.375 | 1.0245 | 1.1948 | 92.8% |

## Limitations and next step

One molecule, one pretrained checkpoint, three fine-tuning seeds and one fixed noise strength. Validation was used for model selection and reporting. Fixed MC streams reduce evaluation randomness but do not measure independent MC-seed variability. Raw energy-sampling intervals are not calibrated predictive intervals. No Bayesian posterior, OOD robustness or overall superiority is established. Do not replace the deterministic baseline solely on these diagnostics. A smaller predeclared noise-strength experiment or a team-shared backbone evaluation is a possible next study; no such study has been run here.

## Verification

All six selected-model errors were recomputed from saved MC arrays. Selection epochs, configs, source hashes, Gold checksum, saved split indices and common starting checkpoint were checked. Each arm completed exactly 20 epochs. Original results were preserved.
