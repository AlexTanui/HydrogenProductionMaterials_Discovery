# Energy bias diagnosis: selected 100- and 200-epoch checkpoints

Analysis date: October 1, 2026. Ethanol CCSD(T), existing Gold splits.
Both checkpoints were evaluated in inference mode on all 889 training and 111 validation configurations. No test predictions were computed or read. No correction or retraining was performed.

## Results

All energy values below are in kcal/mol. Signed error is prediction minus reference; a positive value means overprediction.

| Checkpoint | Split | Mean signed error | MAE | Residual standard deviation | Overpredicted |
|---|---|---:|---:|---:|---:|
| Epoch 100 | Training | +1.516 | 1.540 | 0.729 | 95.7% |
| Epoch 100 | Validation | +1.622 | 1.634 | 0.656 | 96.4% |
| Epoch 200 | Training | +1.749 | 1.749 | 0.508 | 100.0% |
| Epoch 200 | Validation | +1.815 | 1.815 | 0.492 | 100.0% |

The 200-epoch checkpoint overpredicts every evaluated training and validation energy. Its validation MAE therefore equals its mean signed error. The error distribution is narrower than at epoch 100, but shifted further above zero.

The identity MSE = mean(error)^2 + variance(error) shows that squared mean bias accounts for 93.2% of validation MSE at epoch 200, compared with 86.0% at epoch 100. These are descriptive decompositions of the original errors, not corrected model scores.

The epoch-200 training and validation MAEs are 1.749 and 1.815. Their similar values do not show a large train-validation gap in these aggregate energy metrics. This does not rule out overfitting in other metrics or settings; it indicates that a systematic positive offset is the immediate issue visible here.

## Interpretation

The evidence is consistent with a model that captures relative energy variation more closely while retaining a larger constant offset. This observation does not identify the underlying optimization cause. The existing energy/force loss weights may contribute to the trade-off but have not been isolated experimentally.

Forces are negative derivatives of energy with respect to positions. Adding or subtracting a constant energy offset does not change those derivatives. Therefore, force predictions can improve while an energy offset remains.

## Suggested next controlled check (not performed)

Estimate one additive energy offset from the training split only, freeze it, and compare the original and adjusted predictions on validation. Keep the original checkpoint and record the offset as a separate post-processing artifact. Do not use validation or test residuals to fit the offset. This would test whether a constant shift explains most of the error; it would not establish better generalization by itself.

If training settings are changed later, compare one loss-weight change at a time using the same validation selection protocol, and retain both energy and force metrics. The previously inspected test set should not be used for iterative tuning.

## Reproduction and verification

Run `python -B analyze_training_bias.py --repo REPOSITORY --out OUTPUT_DIRECTORY` using the project environment. It loads existing checkpoints, validates the Gold hash and training mean, and checks recomputed validation MAE against the saved training log within 0.00001 kcal/mol.

References are computed in float64 from Gold energies minus the original training mean. Minor differences around 0.00000003 kcal/mol from training logs are due to the logs' float32 centered targets.

The plot uses shared horizontal limits and bin boundaries. The saved residual arrays contain training/validation energy errors only. Training metrics here use a fixed selected checkpoint, unlike online training metrics accumulated while parameters change within an epoch.
