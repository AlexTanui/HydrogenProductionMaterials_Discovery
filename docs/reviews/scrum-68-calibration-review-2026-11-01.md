# Review of SCRUM-68: post-hoc conformal calibration (1 Nov 2026)

**Reviewed:** `ml/training/calibrate.py`, `tests/ml/test_calibrate.py`, `backend/app/services/calibration.py` and its two callers, and `reports/calibration/*.json`, at `fazin-uq-decisions` (base `fazin-uq-protocol-v1`, which is `origin/main` plus the UQ protocol patch).

**Verdict: not ready to sign off as it stands, but one line away.**
- **The design is right.** It fits on validation only, uses split conformal with a studentized score, and fits a separate scale for energy and force.
- **One unit slip.** The conformal multiplier `q` is the half-width of the 90% interval in units of predicted std. The code uses `q × std` as the new standard deviation.
- **That slip alone explains the reported overshoot.** Coverage lands at 98–100% for a 90% target, and ECE stays near 0.13 after calibration. Validation→test non-exchangeability isn't needed to explain either.
- **Next step:** divide by z₀.₉₅ = 1.645, then re-run the three reports.

## What was checked

| Check | Result |
|---|---|
| Calibration fits on validation only; test only reports | **Yes.** `q` comes from the val split; the test subset is scored before and after, never fitted. |
| Conformal quantile and finite-sample correction | **Correct**, to within one order statistic. `np.quantile(scores, (1-α)(1+1/n), method="higher")` matched the textbook ⌈(n+1)(1−α)⌉-th smallest score for 363 of 404 calibration sizes; for the other 41 it takes the next one up, which is the conservative side. At n = 55,509 that's rank 49,960 instead of 49,959. Immaterial. |
| Coverage level the scale is fitted at | α = 0.1, so **90%**. One scalar per quantity. |
| **Is calibrated coverage on validation at the target?** | **No, by construction.** See below. |

## The finding

`conformal_scale` returns `q` such that the interval **ŷ ± q·σ** covers 90% of validation frames. That is its guarantee, and its own test says so: "q·std should land near the 90th percentile of |N(0,2)| ≈ 2 × 1.6449".

`calibrate_and_report` then scores `quantity_block(error, std * q)`, which reads `q·σ` as a **standard deviation**. The 90% Normal interval for that std is ±1.645·q·σ, wider by a factor of 1.645 than the interval `q` was fitted to. For Gaussian errors that interval covers P(|Z| ≤ 1.645 × 1.645) = **99.3%**, wherever the data comes from.

**Decisive check, with no shift possible.** I used Alex's own functions on synthetic data where the calibration and test sets come from the same distribution. The model claims std 1; the real errors are 3× wider. Coverage is at the 50/80/90/95% levels; script: `ml.training.calibrate.conformal_scale` + `uq_metrics.quantity_block`.

| Errors | Scored on | As calibrate.py reports (std × q) | With std × q / 1.6449 |
|---|---|---|---|
| Gaussian | the calibration set itself | 73.4 / 96.4 / **99.3** / 99.8%, ECE **0.135** | 50.2 / 80.1 / **90.0** / 95.0%, ECE 0.001 |
| Gaussian | a fresh set, same distribution | 72.7 / 96.4 / 99.3 / 99.9%, ECE 0.133 | 49.2 / 79.4 / 89.7 / 95.1%, ECE 0.004 |
| Student t (3 dof) | the calibration set itself | 78.9 / 94.3 / 97.0 / 98.1%, ECE 0.133 | 59.5 / 83.6 / 90.0 / 93.2%, ECE 0.037 |

**The real reports match that pattern.** The Gaussian prediction for "as reported" coverage is 73 / 96.5 / 99.3 / 99.9%, ECE 0.135:

| Report (test subset) | After calibration, as reported | ECE |
|---|---|---|
| `phase2_blip` energy | 68.3 / 97.8 / 99.6 / 99.98% | 0.127 |
| `phase2_blip_nll` energy | 65.5 / 98.2 / 99.7 / 99.96% | 0.121 |
| `phase3_graph_stochastic` energy | 75.2 / 96.6 / 99.1 / 99.8% | 0.139 |
| all three, force | 80.2–81.1 / 95.9–96.0 / 98.1–98.2 / 99.0% | 0.146–0.148 |

Energy errors look roughly Gaussian. Force errors look heavier-tailed, like the Student-t row. The overshoot and the residual ECE of 0.12–0.15 are what the factor of 1.645 produces with no shift at all. Whether some validation→test shift exists too can only be seen after the fix.

## What it affects

- **Unaffected:** `q_energy` and `q_force` as conformal multipliers, and every "before" number.
- **`reports/calibration/*.json` "after" blocks:** coverage, ECE, sharpness (1.645× too large), NLL and CRPS. So are the SCRUM-68 ticket's table and its "5–6× ECE improvement" claim.
- **`/benchmarks`:** its `*_calibrated` fields read those "after" blocks (`backend/app/api/benchmarks.py`).
- **`/predictions`:** `uncertainty_calibrated = uncertainty * q_energy` (`backend/app/services/inference.py:222`) is the 90% half-width, presented as a standard deviation.

## Proposed fix (for Alex; not applied, since these are Alex's files)

In `calibrate.py`, turn the conformal half-width into a Normal std before scoring, and store the multiplier that is actually applied:

```python
from scipy import stats

z = float(stats.norm.ppf(1.0 - alpha / 2.0))     # 1.6449 at alpha = 0.1
std_scale_energy, std_scale_force = q_energy / z, q_force / z
...
"std_scale_energy": std_scale_energy, "std_scale_force": std_scale_force,
"after": {
    "energy": quantity_block(test["e_error"], test["e_std"] * std_scale_energy),
    "force": quantity_block(test["f_error"], test["f_std"] * std_scale_force),
},
```

In `inference.py`, use `calibration["std_scale_energy"]` rather than `q_energy`.

Then re-run the three reports. `phase2_blip`'s should wait for its Step 7 retrain, because its current record is dirty.

**A test that would have caught it:** after scaling, coverage at the nominal 1−α level, measured on the calibration set itself, must be within about 1/√n of 1−α. Currently it's 0.993.

## What should go into the technical report, after the fix

- **Raw and calibrated numbers side by side, on the same test frames.** The raw numbers come from `evaluate.py` on all 55,509 test frames; the calibration reports use a 5,000-frame stride subset. Score both on one set, or say clearly which is which.
- **Caveats to state:**
  - The scale is one scalar fitted at 90%. With heavy-tailed force errors, the 50% and 95% levels will still miss. The synthetic Student-t case leaves ECE at about 0.04 even when exchangeable.
  - The guarantee assumes validation and test are exchangeable. The trajectory blocks are deliberately not, so post-fix test coverage at 90% is the measurement of that shift.
- **The decisive diagnostic to record:** coverage at 90% on validation after the fixed scaling, which should be ≈ 0.90 by construction, next to the same on test.

## Smaller points (no number changes)

- `_collect_errors_and_stds` calls `torch.manual_seed(0)` on the global RNG without restoring it. `evaluate.py` forks the RNG for exactly this reason.
- `_stride_subset` duplicates `ml.training.train.stride_subset`, which could be imported instead.
- `calibrate.py` reads the test split itself, for reporting only. `evaluate.py`'s docstring says the test split is read there and nowhere else. Either note the exception there, or have calibration score test through `evaluate_checkpoint`.
- Force calibration pools every (frame, atom, component) as one exchangeable set. The marginal guarantee is per component, not per frame. That's worth one sentence in the report.

## Re-running it needs these checkpoints (not on this laptop)

| Checkpoint | Must match (sha256 in its run record) | For |
|---|---|---|
| `experiments/checkpoints/phase2_blip_nll.pt` | `bf6c13fb…bb0ba54` | the diagnostic and fixed re-run above; the SCRUM-49 MC-count study |
| `experiments/checkpoints/phase3_graph_stochastic.pt` | `9412d664…16b4991f` | the same |
