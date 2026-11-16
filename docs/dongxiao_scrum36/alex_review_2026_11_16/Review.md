# SCRUM-36: review of Alex's first-pass analysis

Reviewer: Dongxiao Wu | 16 November 2026

## Evidence and attribution

This extends Alex Tanui's 9 November first-pass analysis rather than replacing its authorship. The inspected `alex` branch snapshot is `cb949a5cb472e20dff00c33f8174f604c9c10cef`. Seventeen source files are preserved under `sources/`; `source_manifest.json` records immutable upstream links and SHA256 hashes. No checkpoint was retrained or re-scored during this review. Numerical verification uses saved records, not raw predictions.

## Corrections and updates

| First-pass statement | Reviewed disposition |
|---|---|
| Compare BLIP NLL with Graph base as the headline | SCRUM-40's 12 November resolution selects BLIP NLL versus Graph NLL: both use four training draws and 20 warmup epochs. Keep the base comparison as secondary historical context. |
| Runs have not been re-scored | Superseded by 14 November `test_metrics.json` rescore blocks. Maximum relative differences: BLIP NLL 7.748e-10, Graph base 1.374e-10, Graph NLL 4.555e-11. The generated Markdown table still says otherwise. |
| Graph force coverage 57.2% is overcoverage | It is undercoverage relative to 90%. Energy overcoverage does not imply force overcoverage. |
| Raw ECE/Spearman values can be combined across tables | The 55,509-frame benchmark and 5,000-frame calibration evaluation are different evaluation populations and stochastic passes. Label them separately. |
| Phase 3 corrected widths are 15-40% larger | This describes Graph base CQR. For the matched Graph NLL pair, the recorded differences are +133.77% energy and +0.21% force; force widths are practically similar in this snapshot. |
| Both corrections are monotone rescalings | Scalar calibration rescales standard deviations; the CQR path adds a common offset to interval endpoints. Its JSON does not report Spearman, so do not claim measured unchanged CQR uncertainty/error ranking. |
| Re-score proves complete reproducibility | The recorded checks reproduce listed accuracy/benchmark metrics. They report a dirty scoring tree plus evaluator hash; this is not a clean-environment reproduction of training, every coverage statistic, or calibration. |

## Current headline: matched training hooks

Use the existing full-test results for BLIP NLL versus Graph NLL. Graph NLL has energy MAE 0.1292 versus 0.0750 (+72.3%) and force MAE 0.1846 versus 0.1787 (+3.3%). Raw energy ECE is 0.2007 versus 0.6470; force ECE is 0.0277 versus 0.7379. Lower ECE accompanies worse point accuracy in this pair. Shared hooks do not eliminate differences in parameter count, learned variance floors, code revision, or model capacity. This is not an isolated causal experiment on noise location.

Recorded training times are 2.29 versus 3.98 hours and inference times 6.253 versus 12.878 ms/frame for Graph NLL versus BLIP NLL. These are source-reported timings, not a fresh controlled speed benchmark.

## Post-correction evidence: separate 5,000-frame evaluation

All six calibration reports record 55,509 calibration frames and 5,000 evaluation frames, at nominal 90%. The saved implementation loads the Gold validation split for calibration and a stride subset of the Gold test split for evaluation. Energy widths use kcal/mol; force-component widths use kcal/mol/Angstrom.

| Method | Model | Energy coverage | Energy width | Force coverage | Force width |
|---|---|---:|---:|---:|---:|
| CQR | BLIP NLL | 91.54% | 0.3170 | 91.742% | 0.8373 |
| CQR | Graph base | 90.58% | 0.4433 | 91.873% | 0.9641 |
| CQR | Graph NLL | 90.30% | 0.7411 | 91.662% | 0.8390 |

| Method | Model | Energy coverage | Energy ECE | Force coverage | Force ECE |
|---|---|---:|---:|---:|---:|
| Scalar | BLIP NLL | 91.54% | 0.01735 | 91.885% | 0.04468 |
| Scalar | Graph base | 90.94% | 0.01100 | 92.003% | 0.04826 |
| Scalar | Graph NLL | 91.08% | 0.00920 | 91.616% | 0.04146 |

These are observed marginal coverages, not exactly matched coverages or proof of conditional calibration. Calibration does not improve the underlying point predictions. Graph NLL scalar force ECE increases from 0.02802 to 0.04146 on this subset even while its 90% coverage moves toward the target: improving one coverage level need not improve average error across all levels.

## Calibration validity and statistical limits

1. Calibration uses the whole validation split, while training uses validation for selection/scheduling. A dedicated calibration set independent of these decisions is not established. Disjointness from training alone is insufficient. Do not present these reports as our independent Split Conformal baseline or as a validated finite-sample coverage guarantee.
2. Calibration JSONs do not export exact frame IDs, split fingerprints, or complete invocation provenance. The current source explains defaults but cannot certify every historical invocation. Frame-level overlap checks and trajectory boundaries remain required.
3. MD frames are correlated; pooled force components within a structure are also dependent. Do not use all components as independent observations for significance tests.
4. `calibrate.py` implements the corrected quantile via a level passed to NumPy `method="higher"`, and clamps out-of-range levels. This is not the exact order-statistic implementation `ceil((n+1)*(1-alpha))` with an infinite interval when the rank exceeds n. Audit with the metric owner before claiming the standard correction. This review does not edit team calibration code.
5. Each variant has one training run. Re-scoring and changing inference seeds do not supply independent training replications. No p-values or confidence intervals are justified from these summaries alone.
6. SCRUM-49 found no tested K satisfying its stated three-inference-seed stability criterion for BLIP NLL and Graph base. This does not prove larger K can never help and does not directly test Graph NLL. Small ranking differences should not be treated as established improvements.
7. The structural explanation in SCRUM-70 remains a hypothesis requiring controlled tests. A gated additive perturbation has first spatial derivative `eps * ds/dR`; the zero derivative of the sampled noise does not prove the remaining pathway weak, second-order, or an inevitable energy/force trade-off. Downstream nonlinear propagation must also be considered.

## Acceptance status and next actions

The first-pass review, corrected comparison, and integration into `docs/technical_report.md` are complete. SCRUM-36 can be reviewed as a descriptive write-up, but inferential analysis is not complete.

- Ask the evaluation owner to regenerate stale benchmark Markdown from updated rescore records.
- Agree exact calibration provenance, independent selection/calibration separation, and quantile handling with the calibration owner.
- Export paired per-frame predictions and trajectory IDs; assess block-wise errors and dependence before bootstrap analysis.
- Run predeclared replicated training only if approved compute and schedule permit; keep results exploratory because existing test results have informed decisions.

## Suggested Jira update (not posted)

Reviewed Alex's first-pass SCRUM-36 analysis at alex@cb949a5 and extended docs/technical_report.md. Updated the headline to the SCRUM-40 matched NLL pair; verified 14 November rescore records; separated full-test raw metrics from 5,000-frame post-correction results; corrected force undercoverage and the CQR width comparison. Sources and a reproducible verification script are included. Remaining limitations: one training run per variant, calibration/selection independence and provenance, dependent MD frames, and no paired uncertainty intervals/significance analysis yet. Ready for team review as a descriptive report; not claiming inferential completion.
