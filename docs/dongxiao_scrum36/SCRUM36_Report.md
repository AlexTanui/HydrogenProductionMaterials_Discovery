> Historical analysis, 15 November 2026. See the [16 November review](alex_review_2026_11_16/Review.md) for the current matched-pair headline, BLIP MSE exclusion, rescore updates, and post-correction results.

# SCRUM-36: Phase 3 versus Phase 1/2
Author: Dongxiao Wu. Analysis date: 15 November 2026.

## Scope and evidence
This is a reproducible descriptive analysis of existing team results, not a new training
or inference experiment. It uses raw, uncalibrated uncertainty outputs and excludes the
post-hoc conformal/CQR reports, other architectures and DimeNet CCSD(T) experiments.
Repository snapshot: 75dc38f5005b2e9821b4c2ab09e6ab4cd58a5da7.
The source manifest contains immutable GitHub links and SHA256 hashes for 15 input files.

The numerical comparison and initial write-up are complete. Inferential statistical
analysis remains pending aligned per-frame predictions and replicated training runs.
This draft does not establish statistical significance or close that part of SCRUM-36.

## Comparability audit
All five records use ethanol DFT and the same test fingerprint, representing 55,509
configurations with nine atoms each. The exact 10,000 training IDs and 5,000 validation IDs
match across all five runs, as do their recorded hashes. Data and training configuration
blocks match. Training seed is 42; scoring seed is 0; evaluation batch size is 32.
All stochastic models use 20 Monte Carlo draws. These draws are not independent training
replications. The deterministic record has no uncertainty estimates; N/A must not be
interpreted as zero ECE.

Energy errors refer to total energy in kcal/mol; force errors refer to Cartesian
components in kcal/mol/Angstrom. Test fingerprints are recorded evidence, not independently
recomputed from raw trajectories in this analysis. There are 1,498,743 force components,
but they are not 1,498,743 independent statistical observations.

The common configuration does NOT make this a controlled one-factor causal ablation.
Model sizes differ (MPNN 384,897; BLIP 740,953; Graph 385,925 parameters). MSE and NLL
training differ; NLL uses four training draws and 20 warmup epochs, while MSE variants
use one draw. Model-specific variance floors differ. Checkpoint-selection descriptions
and training code revisions also differ: the MPNN is from 755d57f, BLIP MSE/NLL and
Graph MSE from 1eaea76, and Graph NLL from c9502b2. All five records report clean
working trees. The MPNN's older selection wording does not prove identical selection
semantics; a common-code rerun would strengthen attribution.

## Point prediction results
| Model | Energy MAE | Energy RMSE | Force MAE | Force RMSE |
|---|---|---|---|---|
| MPNN | 0.0779 | 0.0962 | 0.1775 | 0.2774 |
| BLIP MSE | 0.0920 | 0.1120 | 0.1826 | 0.2828 |
| BLIP NLL | 0.0750 | 0.0943 | 0.1787 | 0.2788 |
| Graph MSE | 0.0897 | 0.1126 | 0.2088 | 0.3264 |
| Graph NLL | 0.1292 | 0.1612 | 0.1846 | 0.2919 |

Lower error is better. BLIP NLL has the lowest recorded energy MAE/RMSE; the deterministic
MPNN has the lowest recorded force MAE/RMSE. These rankings describe these checkpoints only.

### Where Phase 3 does not beat BLIP
- Graph MSE versus BLIP MSE: energy MAE is 2.45% lower, but energy RMSE is 0.52% higher,
  force MAE 14.35% higher and force RMSE 15.40% higher. A lower energy MAE alone does not
  establish overall improvement.
- Graph MSE versus BLIP NLL: energy MAE is 19.63% higher and force MAE 16.86% higher.
- Graph NLL versus BLIP NLL: energy MAE is 72.31% higher and force MAE 3.28% higher.
  Both RMSE values are also higher (70.91% energy and 4.69% force).
- Graph NLL versus BLIP MSE: energy MAE is 40.51% higher and force MAE 1.07% higher.
- Relative to deterministic MPNN, both Phase 3 variants have higher energy and force errors.

Percentages use (Phase 3 - comparator) / comparator * 100, with positive values meaning
higher error. Absolute differences and all four error metrics are provided in
phase3_differences.csv. No significance is implied by these percentages.

## Raw uncertainty results
| Model | Energy ECE | Force ECE | Energy Spearman | Force Spearman |
|---|---|---|---|---|
| MPNN | N/A | N/A | N/A | N/A |
| BLIP MSE | 0.5964 | 0.6715 | -0.0025 | 0.2641 |
| BLIP NLL | 0.6470 | 0.7379 | 0.0190 | 0.2836 |
| Graph MSE | 0.1538 | 0.2969 | 0.0021 | 0.2622 |
| Graph NLL | 0.2007 | 0.0277 | 0.0013 | 0.2948 |

ECE is the equal-weight mean absolute difference between observed and nominal coverage
at 50%, 80%, 90% and 95%, under Normal(mean, variance) central intervals. The eight
energy/force ECE values were independently recomputed from the saved coverage fractions.
Spearman correlates predicted standard deviation with absolute error: per frame for energy,
and pooled Cartesian components for forces. Spearman values are copied from source records;
they cannot be independently recomputed without paired prediction arrays.

| Model | Energy coverage @90% | Force coverage @90% | Mean energy std | Mean force std |
|---|---|---|---|---|
| MPNN | N/A | N/A | N/A | N/A |
| BLIP MSE | 0.2263 | 0.1373 | 0.0231 | 0.0206 |
| BLIP NLL | 0.1662 | 0.0586 | 0.0122 | 0.0087 |
| Graph MSE | 0.9946 | 0.5737 | 0.2143 | 0.1137 |
| Graph NLL | 0.9999 | 0.8631 | 0.4969 | 0.2107 |

Coverage values are fractions, not percentages. Mean standard deviations have the same
units as the corresponding energy/force errors. Under these Gaussian intervals, mean
width at a fixed nominal level is proportional to mean standard deviation. Better coverage
must therefore be considered together with interval sharpness.

Graph variants have lower ECE than both BLIP variants in these records. Graph NLL is
especially better on force ECE (0.0277 versus 0.7379 for BLIP NLL), and its observed
force coverage at nominal 90% is 86.31%, versus 5.86% for BLIP NLL. It still undercovers.

Graph NLL energy intervals cover 99.99% at nominal 90%, and Graph MSE covers 99.46%.
This is overcoverage, not evidence of perfectly calibrated predictions. Graph NLL has
worse energy ECE than Graph MSE (0.2007 versus 0.1538) and substantially larger mean
energy standard deviation (0.4969 versus 0.2143).

Energy uncertainty-error correlations are near zero for every stochastic variant.
Graph MSE force Spearman (0.2622) is below both BLIP variants; Graph NLL force Spearman
(0.2948) exceeds BLIP NLL (0.2836), but the small observed gap has no significance
assessment. The results do not support a blanket statement that Phase 3 is better.

## Statistical interpretation and remaining work
Only one trained checkpoint per configuration is available in the selected records.
Aggregated MAE/RMSE/ECE values do not contain the joint per-frame error distribution,
so valid paired confidence intervals or p-values cannot be reconstructed from them.
A nominally large frame/component count does not resolve this problem.

Recommended follow-up, to be agreed before additional testing:
1. Obtain frozen, aligned exports for all five models: Gold ID, original trajectory ID
   and source-frame/time mapping, E_ref/E_pred, F_ref/F_pred, and E_var/F_var for stochastic
   models. Record checkpoint hash, source revision, units and MC seed/count.
2. Verify identical frame IDs/references and recompute the published metrics first.
   A checkpoint is not required if these audited exports already exist.
3. Compute paired per-frame absolute-energy errors and per-frame mean absolute force
   component errors. Keep all atoms/components of a frame together.
4. If suitable trajectory provenance is available, use paired trajectory/block resampling,
   applying the same sampled blocks to all models. Choose block lengths using training-only
   dependence diagnostics or physical reasoning; report sensitivity across predeclared lengths.
   Do not randomly bootstrap individual Cartesian components.
5. Block bootstrap is not automatically valid for nonstationary trajectories; assess stationarity,
   effective independent blocks and trajectory boundaries first. If unsuitable, restrict conclusions
   to descriptive block-wise results or obtain independent trajectories.
6. Repeat training with multiple predeclared seeds and a common code/protocol to measure
   training variability. Seed 42 used across different model architectures does not imply
   identical initialization. Do not treat MC draws as extra training seeds.
7. Predeclare primary comparisons/metrics and any multiple-testing correction before
   significance testing. These tests will remain exploratory because the test results
   have already informed discussion.

No confidence intervals or p-values have been fabricated. Training times and inference
times are preserved in the CSV as source-reported context; no speed-up claim is made
without a dedicated common-hardware benchmark.

## Findings for the technical report
Moving stochasticity into graph representations improved raw interval calibration relative
to BLIP in these recorded ethanol experiments, but did not deliver a uniform accuracy gain.
In particular, both Phase 3 variants had larger energy and force MAE than BLIP NLL.
The NLL graph variant substantially improved force coverage compared with BLIP NLL while
retaining severe energy overcoverage and weak energy uncertainty ranking. The evidence
therefore indicates a trade-off between point accuracy and uncertainty behavior, rather
than an overall winner. Replicated training and dependence-aware paired analysis are
needed before generalizing these findings.

## Reproduction and evidence
From the repository root:

    .\.venv\Scripts\python.exe -B docs/dongxiao_scrum36/analyze.py

The script verifies pinned source hashes, ID consistency, metric conventions, and ECE
arithmetic, then regenerates only its own results directory. It does not touch model
checkpoints, Gold arrays, existing team result files, or Git branches.
See source_manifest.json for all source URLs. Generated outputs:
comparison.csv, phase3_differences.csv, audit.json, accuracy.png and coverage.png.
