# Split conformal energy baseline — 29 October 2026

## Status
Implementation and eight tests pass. **No new molecular calibration result is claimed.**
The example plots and table are synthetic integration-test outputs, prominently labelled.
No existing model, Gold data, experiment result, or staged change was overwritten.
Work is maintained on dongxiao. Git history records publication; no shared branch merge is part of this experiment.

## Repository inspection
- Local ml/models/mpnn.py, ml/models/blip.py and ml/training/evaluate.py are empty placeholders.
- Latest remote main inspected: 46b5f99e107d2ee5c75634ce9da63bc4ea5dc155.
  It contains Raj's MPNN, registry/scaling wrapper, training and evaluation implementation.
- Raj's recorded pretrained model is experiments/checkpoints/phase1_baseline.pt.
  Recorded SHA256: 36a760377dbc61d26ad5a8897730f65c3170df5f2581ea10d36fcbbc05f55b4c.
  This checkpoint is missing locally; experiments/checkpoints does not exist.
  Aggregate run/test records cannot reconstruct calibration residuals or intervals.
- Training record: ethanol DFT, 10,000 stride-selected training frames,
  5,000 validation frames, checkpoint selected by validation loss.
  Full chosen indices are recorded remotely in experiments/runs/phase1_baseline/subsets.json.
- Fazin branch a1c2ee86b7c5315f9f8e0802f9d375b108180861 exports gold_index,
  E_ref, E_pred, optional E_var, and provenance metadata using --save-predictions.
  Main's evaluated source does not expose this export. Do not assume it is merged.
- Local Gold ethanol DFT: 555,092 frames, 9 atoms.
  train_idx 0..444073 (444,074); val_idx 444074..499582 (55,509);
  test_idx 499583..555091 (55,509). They exhaust the frames; no calibration split exists.
- Gold stores z/R/E/F, molecule/theory and saved split indices. It does not store
  original trajectory IDs, timestamps or an explicit source-frame map.
  Gold row indices are available; do not label them as validated physical timestamps.
  Silver deduplication and literature concatenation require provenance review before temporal claims.
- Existing pipeline uses contiguous 80/10/10 trajectory blocks, without an exclusion gap,
  or preserves shipped literature test sets and splits the remaining pool.
  MD22 Gold directory has no dataset locally. Do not mix molecules or theory levels.

## Independent calibration protocol
1. Freeze the checkpoint hash, inference code, energy scaling, molecule/theory,
   confidence levels, and final-test identities before inspecting new calibration errors.
2. Ask Raj for the exact checkpoint and all fit/selection/tuning IDs (including prior
   experiments relevant to this selected model), and Shijin for source trajectory provenance.
3. Prefer an independently reserved same-system/same-theory trajectory or a documented
   never-used pool. Promote any new source through the team's Gold pipeline.
4. Unselected frames within old train/validation blocks are only *candidates*, not approved
   calibration data. Require usage review and trajectory analysis. Do not automatically
   split the old validation set, borrow final-test frames, or regenerate Gold splits.
5. Save explicit calibration IDs and frozen final-test IDs in the experiment manifest.
   Predeclare any temporal gaps/thinning from physical knowledge or training-only diagnostics.
   Contiguous blocking and gaps reduce dependence but do not establish exchangeability.
6. The historical final test has already been evaluated. It can support a clearly labelled
   retrospective comparison; it cannot be described as newly untouched. A strict new final
   assessment requires a reserved evaluation pool unused in method selection.
7. If no unused calibration pool exists, the reuse-checkpoint experiment is blocked.
   A newly reserved split with retraining is an alternative requiring a new agreed protocol,
   not something this implementation silently does.

## Method
For calibration residuals s_i = abs(y_i - f(x_i)), n calibration frames, target c:
k = ceil((n + 1) c). Use the k-th smallest residual, q; use +infinity when k > n.
Intervals are [f(x)-q, f(x)+q], at c = 0.80, 0.90, 0.95.
There is no interpolation and no additional Gaussian multiplier.
MAE/RMSE use unchanged point predictions. Coverage includes interval boundaries;
average width is 2q. The guarantee is marginal for a new exchangeable observation,
not conditional coverage at every geometry, simultaneous trajectory coverage, or
a guarantee that this particular test set reaches its nominal fraction.

## Run
From the repository root, using the existing virtual environment:

    .\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_split_conformal.py -v
    .\.venv\Scripts\python.exe -B -m ml.training.evaluate_split_conformal --manifest path/to/manifest.json --output path/to/new_run

The runner uses NumPy and Matplotlib. It accepts frozen prediction NPZs with:
- gold_index: unique nonnegative integer IDs in one canonical Gold namespace
- E_ref, E_pred: finite 1-D absolute TOTAL energy arrays (kcal/mol)
- Optional E_var is intentionally not used by this baseline.

Required manifest fields:
- calibration, test: paths to the respective NPZs, relative to the manifest
- calibration_sha256, test_sha256: file hashes; checked against input bytes
- dataset_sha256, checkpoint_sha256: audited source identities
- model_identity, energy_unit="kcal/mol", energy_kind="total_absolute"
- fit_selection_tuning_ids: complete forbidden IDs for this frozen predictor
- final_test_ids: the complete preregistered test ID list
- calibration_unused_for_fit_selection_tuning=true
- test_unused_for_method_selection=true
- independence_evidence, trajectory_limitations: nonempty review statements
- illustrative=true ONLY for artificial fixtures; omit for observed predictions

The example manifest under illustrative_demo_inputs shows the schema, but its fake model
hashes identify a test fixture. NEVER copy its attestations into a real experiment.
Dataset/checkpoint hashes are recorded attestations; this export-only runner cannot prove
their relationship to the prediction bytes. Verify them in the generating inference job.
The runner checks export hashes, ID uniqueness/overlap, full final-test IDs, shape, finiteness,
units and attestations. It refuses existing output directories. It does not infer provenance.
For multiple source trajectories, assign a collision-free canonical Gold namespace and
retain an external ID-to-trajectory/source-frame map; integer IDs alone do not ensure independence.

Outputs: comparison.csv, results.json, energy_intervals.png, coverage.png.
Plots display up to 150 frames sampled deterministically by ID; all test rows contribute metrics.
Infinite intervals remain infinite in CSV and are written as "Infinity" in JSON.

## BLIP feasibility and existing implementation review
The shared BLIP predicts per-frame energy means and variances from MC samples. A future
normalized score can be abs(y-mu)/max(sqrt(variance), epsilon), yielding mu +/- q*scale(x).
Freeze MC protocol and epsilon before calibration, verify finite nonnegative variance and
aligned IDs, and use the SAME approved calibration/test pools for fixed/adaptive comparisons.
MC uncertainty need not be a calibrated Gaussian for this score construction. Poorly informative
scales may produce inefficient intervals. Adaptive implementation is deferred until the real
MPNN baseline and data independence have been verified.

Remote ml/training/calibrate.py already implements an exploratory stochastic calibration path.
It is preserved, but does not satisfy this requested experiment:
- It reuses validation data (potential checkpoint-selection overlap).
- It excludes deterministic models without predict_distribution.
- It uses np.quantile(..., method="higher") with a corrected probability, which is not
  generally the exact ceil((n+1)c) order statistic; it also clips small-sample levels to 1.
- Its residual/std quantile is then multiplied into std and passed to Normal intervals.
  At 90%, this produces an additional Normal z multiplier, rather than mu +/- q*std.
  If the interface requires a rescaled Normal std, the scale would need q/z for that
  particular target; independently calibrating every target is clearer.
These are code-review observations, not re-evaluated numerical results. No team file was changed.

## Paper connections
Kuleshov, Fenner and Ermon (2018), Accurate Uncertainties for Deep Learning Using
Calibrated Regression, ICML 80:2796–2804:
https://proceedings.mlr.press/v80/kuleshov18a.html
Their post-hoc regression calibration motivates checking observed probability against
claimed probability, and considering sharpness alongside calibration. Their CDF/isotonic
recalibration is distinct from this finite-sample split conformal implementation.

Bao et al. (2025), A Review and Comparative Analysis of Univariate Conformal Regression
Methods, PMLR 266:282–304:
https://proceedings.mlr.press/v266/bao25a.html
Section 3's absolute-residual split baseline motivates our first experiment. We use
the exact finite-sample order-statistic construction and defer more elaborate methods.
Both primary PDFs were consulted; no claim is made that either paper establishes
exchangeability for this MD trajectory dataset.

## Client meeting summary
We implemented and tested a model-independent split conformal energy evaluation pipeline
for the pretrained MPNN, supporting 80%, 90% and 95% intervals. It includes finite-sample
quantiles, leakage checks, coverage/width tables and diagnostic plots. The repository audit
found that the current validation split was used for model selection and is not automatically
an independent calibration set. Real evaluation awaits Raj's exact checkpoint (or audited
aligned predictions) and a documented unused calibration/evaluation protocol. No improvement
or coverage guarantee on molecular data is claimed yet.
