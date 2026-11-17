# Dongxiao: fixed edge-noise pilot

This directory contains an English report, executed analysis notebook, charts and
compact metrics. It is an independent experiment on the existing DimeNet++ adapter,
not a replacement for the team's shared MPNN/BLIP/graph-stochastic implementation.

## Open the results

- `Edge_Noise_Report.html`: self-contained report with embedded images.
- `Edge_Noise_Lab.ipynb`: code and actual analysis outputs, reading the saved run.
- `metrics.json`: compact configuration, provenance, histories and evaluation.
- `../../../docs/model_cards/dongxiao_edge_noise_20.md`: both selected model cards.

Training has already been run. Opening the HTML does not start training. For the
notebook, select the project's `.venv` interpreter. The analysis reads local sample
arrays from the ignored result folder, so a clone without those files can view the
saved outputs but cannot regenerate every chart without rerunning the experiment.

## Commands (PowerShell, repository root)

```powershell
cd C:\Users\64512\HydrogenProductionMaterials_Discovery
git branch --show-current
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p "test_dongxiao*.py" -v
.\.venv\Scripts\python.exe -B notebooks/dongxiao_dimenet/edge_noise/build_report.py
```

The branch must be `dongxiao`. To reproduce training separately:

```powershell
.\.venv\Scripts\python.exe -B -m ml.training.dongxiao_edge_experiment --config experiments/configs/dongxiao_edge_noise_20.yaml
```

The training runner refuses to overwrite an existing output directory. For a new
run, copy the configuration and choose a new `train.output_dir` under
`experiments/results`. Preserve the original run. The report builder currently
targets `dongxiao_edge_noise_20`; changing report input/output paths is necessary
to make a separate report for a different run.

## What was run

Both arms start from the same historical 200-epoch checkpoint, with fresh Adam
states and the same training batch order. Each receives 20 epochs on 889 saved
Gold training structures, with 111 saved validation structures. The only model
intervention is a fixed 0.05 standard deviation Gaussian multiplier on initial
directed-edge embeddings; sigma zero is the control. Checkpoint selection uses
validation joint MSE of MC means (8 samples for noise); final diagnostics use a
separate stream of 32 samples. No new test evaluation or energy offset correction.

Weights and prediction arrays remain in the ignored directory
`experiments/results/dongxiao_edge_noise_20`. Do not add those files or Gold data
to Git. No commit or push is part of this experiment.

## Continue from here

Current completed result: deterministic selected epoch 18, raw validation energy
MAE 1.128 kcal/mol and force MAE 1.131 kcal/mol/Angstrom; edge-noise selected epoch
19, energy MAE 0.460 and force MAE 1.270 in the same units. Force Spearman is 0.347.
This is mixed evidence, not an overall improvement. Raw energy intervals are too
wide on this validation sample (nominal 50% coverage, observed 92.8%).

Verification completed: 4 stochastic physics tests, 2 analytic UQ tests and the
3 existing deterministic DimeNet tests passed. Gold checksum was verified before
and after training. No existing tracked source was modified; all additions are
separate Dongxiao files. Nothing has been committed or pushed.

1. Read the report's findings, including any accuracy loss or poor calibration.
2. Agree with the team on the shared-backbone interface and ownership before
   interpreting this as a project-wide comparison.
3. Replicate with additional training seeds or a predeclared validation-only
   noise-strength study. Preserve every result, including negative outcomes.
4. Learned noise and formal predictive calibration remain future work. Do not
   repeatedly inspect or tune against the historical test set.

## If interrupted

`experiments/results/dongxiao_edge_noise_20/metrics.json` records status and the last
completed epoch for each arm. `*_best.pt` includes optimizer and RNG states for the
selected epoch, but this runner does not implement automatic resume. A partial run
must be preserved; use a new output directory for a restart. A report is generated
only when both arms are complete.
