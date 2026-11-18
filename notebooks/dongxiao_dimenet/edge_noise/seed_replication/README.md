# Fine-tuning seed replication

Seeds 43 and 44 repeat the seed-42 pilot with unchanged Gold splits, model,
starting checkpoint, noise strength and training budget. This studies variation
in fine-tuning, not independently pretrained models. See `protocol.json`.

From the repository root, on `dongxiao`:

```powershell
.\.venv\Scripts\python.exe -B notebooks/dongxiao_dimenet/edge_noise/run_replication.py
```

This command preserves matching completed experiments and generates the verified
report after both replications finish. Do not launch a second copy while training
is already running. Partial/conflicting runs are preserved and require inspection;
there is no automatic resume of partial training.

To regenerate the report from completed runs, without training:

```powershell
.\.venv\Scripts\python.exe -B notebooks/dongxiao_dimenet/edge_noise/build_replication_report.py
```

Open `Replication_Report.html` for the presentation version, or read
`Replication_Report.md`. `replication_metrics.csv` and `replication_summary.json`
contain all results. Source code is included in the HTML for traceability.

Training progress is recorded in `seed43_training.log` and `seed44_training.log`.
Each log reports the arm, additional epoch and validation metrics. Full
checkpoints and MC arrays remain under Git-ignored `experiments/results/`.
No Gold files, historical results or shared model implementations are edited.

This study does not evaluate the test set, calibrate prediction intervals,
implement learned noise, or compare against BLIP. All seeds must be reported,
including any that contradict the original pilot.
