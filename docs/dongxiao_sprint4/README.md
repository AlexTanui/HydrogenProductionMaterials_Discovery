# Dongxiao Sprint 4 progress snapshot — work in progress

Snapshot date: 22 October 2026. These research notes are saved for team review;
they do not mark SCRUM-34, Phase 3 or the NLL experiment as complete.

## Included work

- `literature_review.md`: short comparison of edge dropout, node feature noise
  and edge feature noise, with DimeNet++ experimental lessons.
- `SCRUM34_demo_audit.md`: audit of the team implementation and UQ definitions.
- `verified_uq.json`: recomputed legacy metrics and checked historical BLIP record.
- `recheck_uq.py`: reproducible audit script; run `--help` for required paths.
- Earlier experiment code, executed notebook and reports are under
  `../../notebooks/dongxiao_dimenet/edge_noise/`.

## Open items

1. The team must confirm the Phase 3 candidate and metric reporting conventions.
2. The common backbone is Raj MPNN. DimeNet++ results are separate exploratory
   evidence; different theory levels and UQ aggregation rules prevent direct ranking.
3. At the audited team revisions (main/alex `e52e174`, fazin `195eec3`),
   `phase2_blip_nll` had a configuration but no verified run/checkpoint/predictions.
   Historical `phase2_blip` metrics are not NLL results.
4. Independent prediction alignment requires frame IDs and atom ordering in
   prediction exports. Counts and units alone are insufficient.
5. Team review and any direct message to Alex remain pending.

The audit describes the state when it was performed, including its original
checkout-access limitations and the fact that nothing had yet been pushed.
Those statements are historical context, not a live Git status report. This
snapshot is now being saved on `dongxiao`; it does not merge team main into that
branch. No datasets, model checkpoints or raw prediction arrays are included.

## Rechecking

Use a Python environment with NumPy, SciPy and PyTorch. Supply the integrated
team checkout via `--team`, the original local experiment checkout via `--legacy`,
and a new output filename via `--out`. The team checkout needs the shared UQ
module and historical BLIP test record. The legacy checkout needs its ignored
prediction arrays; a fresh Git clone alone does not contain those arrays.

```powershell
python docs/dongxiao_sprint4/recheck_uq.py --team PATH_TO_TEAM_CHECKOUT --legacy PATH_TO_LOCAL_EXPERIMENTS --out NEW_AUDIT_OUTPUT.json
```
