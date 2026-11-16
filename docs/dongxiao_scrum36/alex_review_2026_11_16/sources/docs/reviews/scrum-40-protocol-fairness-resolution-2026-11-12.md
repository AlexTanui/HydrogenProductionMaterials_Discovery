# SCRUM-40 resolution: the MC-sample/warmup asymmetry needed reconciling, and a protocol-matched comparison already existed

**Decision.** The asymmetry flagged in `docs/reviews/scrum-36-phase3-vs-phase1-2-analysis-2026-11-09.md` §1 (`phase2_blip_nll` trains with `train_num_mc_samples=4`/`nll_warmup_epochs=20`; `phase3_graph_stochastic` trains with `1`/`0`) is **not acceptable model-specific variation** — it changes the result materially (below) and is not forced by anything intrinsic to where stochasticity lives. It needed reconciling, and a reconciled pair already exists: `phase3_graph_stochastic_nll` (SCRUM-70, trained for a different reason — fixing Phase 3's force miscalibration at the source — but it happens to use the exact same hooks as `phase2_blip_nll`: `train_num_mc_samples=4`, `nll_warmup_epochs=20`). The harness's own fairness check (`ml/training/benchmark.py`'s `check_same_protocol`) confirms this pair matches on every shared-protocol field; the only resolved pair is `phase2_blip_nll` vs `phase3_graph_stochastic_nll`, generated at
[reports/phase2_vs_phase3_nll.md](../../reports/phase2_vs_phase3_nll.md) (`python -m ml.training.benchmark --runs phase2_blip_nll phase3_graph_stochastic_nll --title "Phase 2 (BLIP, NLL) vs Phase 3 (graph-space, NLL) - protocol-matched" --out reports/phase2_vs_phase3_nll`).

**Going forward, the official Phase 2-vs-Phase 3 headline comparison should use this matched pair, not `phase2_blip_nll` vs `phase3_graph_stochastic` (base).** The mismatched-hook comparison in SCRUM-36 is still valid as a Phase-3-base-vs-Phase-2 result, but it is not the fair, protocol-equal comparison the project needs for RQ2/RQ3 — this one is.

## What changes once the hooks are matched

| Metric | phase2_blip_nll | phase3_graph_stochastic (base, mismatched hooks) | phase3_graph_stochastic_nll (matched hooks) |
|---|---|---|---|
| train_num_mc_samples / nll_warmup_epochs | 4 / 20 | 1 / 0 | **4 / 20** |
| Energy MAE / RMSE | 0.0750 / 0.0943 | 0.0897 / 0.1126 (+19.6% / +19.4%) | **0.1292 / 0.1612 (+72.3% / +70.9%)** |
| Force MAE / RMSE | 0.1787 / 0.2788 | 0.2088 / 0.3264 (+16.8% / +17.1%) | **0.1846 / 0.2919 (+3.3% / +4.7%)** |
| Parameters | 740,953 | 385,925 (52%) | 385,925 (52%) |
| Training time | 3.98 h | 0.77 h (5.2x faster) | 2.29 h (1.74x faster) |
| Inference (ms/frame, K=20) | 12.878 | 6.579 (1.96x faster) | 6.253 (2.06x faster) |
| ECE energy / force (raw) | 0.647 / 0.738 | 0.152 / 0.298 (4.2x / 2.5x better) | **0.201 / 0.0277 (3.2x / 26.6x better)** |
| Spearman energy / force | 0.019 / 0.284 | 0.0042 / 0.2626 | 0.0013 / 0.2948 |

**This is not just a number shift — it changes the shape of the finding:**

- **Force gets dramatically better on every axis once hooks are matched.** Force MAE gap narrows from +16.8% to +3.3% — nearly closed. Force ECE improves another order of magnitude, from already-2.5x-better to **26.6x better** than BLIP-NLL. Force uncertainty ranking flips from slightly worse (0.2626 vs 0.2839) to slightly *better* (0.2948 vs 0.2836) than BLIP-NLL. This is exactly what SCRUM-70 set out to fix (force miscalibration at the source, not post-hoc), and the matched-hook run confirms it worked.
- **Energy gets dramatically worse.** The energy MAE gap more than triples, from +19.6% to **+72.3%**. This is a new finding the SCRUM-36 analysis did not have, because it compared against the mismatched-hook base run. The mismatch wasn't just "unfair" in the abstract — it was hiding a real, large energy regression that the matched-hook training schedule introduces.
- **Training/inference-speed advantage shrinks** (because 4 MC training samples cost ~4x what 1 does): 5.2x faster training becomes 1.74x; inference speed is essentially unchanged (still ~2x, since eval-time MC count was always 20 for both).

**Reading:** matching the protocol did not just make a mismatched comparison fair — it revealed that the NLL-with-warmup training recipe, as currently implemented in `graph_stochastic.py`, trades energy accuracy for force calibration far more sharply than it does in `blip.py`. That asymmetric effect (one quantity gets much better, the other much worse, under the same training-recipe change) is itself worth investigating — it suggests the recipe interacts differently with how Phase 3 structures its representation-space noise than with BLIP's weight-space noise, not just a straightforward "more MC samples and warmup helps."

## What this means for existing documents

- `docs/reviews/scrum-36-phase3-vs-phase1-2-analysis-2026-11-09.md` compared `phase2_blip_nll` against `phase3_graph_stochastic` (base, mismatched hooks). That comparison stands as what it is — a real, reported result — but is no longer the fair headline comparison per this decision. A follow-up SCRUM-36 addendum comparing against `phase3_graph_stochastic_nll` would be the correct next version of that document.
- `docs/technical_report.md` §4 is being updated in the same session to lead with the matched-pair numbers above rather than the mismatched pair.

## Caveats (same as before, still unresolved)

- Still one run per phase per variant, not re-scored (SCRUM-38/71).
- The energy regression under matched hooks is new and unexplained — worth its own investigation, not just noted here.
- `phase2_blip` (MSE variant, non-NLL) is still excluded from any official table pending its clean retrain (SCRUM-69).
