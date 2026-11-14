# Phase 3 vs Phase 1/2: statistical analysis (SCRUM-36, 9 Nov 2026)

Written for Dongxiao Wu (SCRUM-36 assignee) and the team, from the benchmark harness's own output — nothing here is re-derived by hand. Sources: [reports/all_phases.md](../../reports/all_phases.md), [reports/phase2_vs_phase3.md](../../reports/phase2_vs_phase3.md), `reports/calibration/*.json`, `reports/calibration/cqr/*.json`.

**Short answer.**
- **Phase 3 is less accurate than Phase 2 (BLIP-NLL)** — 17–20% higher energy and force MAE/RMSE — while using about half the parameters, training ~5x faster, and scoring ~2x faster at the same MC sample count.
- **Raw, uncorrected Phase 3 is far better calibrated** than raw BLIP-NLL: energy ECE is 4.2x lower, force ECE 2.5x lower. The representation-space mechanism's main payoff so far isn't accuracy — it's uncertainty quality straight out of training.
- **Once both go through the same post-hoc fix** (SCRUM-68 scalar rescale or the newer SCRUM-71 CQR), both land near nominal 90% coverage — but Phase 3's corrected intervals are consistently *wider* (less sharp, less informative) than BLIP-NLL's at that same coverage.
- **This is one run per phase, not re-scored.** SCRUM-38 (below) and SCRUM-40 are both still open, so treat every number here as a first read, not a settled result.

## 1. What's actually comparable

Per `phase2_vs_phase3.md`, `phase2_blip_nll` and `phase3_graph_stochastic` share identical `data:`/`train:` blocks, training-loop settings, train/val frames, and test scoring — same 55,509-frame ethanol (DFT) test split, same fingerprint (`518af6e3…`), same seed-0/batch-32 scoring. That part of the comparison is apples-to-apples.

**Not checked for equivalence** (model-specific hooks, per that report's own caveat):

| Hook | phase2_blip_nll | phase3_graph_stochastic |
|---|---|---|
| `kl_weight` | 0.01 | 0.01 |
| `train_num_mc_samples` (NLL) | 4 | 1 |
| `nll_warmup_epochs` | 20 | 0 |

`kl_weight` matches; the MC-sample-count and warmup asymmetry don't. That's a real asterisk on this analysis — SCRUM-40 owns deciding whether it needs reconciling before the comparison can be called fair.

## 2. Accuracy: Phase 3 underperforms consistently

| Metric | phase2_blip_nll | phase3_graph_stochastic | Phase 3 vs Phase 2 |
|---|---|---|---|
| Energy MAE (kcal/mol) | 0.0750 | 0.0897 | +19.6% |
| Energy RMSE | 0.0943 | 0.1126 | +19.4% |
| Force MAE (kcal/mol/Å) | 0.1787 | 0.2088 | +16.8% |
| Force RMSE | 0.2788 | 0.3264 | +17.1% |
| Parameters | 740,953 | 385,925 | 52% of Phase 2 |
| Training time | 3.98 h | 0.77 h | 5.2x faster |
| Inference (ms/frame, K=20) | 12.878 | 6.579 | 1.96x faster |

Both runs used their full 200-epoch budget and stopped on `max_epochs`, not early stopping — neither model's validation loss plateaued inside the patience-20 window.

## 3. Calibration before any correction: Phase 3 wins, and in the opposite direction

| Metric | phase2_blip_nll | phase3_graph_stochastic |
|---|---|---|
| Energy ECE | 0.6440 | 0.1520 |
| Energy coverage @ nominal 90% | 16.9% (undercovers) | 99.5% (overcovers) |
| Force ECE | 0.7379 | 0.2981 |
| Force coverage @ nominal 90% | 5.9% (undercovers) | 57.2% (overcovers) |
| Energy Spearman (uncertainty↔error) | −0.0050 | 0.0042 |
| Force Spearman | 0.2839 | 0.2626 |

BLIP-NLL is sharply **overconfident** (its intervals are too narrow, so true coverage falls far below nominal). Phase 3 is **underconfident** in the opposite direction — its intervals are too wide, overshooting nominal coverage. Both are miscalibrated, but Phase 3's error is 2.5–4x smaller in ECE terms. Energy-uncertainty Spearman is near zero for both — neither model's energy uncertainty tracks its own error at all; force uncertainty carries a real, comparable signal for both.

## 4. Calibration after correction: the gap narrows, but sharpness flips

**Scalar rescale (SCRUM-68):**

| | phase2_blip_nll after | phase3_graph_stochastic after |
|---|---|---|
| Energy ECE | 0.0174 | 0.0110 |
| Energy coverage @ 90% | 91.5% | 90.9% |
| Force ECE | 0.0447 | 0.0483 |
| Force coverage @ 90% | 91.9% | 92.0% |

**CQR (SCRUM-71), coverage @ 90% and mean interval width:**

| | phase2_blip_nll after | phase3_graph_stochastic after |
|---|---|---|
| Energy coverage | 91.5% | 90.6% |
| Energy mean width | 0.317 | 0.443 (+40%) |
| Force coverage | 91.7% | 91.9% |
| Force mean width | 0.837 | 0.964 (+15%) |

Both methods bring both checkpoints to within ~2 points of nominal 90% coverage — the post-hoc fix works for both phases. But at matched coverage, **Phase 3's intervals are 15–40% wider than BLIP-NLL's**, meaning they're less informative per unit of correctness. Spearman is unchanged before/after by construction (both corrections are monotone rescalings of the same per-frame uncertainty, so rank order — and therefore Spearman — can't move); that's expected, not a bug in either report.

Put together: Phase 3's raw uncertainty is better-shaped (lower ECE) but lower-resolution (needs a wider interval to actually hit the target coverage) than BLIP-NLL's. BLIP-NLL's raw overconfidence was hiding a sharper underlying signal once corrected.

## 5. Where Phase 3 does *not* beat Phase 2 (ROADMAP §4 requires this be stated plainly)

- **Raw accuracy**: 17–20% higher MAE/RMSE on both energy and force.
- **Sharpness after correction**: wider, less efficient intervals on both quantities, by either correction method.
- **Force uncertainty ranking**: Spearman 0.2626 vs 0.2839 — BLIP-NLL's force uncertainty tracks error slightly better, even though its absolute calibration (ECE) is worse.
- **Known unresolved gap (SCRUM-70, in review):** for the `_nll` checkpoint variant, post-hoc correction alone does not fully fix Phase 3's force miscalibration — a training-time fix is still needed. Not fully covered by the two base checkpoints analyzed above, but it's the same failure mode and should be read alongside this analysis.

## 6. Where Phase 3 wins

- Energy/force ECE before correction: 4.2x / 2.5x better than BLIP-NLL.
- Training time: 5.2x faster (0.77 h vs 3.98 h).
- Inference: 1.96x faster at identical K=20 MC samples.
- Parameters: roughly half (385,925 vs 740,953).

For RQ2/RQ3 (does moving stochasticity from weights into representations change the accuracy/uncertainty trade-off, not just match or beat BLIP on both axes): the answer so far is yes, and in a specific direction — Phase 3 trades raw accuracy and post-correction sharpness for cheaper compute and a better-shaped raw uncertainty signal. That's a legitimate, reportable finding, not a failed replication.

## 7. Caveats — what this analysis can't yet claim

- **Single run per phase, no seed variance.** There's no noise estimate (the optional repeat-seed study in `compute-budget-2026-11-01.md` hasn't run), so whether the 17–20% accuracy gap is real signal or run-to-run noise is not established.
- **"Not re-scored."** Both run records read `not re-scored` — SCRUM-38 (next, below) is what would confirm these exact numbers reproduce.
- **SCRUM-40 not formally closed.** The `train_num_mc_samples`/`nll_warmup_epochs` asymmetry in §1 hasn't been signed off as acceptable model-specific variation vs. a protocol gap.
- **SCRUM-70 open.** Post-hoc correction is a test-time patch, not a fix to why Phase 3's force uncertainty is shaped the way it is at training time.

## 8. Recommended next steps

1. Land SCRUM-38 (golden-run guard) before treating any number above as a reproducible result — in progress now, see below.
2. SCRUM-40: formally decide on the MC-sample/warmup asymmetry in §1.
3. If schedule allows: a 2-seed noise study on `phase2_blip_nll` and `phase3_graph_stochastic` specifically (~10 CPU-h per `compute-budget-2026-11-01.md`) would settle whether the accuracy gap in §2 survives noise.
