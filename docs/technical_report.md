# Technical report — Adaptive Graph-Space Uncertainty Modelling for MD17

Status: skeleton + Phase 1-3 findings, last extended 17 Nov 2026 (SCRUM-26 /
SCRUM-36 merge). Base outline and §1-3/§6-10 text are Alex/Fazin's; §4-5 fold
in Dongxiao's 16 Nov methodological review (`docs/dongxiao_scrum36/`), which
is more careful about what the numbers do and don't prove and is treated as
authoritative where the two drafts disagreed. Each section still points at
the source document it should be written from rather than duplicating
content that would then need to be kept in sync with the reports it came
from.

## 1. Problem and scope

Three-phase plan on MD17 (ethanol, DFT): deterministic MPNN baseline (Phase
1) -> weight-space stochasticity, BLIP-style (Phase 2) -> representation-
space stochasticity (Phase 3). See `CLAUDE.md` and `ROADMAP.md` §1 for the
full framing and why MD17 rather than real catalyst screening.

*TODO: one paragraph restating this for a reader outside the team.*

## 2. Method

*TODO (Dongxiao/Ruturaj):* architecture summary per phase, referencing
`ml/models/mpnn.py`, `ml/models/blip.py`, `ml/models/graph_stochastic.py`.
Shared training/eval protocol: `docs/phase1_protocol.md`.

## 3. Phase 1 results

Final bake-off table: `reports/phase1_benchmark.md`. Model cards:
`docs/model_cards/`. Full numbers: `docs/reviews/sprint4-to-present-submission-2026-11-08.pdf`
("Phase 1 Final Bake-off Results").

All five architectures were retrained under one shared protocol
(`ml/training/train.py`, fixed data/splits/optimiser/epoch budget) so the
comparison is apples-to-apples, closing the gap the Sprint 3 retrospective
had flagged. On raw accuracy, **DimeNet++ and PaiNN won**, not MPNN:
DimeNet++ has the lowest energy error (MAE 0.0537, RMSE 0.0688) and PaiNN
the lowest force error (MAE 0.1117, RMSE 0.1858); MPNN is mid-pack (energy
MAE 0.0779, force MAE 0.1775) and SchNet is clearly worst on every
accuracy metric despite being fastest to train (0.29 h, smallest model at
61,569 parameters).

**MPNN was nonetheless the architecture chosen to carry into Phase 2/3**
(`ml/models/blip.py`, `ml/models/graph_stochastic.py` both extend
`ml/models/mpnn.py` — see §1's three-phase table). That pick was not based
on winning the bake-off; it wasn't the most accurate model on either
metric. The stated rationale (ROADMAP.md, team decision) is architectural
suitability for the two kinds of stochasticity the plan requires: a
Gilmer-style message-passing backbone gives a clean, well-understood place
to inject Gaussian weight noise (Phase 2) and later representation noise
(Phase 3) without redesigning the base model twice. Stating this plainly
here because the alternative reading — "the team picked the best model" —
isn't supported by the table above.

## 4. Phase 2 vs Phase 3 — accuracy and uncertainty

Full analyses: `docs/reviews/scrum-36-phase3-vs-phase1-2-analysis-2026-11-09.md`
(Alex's first pass) and `docs/dongxiao_scrum36/SCRUM36_Report.md` plus
`docs/dongxiao_scrum36/alex_review_2026_11_16/Review.md` (Dongxiao's 16 Nov
review of that pass). **Protocol fairness resolved**: SCRUM-40 found the two
runs Alex's original analysis compared were not hook-matched
(`train_num_mc_samples`/`nll_warmup_epochs` differed); the fair,
protocol-equal comparison is `docs/reviews/scrum-40-protocol-fairness-resolution-2026-11-12.md`
and `reports/phase2_vs_phase3_nll.md`. Dongxiao's review (16 Nov) re-audited
that matched-protocol comparison against the upstream re-score records and
is the version reported below; it corrects wording and emphasis rather than
the underlying numbers, which it confirms independently.

**Point prediction (matched-protocol, lower is better):**

| Model | Energy MAE | Energy RMSE | Force MAE | Force RMSE |
|---|---|---|---|---|
| MPNN | 0.0779 | 0.0962 | 0.1775 | 0.2774 |
| BLIP MSE | 0.0920 | 0.1120 | 0.1826 | 0.2828 |
| BLIP NLL | 0.0750 | 0.0943 | 0.1787 | 0.2788 |
| Graph MSE | 0.0897 | 0.1126 | 0.2088 | 0.3264 |
| Graph NLL | 0.1292 | 0.1612 | 0.1846 | 0.2919 |

BLIP NLL has the lowest recorded energy MAE/RMSE; the deterministic MPNN has
the lowest recorded force MAE/RMSE. **Phase 3 does not beat BLIP on point
accuracy in any matched pairing**: Graph NLL vs BLIP NLL is +72.3% energy
MAE / +3.3% force MAE; Graph MSE vs BLIP MSE is +14.35% force MAE despite a
2.45% lower energy MAE (and its energy RMSE is actually 0.52% *higher*, so
even that one bright spot doesn't hold under RMSE); both Phase 3 variants
have higher energy and force error than deterministic MPNN. Exact deltas:
`docs/dongxiao_scrum36/results/phase3_differences.csv`. These are single-run
percentage differences with no significance assessment attached — read as
descriptive, not as a tested claim that one model is reliably better.

**Raw (pre-post-hoc) uncertainty:**

| Model | Energy ECE | Force ECE | Energy Spearman | Force Spearman | Energy cov@90% | Force cov@90% |
|---|---|---|---|---|---|---|
| BLIP MSE | 0.5964 | 0.6715 | -0.0025 | 0.2641 | 22.63% | 13.73% |
| BLIP NLL | 0.6470 | 0.7379 | 0.0190 | 0.2836 | 16.62% | 5.86% |
| Graph MSE | 0.1538 | 0.2969 | 0.0021 | 0.2622 | 99.46% | 57.37% |
| Graph NLL | 0.2007 | 0.0277 | 0.0013 | 0.2948 | 99.99% | 86.31% |

**Phase 3's real win is force calibration, not accuracy.** Once matched on
MC samples/warmup: Graph NLL's force ECE is 26.6x better than BLIP NLL
(0.0277 vs 0.7379) and its force coverage@90% (86.31%) is far closer to
nominal than BLIP NLL's badly-undercovered 5.86% — BLIP's NLL training
actually makes its calibration *worse* than its own MSE run, consistent
with Fazin's 31 Oct variance-floor finding (comment 10117 on SCRUM-70: the
training-time variance floor dominates BLIP's predicted spread, 610x the
squared energy error at its worst). Graph NLL's force Spearman (0.2948)
also edges out BLIP NLL (0.2836), though the gap is small enough that one
run per variant doesn't support calling it a reliable ranking advantage.

**Energy calibration is the opposite story — both Phase 3 variants are
badly overcovered, not well-calibrated**, and NLL training makes Graph's
energy *worse* on both axes versus Graph MSE (ECE 0.2007 vs 0.1538, mean
predicted std 0.4969 vs 0.2143) while also carrying the worst energy MAE of
any model in the table. Energy uncertainty-error Spearman is within noise
of zero for every stochastic variant (BLIP and Graph alike) — none of the
five models' energy uncertainty actually tracks error; only force
uncertainty does, and only modestly (Spearman ~0.26-0.29 across BLIP/Graph
NLL variants).

**Read together, this is an asymmetric trade, not a win or a loss**: moving
stochasticity into representations instead of weights (Phase 3 vs Phase 2)
buys a large, real force-calibration improvement at the cost of energy
accuracy and energy calibration both getting worse, under the same
training-recipe change. One run per phase per variant (not yet re-scored
per SCRUM-38/71 beyond the independent-rescore reproducibility check in
§7) — a first read supported by a careful audit, not a settled result.
Training/inference cost advantage for Phase 3 shrinks once MC samples are
matched: 1.74x faster training (was reported as 5.2x before matching),
~2.06x faster inference. Parameter count is unchanged (385,925 vs 740,953,
52% smaller).

**SCRUM-70 (Phase 3 force miscalibration, training-time fix):** the above
NLL numbers are "fix 1" (training-time Gaussian NLL instead of MSE, no
change to where noise enters the model). A second, structural fix
(multiplicative noise on the pre-aggregation message instead of additive
noise on the post-aggregation node state — see
`docs/reviews/scrum-70-structural-root-cause-2026-11-12.md` and commit
`633d1f2` on branch `alex`, not yet merged or reviewed by Ruturaj) pushes
force coverage@90% to 91.61% but makes force ECE *worse* (0.0277 → 0.0424,
under-coverage traded for over-coverage) and both energy and force MAE
regress further. Dongxiao's review flags the open methodological gap this
fix doesn't close on its own: the zero first derivative of additive noise
"does not prove the remaining pathway weak, second-order, or an inevitable
energy/force trade-off" — i.e. the root-cause argument is a plausible
hypothesis consistent with the observed numbers, not a proof, and the
multiplicative fix's own mixed result is itself evidence this needs a
controlled test rather than being taken as closed. Tracked as a separate,
clearly-labelled experiment per SCRUM-70's resolution, not a replacement
for the Graph NLL numbers above.

*Base analysis is Fazin/Alex's; methodological review and sign-off is
Dongxiao's (SCRUM-36, 16 Nov) and is incorporated above.*

## 5. Calibration methodology

Scalar (Kuleshov 2018) and CQR (Bao 2025) post-hoc corrections:
`docs/reviews/cqr-calibration-report-2026-11-05.pdf`,
`docs/reviews/cqr-conformal-calibration-2026-11-04.md`,
`docs/reviews/scrum-68-calibration-review-2026-11-01.md`.

**Post-hoc results** (separate 5,000-frame evaluation split, calibrated
against the Gold validation split; all values are observed marginal
coverage, not a finite-sample guarantee):

| Method | Model | Energy cov@90% | Energy width | Force cov@90% | Force width |
|---|---|---:|---:|---:|---:|
| CQR | BLIP NLL | 91.54% | 0.3170 | 91.74% | 0.8373 |
| CQR | Graph MSE | 90.58% | 0.4433 | 91.87% | 0.9641 |
| CQR | Graph NLL | 90.30% | 0.7411 | 91.66% | 0.8390 |

| Method | Model | Energy cov@90% | Energy ECE | Force cov@90% | Force ECE |
|---|---|---:|---:|---:|---:|
| Scalar | BLIP NLL | 91.54% | 0.0174 | 91.89% | 0.0447 |
| Scalar | Graph MSE | 90.94% | 0.0110 | 92.00% | 0.0483 |
| Scalar | Graph NLL | 91.08% | 0.0092 | 91.62% | 0.0415 |

Post-hoc correction gets every model close to nominal 90% coverage on both
quantities — expected, since that's what it's designed to do — but **it is
not a substitute for fixing raw calibration** (SCRUM-70's point): Graph
NLL's scalar-corrected force ECE is actually a bit *worse* than its raw
force ECE (0.0415 vs 0.0277, on a different 5,000-frame subset), a reminder
that "moves coverage toward nominal" and "lowers average error across all
levels" are different properties and don't always move together. Open
item: Phase 3 energy calibration diverges 13-33% between the scalar and CQR
methods depending on configuration, unresolved (see §9).

**Methodological caveats carried over from Dongxiao's 16 Nov review** (full
list: `docs/dongxiao_scrum36/SCRUM36_Report.md` §"Calibration validity and
statistical limits"), kept here because they bound what any calibration
number in this report can be read as proving:

1. Calibration uses the whole validation split, which training also uses
   for selection/scheduling — not a calibration set independent of those
   decisions. These reports are not a validated finite-sample coverage
   guarantee on that basis alone.
2. Calibration JSONs don't export exact frame IDs or full invocation
   provenance; frame-level overlap and trajectory-boundary checks between
   train/val/test remain to be done explicitly rather than assumed from
   disjoint ID ranges.
3. MD frames (and pooled force components within one structure) are
   correlated, not independent observations — significance testing on the
   pooled 1,498,743 force components as if independent is invalid.
4. `calibrate.py`'s corrected-quantile implementation (NumPy
   `method="higher"`, clamped out-of-range levels) is not the textbook
   exact order-statistic correction
   (`ceil((n+1)*(1-alpha))` with an infinite interval when the rank exceeds
   `n`). Worth auditing against the metric owner (Fazin) before citing this
   as "the standard" CQR correction in the final report.
5. Each variant has exactly one training run; re-scoring or changing the
   inference seed is not an independent training replication, so no
   confidence interval or p-value is justified from these tables alone.
6. SCRUM-49 found no tested MC sample count `K` meeting its own
   three-inference-seed stability bar for BLIP NLL or Graph MSE — it didn't
   test Graph NLL specifically, and a negative stability result for two
   variants doesn't prove no `K` would work; small inter-variant ranking
   differences (e.g. the Graph NLL vs BLIP NLL Spearman gap above) should
   not be read as an established, stable advantage on that basis.

## 6. Validation: leakage audit and compute budget

`docs/reviews/leakage-audit-2026-10-31.md`,
`docs/reviews/compute-budget-2026-11-01.md`. Open item: SCRUM-72 (purge
buffer + split provenance, not yet applied).

## 7. Reproducibility

Golden-run regression guard: `tests/ml/test_golden_run.py` (SCRUM-38).
**Independent re-score (SCRUM-71, 14 Nov):** every real checkpoint with a
local `.pt` file was independently re-scored and reproduces its recorded
accuracy/benchmark metrics to within 7.748e-10 relative (BLIP NLL),
1.374e-10 (Graph MSE) and 4.555e-11 (Graph NLL) — see each run's `rescore`
block and `reports/all_phases.md`. This confirms the *recorded* numbers are
reproducible from the *same* checkpoint; it says nothing about whether
retraining from scratch with the same config reproduces them (that would
need a repeated-seed study — see §5 caveat 5 and §9's open items), and it
is a different question from the training-data provenance questions in §6.

Dongxiao's 16 Nov review independently reran this reproducibility check
(`docs/dongxiao_scrum36/alex_review_2026_11_16/verify_review.py`) against
pinned source hashes and confirmed the same tolerances; it also flagged
that some generated Markdown in the repo still said "not re-scored" after
SCRUM-71 landed and was stale at the time of her review — worth a final
pass before submission to make sure every generated report reflects the
current re-score status.

## 8. Platform

Backend/frontend wiring: SCRUM-42/43 (done). Three-phase dashboard
comparison view: SCRUM-44 (done, SCRUM-71 merge). **Integration/demo
readiness audit (SCRUM-46, Dongxiao, 15 Nov):** as of that audit, the
checkout under review was an early interface scaffold whose prediction
page served hash-generated placeholder outputs rather than real inference,
with the frontend dependencies and runtime benchmark JSONs missing from
that particular environment — a live end-to-end rehearsal had not been
completed from that checkout. Per CLAUDE.md's current-state note, `backend/`
and `frontend/` are wired to real checkpoints as of SCRUM-42/43, so this
reads as a stale/incomplete local setup at audit time rather than a defect
in the shipped code — but it hadn't been re-verified against a from-scratch
checkout as of this report, and the default live Phase 3 route serves
Graph MSE, not Graph NLL, which matters for correctly labelling any
dashboard numbers against this section's tables.

## 9. What worked, what didn't, what's next

Full write-up: `docs/reviews/sprint4-to-present-submission-2026-11-08.pdf`
(covers 22 Sep - 8 Nov). Summary below condenses it; see the PDF for the
supporting tables and figures.

**What worked.** Freezing interfaces early prevented collisions twice —
the Sprint 1 architecture freeze let five people build five Phase 1
models in parallel, and the Sprint 4 shared harness (`ml/training/train.py`,
`ml/config.py`, the model registry) repeated that for Phase 2/3, letting
BLIP and graph_stochastic plug into one registry and one scored protocol.
Three substantive bugs (BLIP batch-dependence, BLIP-NLL variance collapse,
the SCRUM-68 calibration 1.645x unit slip) were caught by review and fixed
before being reported, not after. Findings were tested rather than
assumed: the leakage audit, compute-budget study, and atom-type
stratification were all run to check a plausible explanation rather than
confirm an assumed one — the atom-type check in particular refuted its
own starting hypothesis and that refutation was reported honestly. A
second, distribution-free calibration method (CQR) was run against the
first (scalar rescale) specifically to stress-test its assumptions, and
surfaced a real disagreement on Phase 3 energy that a single method would
have hidden. Independent cross-review caught real issues before they
reached the report: Dongxiao's 16 Nov pass on the SCRUM-36 analysis
corrected several claims that had outrun what the underlying single-run
data could support (precise wording above now reflects that review).

**What failed, or is still a gap.** Process tracking lags the engineering
work: `docs/sprint_reports.md` stops at Sprint 4 (1 Oct) with five weeks
of Sprint 5/6/overrun work never written up, and per-sprint retrospective
files were never filled in beyond the template. Commit authorship and
Jira assignment have drifted apart — tickets assigned to Ruturaj in Jira
(SCRUM-31/32/67, BLIP and Phase 3) show their actual commits authored by
Alex Tanui. Real work shipped with no matching Jira ticket (SCRUM-71's
CQR/atom-type work, 5 commits, 4 Nov — see the naming collision noted
against the newer SCRUM-71 filed in this session). Meeting minutes aren't
retrievable from the repo or Jira. On the technical side: `phase2_blip`
still needs retraining from a clean commit (SCRUM-69/48) before it can
re-enter the official comparison table; the leakage audit's random-split
control shows a 17-26% accuracy gap in leakage's direction, not yet
attributed to leakage vs. noise; Phase 3 energy calibration still diverges
13-33% between the scalar and CQR methods, unresolved; and — per §5's
caveats — no variant has a repeated-seed training replication, so none of
the cross-model accuracy or calibration deltas in §4-5 carry a formal
significance claim.

**What's next**, in priority order: retrain `phase2_blip` clean (SCRUM-69)
and re-run its calibration; get the MC-stability study run against Graph
NLL specifically (SCRUM-49 covered BLIP NLL and Graph MSE, not Graph NLL);
resolve whether SCRUM-70's structural fix (§4) should replace, sit beside,
or be re-tuned relative to the NLL-only fix, pending Ruturaj's review;
isolate Phase 3 energy's remaining scalar-vs-CQR width gap; split force's
heavy-tailed hydrogen residual by hydroxyl-vs-methyl H now that atom-type
alone didn't explain it; obtain frame-aligned paired exports (Gold ID,
trajectory ID, E/F reference and prediction, E_var/F_var) for all five
models so that paired, dependence-aware significance testing (block
resampling over trajectories, not naive per-component bootstrapping) is
possible at all — currently it isn't, from the aggregated metrics alone;
and decide whether Phase 1's numbers need a run-to-run noise estimate
before being treated as final. Process-side: file SCRUM-71 retroactively
against its already-shipped CQR work (distinct from the new, same-numbered
reproducibility ticket filed this session — flagged as an open naming
collision), reconcile Jira assignee vs. git-commit author on SCRUM-31/32/67,
and write up the Sprint 5/6/overrun retrospectives that are still missing.

## 10. Conclusion

*TODO.* Draft once §2 (method) and the Ruturaj/Dongxiao sign-offs on §4's
open items (SCRUM-70 mechanism review, MC-stability for Graph NLL) land —
the honest headline per the evidence gathered so far is that Phase 3 is a
real, substantial win on raw force calibration and a real loss on energy
accuracy and energy calibration relative to Phase 2, not a strict
improvement in either direction, and every cross-model number in §4-5 is a
single-run descriptive comparison rather than a statistically tested one
(§5's caveats, §9's "what's next").
