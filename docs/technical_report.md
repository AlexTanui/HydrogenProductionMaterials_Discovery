# Technical report — Adaptive Graph-Space Uncertainty Modelling for MD17

Status: skeleton only, started 9 Nov 2026 (SCRUM-26). Each section currently
points at the source document it should be written from rather than
duplicating that content — fill in-place, don't restate numbers here that
would then need to be kept in sync with the reports they came from.

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

Full analysis: `docs/reviews/scrum-36-phase3-vs-phase1-2-analysis-2026-11-09.md`
(SCRUM-36). **Protocol fairness resolved**: SCRUM-40 found the two runs
that analysis compared were not hook-matched (`train_num_mc_samples`/
`nll_warmup_epochs` differed); the actual fair, protocol-equal comparison
is `docs/reviews/scrum-40-protocol-fairness-resolution-2026-11-12.md` and
its table, `reports/phase2_vs_phase3_nll.md`. The numbers below are the
matched-protocol ones; SCRUM-36's original (mismatched-hook) numbers are
noted in parens for continuity, not as the headline anymore.

Moving stochasticity from BLIP's weight space (Phase 2) into node/edge
representations (Phase 3) is not a strict improvement — it's a trade, and
matching the training protocol (`train_num_mc_samples=4`,
`nll_warmup_epochs=20` for both, instead of Phase 3 running at `1`/`0`)
changes *which* trade it is. **Phase 3's force channel improves
dramatically once matched**: force MAE is only +3.3% worse than BLIP-NLL
(was +16.8% mismatched), force ECE is **26.6x better** (was 2.5x), and
force uncertainty ranking now slightly *beats* BLIP-NLL (Spearman 0.2948
vs 0.2836; was slightly behind at 0.2626 vs 0.2839). This is exactly what
SCRUM-70 (training Phase 3 with NLL to fix force miscalibration at the
source, not post-hoc) set out to do, and the matched-hook run confirms it
worked. **But Phase 3's energy channel gets substantially worse**: energy
MAE is now **+72.3% worse** than BLIP-NLL (was +19.6% mismatched) — more
than triple the previously-reported gap. Energy ECE is still better than
BLIP-NLL's (0.201 vs 0.647, 3.2x) but less so than the mismatched
comparison suggested (was 4.2x). Training/inference cost advantage
shrinks once MC samples are matched: 1.74x faster training (was reported
as 5.2x), ~2.06x faster inference (unchanged, since eval-time MC count was
always equal). Parameter count is unchanged (385,925 vs 740,953, 52%).

This asymmetric effect — one quantity (force) improving sharply while the
other (energy) regresses sharply, under the same training-recipe change —
is itself a new, unexplained finding, not just a fairness correction. It
suggests NLL-with-warmup training interacts differently with Phase 3's
representation-space noise than with BLIP's weight-space noise, rather
than being a uniform "more MC samples and warmup helps" effect. Not
investigated further yet.

Per ROADMAP §4's requirement to state explicitly where Phase 3 does not
win, under the now-fair comparison: **raw energy accuracy** is Phase 3's
clear loss (not just a modest one). Phase 3's wins, now stronger than
first reported: force accuracy (nearly matched), force calibration (both
raw ECE and ranking), and parameter count; its win on training/inference
cost is smaller than first reported once samples are matched. This is one
run per phase per variant, not re-scored (SCRUM-38/71) — a first read, not
a settled result.

*Draft above is mine (Fazin/harness side); SCRUM-36 is assigned to
Dongxiao for review and sign-off before this is treated as final.*

## 5. Calibration methodology

Scalar (Kuleshov 2018) and CQR (Bao 2025) post-hoc corrections:
`docs/reviews/cqr-calibration-report-2026-11-05.pdf`,
`docs/reviews/cqr-conformal-calibration-2026-11-04.md`,
`docs/reviews/scrum-68-calibration-review-2026-11-01.md`. Open item:
SCRUM-70 (Phase 3 force miscalibration needs a training-time fix, not just
post-hoc).

## 6. Validation: leakage audit and compute budget

`docs/reviews/leakage-audit-2026-10-31.md`,
`docs/reviews/compute-budget-2026-11-01.md`. Open item: SCRUM-72 (purge
buffer + split provenance, not yet applied).

## 7. Reproducibility

Golden-run regression guard: `tests/ml/test_golden_run.py` (SCRUM-38).
Open item: no real checkpoint has been independently re-scored yet
(SCRUM-71) — only the synthetic golden run has.

## 8. Platform

Backend/frontend wiring: SCRUM-42/43 (done). Three-phase dashboard
comparison view: SCRUM-44 (open).

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
have hidden.

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
re-enter the official comparison table; no run has been independently
re-scored yet (SCRUM-38/71); the leakage audit's random-split control
shows a 17-26% accuracy gap in leakage's direction, not yet attributed to
leakage vs. noise; and Phase 3 energy calibration still diverges 13-33%
between the scalar and CQR methods, unresolved.

**What's next**, in priority order: retrain `phase2_blip` clean (SCRUM-69)
and re-run its calibration; get the golden-run guard and MC-stability
study actually run against the real `phase2_blip_nll`/`phase3_graph_stochastic`
checkpoints (SCRUM-38/49/71, not just the synthetic golden run); isolate
Phase 3 energy's remaining scalar-vs-CQR width gap; split force's
heavy-tailed hydrogen residual by hydroxyl-vs-methyl H now that atom-type
alone didn't explain it; and decide whether Phase 1's numbers need a
run-to-run noise estimate before being treated as final. Process-side:
file SCRUM-71 retroactively against its already-shipped CQR work (distinct
from the new, same-numbered reproducibility ticket filed this session —
flagged as an open naming collision), reconcile Jira assignee vs.
git-commit author on SCRUM-31/32/67, and write up the Sprint 5/6/overrun
retrospectives that are still missing.

## 10. Conclusion
