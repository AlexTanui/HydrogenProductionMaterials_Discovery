# SCRUM-49: MC sample-count stability study — results

The stability study (`ml/training/mc_stability.py`) was blocked on `phase2_blip_nll.pt` and
`phase3_graph_stochastic.pt` as of `docs/reviews/compute-budget-2026-11-01.md` §3. Both checkpoints
now exist (trained under SCRUM-24's Option A fixes, committed `1eaea76`). Ran the study against both,
using the criterion agreed 1 Nov: smallest K from which energy/force ECE, coverage@90% and Spearman
are each within 0.01 of their K=100 value (mean over 3 seeds), with seed-to-seed spread also within
0.01. Raw output: `reports/mc_stability/phase2_blip_nll.json`, `reports/mc_stability/phase3_graph_stochastic.json`.

## Result: no stable K found for either checkpoint, up to K=100

**phase2_blip_nll** (`reports/mc_stability/phase2_blip_nll.json`):

| K | gap to K=100 | seed spread | stable? |
|---|---|---|---|
| 2 | 0.1877 | 0.0223 | no |
| 5 | 0.0938 | 0.0204 | no |
| 10 | 0.0501 | 0.0171 | no |
| 20 | 0.0306 | 0.0157 | no |
| 50 | 0.0065 | 0.0197 | no (gap passes, spread doesn't) |
| 100 | 0.0000 | 0.0133 | no (spread alone fails, even at the reference K) |

**phase3_graph_stochastic** (`reports/mc_stability/phase3_graph_stochastic.json`):

| K | gap to K=100 | seed spread | stable? |
|---|---|---|---|
| 2 | 0.3003 | 0.0129 | no |
| 5 | 0.0960 | 0.0090 | no (gap fails) |
| 10 | 0.0422 | 0.0195 | no |
| 20 | 0.0195 | 0.0258 | no |
| 50 | 0.0053 | 0.0285 | no (gap passes, spread doesn't — and is worse than at K=20) |
| 100 | 0.0000 | 0.0169 | no (spread alone fails) |

**`stable_k: None` for both.** The *gap* component (convergence toward the K=100 reference as K grows)
behaves as expected — it shrinks monotonically and is well under 0.01 by K=50 for both checkpoints.
The *spread* component (seed-to-seed variability at a fixed K, 3 seeds) does not: it doesn't shrink
monotonically with K for either run, and stays above the 0.01 tolerance even at K=100 — the reference
point itself fails its own spread criterion.

## What this means

**More MC samples will not fix this.** If the seed-to-seed spread were shrinking with K, however slowly,
raising K past 100 would be a reasonable next step. It isn't — Phase 3's spread is actually *larger*
at K=50 (0.0285) than at K=10 (0.0195), and larger than K=100's (0.0169). This looks like genuine
seed-to-seed estimation noise on a 2,000-frame validation sample under the current UQ metrics, not a
bias that more sampling per frame resolves.

**Caveat on the measurement itself:** the agreed criterion uses 3 seeds, and `statistics.pstdev` of
3 values is itself a noisy estimate of "spread" — with only 3 draws, one unusual seed can move it
substantially. This result should be read as "the 3-seed protocol doesn't demonstrate stability," not
as a precise bound on the true seed-to-seed variance. A 5+ seed re-run would pin this down better, at
proportionally higher cost (current cost: ~6 min/checkpoint for 3 seeds × 6 K values up to 100).

**Practical recommendation:** keep the existing project default, `num_mc_samples: 20`, rather than
raising K. The data gives no evidence a higher K would stabilize the metrics, so paying 2.5x–5x more
inference cost (linear in K, per `compute-budget-2026-11-01.md`) for K=50/100 would not buy the
stability the criterion was meant to confirm. Instead, this should be recorded as a measurement-precision
caveat on every ECE/coverage/Spearman number reported anywhere in the project: those figures carry an
inherent ~0.01–0.03 (absolute) seed-level noise floor at K=20 that does not visibly shrink with more
samples in the tested range. The large, headline differences already reported (e.g. BLIP-NLL vs Phase 3
energy ECE: 0.647 vs 0.201, a 0.45 gap) are far outside this noise floor and stand. Finer distinctions on
the same order as the noise floor — e.g. Spearman differences of 0.01–0.03 between near-tied runs — should
be treated as not clearly distinguishable from seed noise until re-measured with more seeds.

## Disposition

This is the measurement the ticket asked for, and it's conclusive: the criterion, as specified, finds
no stable K. Per the ticket's own instruction that the stability criterion was "agreed with Dongxiao
rather than chosen unilaterally," how to proceed from a null result is a team call, not mine to decide
unilaterally either — logged to SCRUM-49 for that decision rather than closed outright. The two
candidate paths: (a) accept K=20 as the pragmatic default anyway, with the noise-floor caveat written
into the eval protocol (`glossary.md` §6), which is what I'd recommend; or (b) spend the ~12 extra
CPU-minutes to re-run with 5+ seeds to see whether the null result is a 3-seed artifact before accepting
it as final.
