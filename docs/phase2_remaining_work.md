# Phase 2 (BLIP) — remaining work

Snapshot as of 2026-10-22, written when the team moved on to kicking off
Phase 3 (SCRUM-32) without first closing out everything below. This is the
list of what's still open, so it isn't lost in the switch.

## Open items

1. **`phase2_blip_nll` run not yet validated.** The real NLL-trained run
   (SCRUM-61's fix) is training/has trained under
   `experiments/configs/phase2_blip_nll.yaml`. Scoring it against
   `phase2_blip`'s calibration numbers (ECE, uncertainty-error Spearman)
   is SCRUM-63/SCRUM-64 - not done until someone runs `evaluate.py` and
   `benchmark.py` on the finished checkpoint and writes up the before/after.

2. **SCRUM-62 not signed off.** Fazin's review of the `benchmark.py`
   must-match-vs-model-specific protocol fix (commit `48f10a4`) is still
   sitting In Review.

3. **No BLIP model card.** `docs/model_cards/` has entries for every Phase
   1 model but none for BLIP, despite `phase2_blip` being a committed,
   reported result. Tracked as SCRUM-65 (Ruturaj).

4. **`kl_weight=0.01` was never tuned.** Both `phase2_blip.yaml` and
   `phase2_blip_nll.yaml` say so in their own comments - it was a starting
   value to get a first real run, not a value chosen for calibration
   quality. Retuning it (watching `train_kl`/`val_kl` in `history.csv`) is
   explicitly deferred, not forgotten.

5. **Backend never wired to a real checkpoint.** SCRUM-42 ("Wire POST
   /predictions to real Phase 1 checkpoint") and SCRUM-43 ("Wire
   uncertainty into /predictions + /benchmarks") are both overdue (due
   2026-10-18) and still To Do. `backend/app/services/inference.py` is
   still a stub - BLIP's `predict_distribution` isn't reachable from the
   API at all yet, so there's no live-demo path for Phase 2's actual
   uncertainty output, only the offline benchmark/eval scripts.

6. **SCRUM-39 status is ambiguous.** It's titled "Extend harness with UQ
   metrics; Phase 1 vs Phase 2 comparison" and still shows To Do, but
   `ml/utils/uq_metrics.py` (242 lines: ECE, coverage, Spearman,
   risk-coverage, NLL/CRPS, sharpness) already exists and is wired into
   `evaluate.py`/`benchmark.py`. Either the ticket is stale and should
   close, or there's a real remaining piece - needs Fazin to confirm
   which, not an assumption either way.

7. **SCRUM-38 overdue.** Golden-run regression guard (fixed-seed Phase 1
   reproducibility test) - due 2026-10-18, still To Do.

8. **SCRUM-47 / SCRUM-49 not started.** Leakage audit before protocol lock,
   and profiling compute budget / setting an MC sample cap. SCRUM-49 in
   particular is now overdue in spirit: `phase2_blip_nll` ran K=4 MC
   samples for ~200 epochs on CPU (~5 hours) with no cap or budget check
   in place first.

9. **SCRUM-21 (Phase 2 epic) still open.** Sitting In Review; closing it
   depends on items 1-3 above landing.

## Why this list exists

The team is starting Phase 3 (SCRUM-32) before all of the above is done,
on the reasoning that Phase 3's architecture work doesn't block on it and
the 23/10/2026 demo needs forward progress. Nothing here is cancelled -
it's sequenced after the demo.
