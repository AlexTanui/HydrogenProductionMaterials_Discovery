# Proposal: four `train.py` fixes from the 22 Oct audit (1 Nov 2026)

**Status: proposal only. No code has changed.** Each fix changes how some models train or what their records mean, so it needs team agreement before anyone implements it. Findings are from `docs/reviews/main-audit-2026-10-22.md` (branch `fazin-audit-oct22`), re-checked as still open on `main` at `46b5f99` on 31 Oct.

## Summary

| Finding | Fix | Protocol keys that change | Runs to retrain | Golden test reacts? |
|---|---|---|---|---|
| **4.** Validation divides KL by its own (shorter) loader, so validation counts KL about 2× | Select, schedule and stop on validation **data** loss; log KL separately | `selection` (shared, must-match) | `phase2_blip`, `phase2_blip_nll`, `phase3_graph_stochastic` | **No** |
| **N1.** Validation data sets the NLL variance floor, and steers the LR schedule and early stopping | Floor from the warm-up epoch's **training** error; schedule and stop as in 4 | `loss_override`, `nll_min_*` (model-specific) | `phase2_blip_nll` (and Phase 3 NLL if `e332e1d` lands) | **No** |
| **2c.** NLL values are logged under `*_mse` history columns | Always log the MSE of the mean prediction in `*_mse`; add `*_nll` columns | none | none | **No** |
| **N6.** Validation metrics are a mean of batch means | Weight each batch by its frame count | `selection` (meaning; must-match) | potentially all 8 | **No** |

**The golden test reacts to none of these** (Step 4, accepted 1 Nov). It runs one epoch of the deterministic MPNN, so:
- MPNN has no KL term (items 4 and N1);
- checkpoint selection, the scheduler and early stopping never act in a one-epoch run (N1, N6);
- it scores through `evaluate.py`, not `history.csv` (2c).

**Each fix therefore needs its own test.** One is listed per item below.

## 4. KL in the validation loss

- **What's wrong.** `batch_loss` adds `kl_weight · KL / n_batches`, where `n_batches = len(loader)`: 313 batches for train, 157 for validation. So validation's per-batch KL share is 313/157 ≈ 2× training's. Measured `val_kl / train_kl` medians are 1.990 (`phase2_blip`) and 1.988 (`phase2_blip_nll`).
- **Why it matters.** In `phase2_blip`, KL is 73% of `val_loss` at the selected epoch. Checkpoint selection, early stopping and the LR scheduler all key on `val_loss`, so for BLIP they follow the KL penalty, not the fit. Selecting on data loss would change `best_epoch` from 200 to 198 (`phase2_blip`) and from 200 to 191 (`phase2_blip_nll`).
- **Fix.** Training is unchanged: KL counted once per pass over the training data is the standard minibatch ELBO. Validation:
  - computes `val_data_loss` = energy_w · E-term + force_w · F-term, with no KL;
  - **selection, early stopping and `ReduceLROnPlateau` all use `val_data_loss`;**
  - logs `val_kl` separately, using the *training* divisor so the two columns compare directly.

  `history.csv` gains a `val_data_loss` column. `val_loss` keeps its current meaning, because the backend's job status reads it (`backend/app/services/jobs.py`).
- **Protocol keys.** `selection` changes, for example to "lowest validation data loss (energy and force terms, no KL) on the val subset; early stopping and the LR scheduler use the same quantity". It's in `SHARED_PROTOCOL_KEYS`, so old and new records would refuse to share a table.
  - **For a model with `kl_weight = 0`** (every Phase 1 model), the old and new rules pick the same epoch by construction: there's no KL to drop.
  - **Proposal:** `benchmark.py` treats the two `selection` texts as equal when `kl_weight == 0`, with a test. Phase 1 records then stay valid without retraining. The alternative is retraining all five Phase 1 models, about 5.4 h on the Mac, for no behavioural change.
- **Runs to retrain:** the three runs with `kl_weight > 0`:
  - `phase2_blip`: 1.3 h on the Mac; Step 7 retrains it anyway;
  - `phase2_blip_nll`: 3.9 h;
  - `phase3_graph_stochastic`: 0.7 h.

  Plus about 0.5 h of scoring.
- **Test it needs:**
  - a tiny model with a non-zero KL, trained for 2 epochs, records `best_epoch` by `val_data_loss`;
  - the history's `val_kl` uses the training divisor (`val_kl / train_kl` ≈ 1 for the same weights);
  - the scheduler is stepped with `val_data_loss`, checked with a spy on `scheduler.step`.
- **Not covered:** audit 5a, Phase 3's KL normalisation. `RepresentationNoise.kl_divergence` sums over nodes and channels, a per-sample KL divided as if it were a global one. That's in `graph_stochastic.py`, not `train.py`. Fixing it also means retraining Phase 3, so the two Phase 3 retrains should be done together.

## N1. Validation data shapes training beyond selection

- **What's wrong.**
  - The NLL variance floor is set once, at the last warm-up epoch, from that epoch's **validation** MSE (`train.py`, the `epoch == nll_warmup` block). So validation data sets a training-loss hyperparameter.
  - The LR scheduler and early stopping step on `val_loss`, which for BLIP is mostly KL (item 4).
- **Fix.**
  - **Floor:** set it from the same epoch's **training-pass** MSE (`train_e`, `train_f`, already computed in the loop), or from a held-out slice of the training subset if the team prefers an evaluation-mode estimate. Validation is then read only for selection, scheduling and stopping.
  - **Scheduling and stopping:** covered by item 4's fix.
  - **Still open regardless (SCRUM-61, kept open on 31 Oct):** the floor's size. It is set once and never updated; `phase2_blip_nll`'s floor is 610× its final squared energy error and 83× its force error (audit 2a). Evaluation also adds no floor (2b). This fix stops validation leaking into training; it doesn't settle those.
- **Protocol keys.** `loss_override`'s text changes ("variance floor ... = the last warmup epoch's **training** MSE"), and so do the recorded `nll_min_energy_var` / `nll_min_force_var`. Both are in `MODEL_SPECIFIC_PROTOCOL_KEYS`, so they're reported, not gated on. Item 4 covers `selection`.
- **Runs to retrain:** `phase2_blip_nll`, already in item 4's list. Also Phase 3's NLL config, `phase3_graph_stochastic_nll`, if `e332e1d` (on `origin/alex`) lands; it takes the same code path.
- **Test it needs:**
  - the floor equals the warm-up epoch's training MSE;
  - validation metrics never feed it, checked by replacing the val loader's data with a constant and asserting the floor doesn't move.

  The existing `tests/ml/test_harness.py` test that checks the floor against "that epoch's measured validation MSE in history.csv" pins the current behaviour. It changes with the fix. That's a deliberate behaviour change, not a weakened test, and it should be said in the same commit.

## 2c. NLL logged as MSE

- **What's wrong.** On the NLL branch `batch_loss` returns the NLL terms, and `run_epoch` logs them under `train_energy_mse`, `val_energy_mse` and the like. In `phase2_blip_nll/history.csv`, `val_energy_mse` is 21.78 at epoch 20 (a real MSE) and then 2.51 at epoch 21 and 2.46 at epoch 200 (NLL values). 0.5·ln(2π·21.78) = 2.4595, so the "MSE" column settles at the floor's own constant.
- **Fix.**
  - `*_energy_mse` / `*_force_mse` always hold the MSE of the predictive mean. On the NLL branch that's one extra `F.mse_loss(pred_e, y)` on tensors already computed.
  - New columns `*_energy_nll` / `*_force_nll` hold the NLL terms, empty on the MSE branch.
  - `batch_loss` returns two more values; `run_epoch` and the tests that call `batch_loss` directly follow.
- **Protocol keys:** none; `history.csv` columns aren't protocol.
- **Runs to retrain:** none. The existing `phase2_blip_nll` history keeps its old meaning. Run records are never hand-edited, so its model card should say which epochs' `*_mse` columns hold NLL.
- **Test it needs:** a two-epoch BLIP-NLL run (warm-up 1). Epoch 2's `*_mse` columns equal the MSE recomputed from the mean prediction, and its `*_nll` columns are present; epoch 1's `*_nll` columns are empty.

## N6. Validation averaged per batch, not per frame

- **What's wrong.** `run_epoch` returns the mean of batch means.
  - The validation subset is 5,000 = 156 × 32 + 8 frames, so the last 8-frame batch counts as much as a full one: 4× per frame. It carries 0.64% of the weight instead of 0.16%.
  - Training has 10,000 = 312 × 32 + 16 frames, so its last batch weighs 2×, but training loss is only logged.
  - Reported test scores are unaffected, because `evaluate.py` accumulates sums.
- **Fix.** Accumulate each term × the batch's frame count, and divide by total frames, for train and validation alike. Every protocol molecule has a fixed atom count, so frame weighting equals force-component weighting.
- **Protocol keys.** `selection`'s meaning changes for every model. Its text should say "per-frame mean", which splits every existing record from new ones. Here there is no equivalence argument like item 4's: the small change in validation loss could move `best_epoch` for any model. Per-batch values aren't stored, so nobody can check which.
- **Runs to retrain:** all 8, for strict comparability. That's about 11.4 h on the Mac plus about 0.5 h of scoring.
- **Recommendation: defer N6** to the next full retrain, and state it in `docs/phase1_protocol.md` Limitations as a known, selection-only effect of this size. Adopting it alone costs a full retrain for a 0.48-point weighting difference on one batch in 157.
- **Test it needs:** `run_epoch` on a loader with unequal batch sizes returns the hand-computed frame-weighted mean.

## Sequencing

**Decide this before Step 7.**
- Step 7 retrains `phase2_blip` under the current harness, and the plan says to stop and re-plan if 6b is adopted.
- Adopting items 4 and N1 after Step 7 would retrain BLIP twice, which wastes about 1.5 h.
- The scheduled noise study (MPNN, BLIP-NLL and Phase 3, three seeds each, on this laptop) should run under whichever harness the final tables use.

**Options:**
- **(A, recommended) Adopt 4, N1 and 2c now, as one change. Defer N6.**
  - Retrain the three KL runs: Step 7's `phase2_blip` plus `phase2_blip_nll` and `phase3_graph_stochastic`, about 6 h on the Mac plus 0.5 h scoring.
  - Phase 1 records stay valid through the explicit `kl_weight = 0` equivalence for `selection`.
  - Fold the Phase 3 retrain in with audit 5a if that fix lands.
- **(B)** Adopt all four, including N6, and retrain all 8 (about 12 h). It's cheapest to do this together with the noise study, since that study retrains three of the eight anyway.
- **(C)** Adopt none now, and list all four as limitations of the current tables.

**Also on this list (Step 1 review):** `train.py`'s pinned `protocol.loss` text says the KL term applies to "BLIP only", but GraphStochastic uses it too. The text is must-match, so changing it splits every record. Leave it until a full retrain (option B, or later), and note it in the protocol doc meanwhile.
