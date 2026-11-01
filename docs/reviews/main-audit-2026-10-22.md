# Audit of `main` before Phase 2/3 work (22 Oct 2026)

Read-only audit of `main` at `fdc2353`, on local branch `audit/main-2026-10-22`. No code was changed. All throwaway scripts ran outside the repo. Areas and their tickets:

- **Harness and tests** (`train.py`, `evaluate.py`, `benchmark.py`, `config.py`, registry, scaling, metrics, `tests/`, `backend/tests/`): SCRUM-24, SCRUM-57.
- **Models:** MPNN SCRUM-51, SchNet SCRUM-53, BLIP SCRUM-21/SCRUM-31, PhysNet SCRUM-55, DimeNet++ SCRUM-54, PaiNN SCRUM-52.
- **Data and backend:** `ml/data/` SCRUM-19/SCRUM-28; `backend/app/` SCRUM-25.
- **Phase 3:** `graph_stochastic.py` is a first draft (b616970); SCRUM-22, SCRUM-32.

PRs #8 (demo deck) and #9 (predictions export) were open and are not part of `main`.

## 1. Test suite

`pytest tests/ backend/tests/` with the project `.venv`; both gold files were present, so preprocessing was not re-run.

| Result | Count |
|---|---|
| Passed | 427 |
| Skipped | 14 |
| Failed | 0 |
| Errors | 0 |

- **All 14 skips have one reason:** `experiments/checkpoints/phase2_blip.pt` does not exist (`tests/ml/test_painn_physics.py:579-665`). They are the BLIP trained-checkpoint physics checks.
- **`backend/tests/` collects 0 tests.** `backend/tests/test_api.py` is empty, so the API has no coverage.
- **"Passed" is weaker than it looks.** The Phase 1 trained-checkpoint physics checks passed against the *pre-retrain* checkpoints (item 7).

## 2. Part 1: suspected problems

| # | Suspected problem | Verdict | Severity |
|---|---|---|---|
| 1 | No Phase 2 run passes the benchmark | CONFIRMED | blocker for SCRUM-39 |
| 2a | NLL floor set once, never updated | CONFIRMED | blocker for Phase 2 NLL |
| 2b | Floor in training, absent in evaluation | CONFIRMED | blocker for Phase 2 NLL |
| 2c | NLL values in `*_mse` history columns | CONFIRMED | should-fix |
| 3 | BLIP energy offset | PARTLY: pattern confirmed, bias not measurable here | should-fix |
| 4 | Validation KL doubled; selection affected | CONFIRMED, and worse than suspected | blocker for Phase 2 |
| 5a | Phase 3 KL not comparable to BLIP's | CONFIRMED | blocker for Phase 3 |
| 5b | Phase 3 has no NLL path | CONFIRMED (deliberate, documented) | should-fix |
| 5c | Prior, not error, sets Phase 3 noise | CONFIRMED (at initialisation) | blocker for Phase 3 |
| 5d | Stale or wrong-batch `_last_kl` | NOT CONFIRMED (latent hazard only) | later |
| 6 | Backend `/benchmarks` broken | CONFIRMED | should-fix |
| 7 | Checkpoints don't match run records | CONFIRMED | should-fix |
| 8 | Hand-edited NLL run record | CONFIRMED | should-fix |
| 9 | Test gaps | CONFIRMED (partly for `build_graph`) | should-fix |

### 1. No Phase 2 run passes the benchmark: CONFIRMED

- **`phase2_blip`:** `--runs phase1_baseline phase2_blip` fails with `ValueError: phase2_blip: git_dirty=True at commit e581be8. Accepted results must come from a clean commit (SCRUM-48)` (`ml/training/benchmark.py:222`).
- **`phase2_blip_nll`:** `--runs phase1_baseline phase2_blip_nll` fails with `phase2_blip_nll: scored as a distribution, but test_metrics.json records no eval MC sample count ... (SCRUM-49)` (`benchmark.py:247`).
- **A second problem is masked.** `phase2_blip` *also* has no `num_mc_samples` (`stochastic=True`, field absent), but the dirty-tree check fires first.
- **Fix:** both need their checkpoints re-scored with the current `evaluate.py`, and `phase2_blip` needs a retrain from a clean commit. SCRUM-48, SCRUM-49, SCRUM-39.

### 2. NLL variance floor

**2a, set once, never updated: CONFIRMED.**
- **Where:** the floor is assigned only at `epoch == nll_warmup` (`ml/training/train.py:651-657`), from that epoch's *validation* MSE, and never again. It is added to the variance in `_gaussian_nll_loss` (`train.py:229`, used at `train.py:295-298`).
- **Energy** (`phase2_blip_nll`): floor 21.78 (kcal/mol)², set at epoch 20. Final test MSE is 0.0357, so the floor is **610×** the final squared error.
- **Force:** floor 7.36 against a final MSE of 0.0888, **83×**.
- **Also a validation leak:** validation data sets a training-loss hyperparameter. See N1 in section 3.

**2b, in training but not in evaluation: CONFIRMED.**
- **Evaluation adds no floor:** `ScaledEnergyModel.predict_distribution` (`ml/models/scaling.py:98-130`), `BLIP.predict_distribution` (`ml/models/blip.py:307-332`) and `evaluate.py:330`.
- **The floor isn't in the checkpoint either:** `nll_min_*` are plain attributes, not buffers.
- **Training saw a different distribution from the one scored:**
  - training's predictive std was at least √21.78 = **4.67 kcal/mol** for energy, against **0.0127** scored (`sharpness_mean_std`), **368×**;
  - for forces, 2.71 against 0.0090, **301×**.
- **A second mismatch:** training uses the *biased* variance over K = 4 samples (`blip.py:365`), evaluation the *unbiased* variance over K = 20 (`blip.py:332`).

**2c, history columns: CONFIRMED.**
- `run_epoch` accumulates whatever `batch_loss` returns as `energy_loss`/`force_loss` (`train.py:337-343`). In the NLL branch that is the NLL (`train.py:297-298`), but it is logged under `train_energy_mse`/`val_energy_mse` etc. (`train.py:104`).
- `phase2_blip_nll/history.csv`: `val_energy_mse` is 21.7816 at epoch 20 (MSE), then 2.5098 at epoch 21 and 2.4604 at epoch 200 (NLL).
- 0.5·ln(2π·21.78) = 2.4595, so the logged energy "MSE" settles at the floor's own constant.

### 3. BLIP energy offset: PARTLY

The pattern is confirmed from the records; the bias itself could not be measured.

| Run | Energy RMSE/MAE |
|---|---|
| phase1_baseline | 1.234 |
| phase1_dimenet | 1.281 |
| phase1_painn | 1.276 |
| phase1_physnet | 1.276 |
| phase1_schnet | 1.311 |
| phase2_blip | **1.142** |
| phase2_blip_nll | **1.097** |

Pure Gaussian scatter gives √(π/2) = 1.2533.

- **Model-based estimate:** fitting "constant offset plus Gaussian scatter" to MAE and RMSE gives |bias| ≈ 0.154 kcal/mol (`phase2_blip`) and 0.171 (`phase2_blip_nll`). That would be most of each run's MAE. The sign is not identifiable from MAE and RMSE.
- **A suggestive data fact:** the 10k training subset's mean energy is −97,195.819 kcal/mol and the test split's is −97,196.000, so the training subset sits **0.18 kcal/mol above** the test split. A model anchored on training-set energies would predict too high by about that much. This is a hypothesis, not evidence of the cause.
- **Not checkable here:** no `phase2_blip*.pt` exists on this machine, and `--save-predictions` (PR #9) is not on `main`. Measuring the mean signed error needs the `phase2_blip*.pt` checkpoints (SCRUM-21).

### 4. KL in the validation loss: CONFIRMED, and worse than suspected

- **The doubling.** `batch_loss` divides KL by `n_batches = len(loader)` (`train.py:309-313`, `330`): 313 for train, 157 for val. Measured `val_kl/train_kl` medians are 1.990 (`phase2_blip`) and 1.988 (`phase2_blip_nll`), against 313/157 = 1.994.
- **KL dominates the MSE run's selection.** In `phase2_blip`, KL is **73% of `val_loss` at the selected epoch** (47.0 of 64.5) and **76% of the training loss** at epoch 200 (23.6 of 31.3).
- **Everything keyed on `val_loss` follows KL.** That covers checkpoint selection, early stopping (`train.py:716`) and the LR scheduler (`train.py:682`). `phase2_blip` records 13 of its last 20 epochs as "improved" while its validation *data* loss rose on 8 of 19 steps; MPNN records 6 of 20.
- **Selecting on data loss would change `best_epoch`:** 200 → 198 for `phase2_blip`, and 200 → 191 for `phase2_blip_nll` (selection window ≥ 21, after the reset).

### 5. Phase 3 vs Phase 2 fairness (`ml/models/graph_stochastic.py`)

**5a, KL scaling: CONFIRMED.**
- **Where:** `RepresentationNoise.kl_divergence` sums over nodes and channels (`graph_stochastic.py:128`), accumulated over layers at `241-245`.
- **Measured at initialisation** (real ethanol frames, model.train()):

  | Batch size | Atoms | GraphStochastic KL | BLIP KL | GS loss term (0.01·KL/313) | BLIP loss term |
  |---|---|---|---|---|---|
  | 1 | 9 | 3,636 | 812,417 | 0.116 | 25.96 |
  | 8 | 72 | 28,997 | 812,417 | 0.926 | 25.96 |
  | 32 | 288 | 115,995 | 812,417 | 3.706 | 25.96 |
  | 64 | 576 | 231,820 | 812,417 | 7.406 | 25.96 |

- **Same `kl_weight`, different regularisation.** GraphStochastic's KL is ≈403 per atom. At the protocol's batch 32, the same `kl_weight=0.01` regularises it **≈7× less** than BLIP.
- **It also moves with batch and molecule size:** a 370-atom MD22 molecule carries 41× ethanol's KL.
- **The normalisation is wrong for a per-sample KL.** `train.py` divides this per-batch, per-sample KL by `n_batches`, as if it were a global weight KL.

**5b, no NLL path: CONFIRMED.**
- There is no `predict_training_distribution`, so `supports_training_distribution` is False and `nll_active` can never be True.
- This is deliberate and documented (`graph_stochastic.py:176-183`). Unknown `train_num_mc_samples`/`nll_warmup_epochs` keys fail loudly at build time, so nothing is silently ignored.

**5c, the noise level is set by the prior and an MSE variance penalty, not by the error: CONFIRMED** (at initialisation).
- **The algebra:** under single-sample MSE the expected gradient on the noise scale is a variance penalty (≥ 0), independent of the residual to first order. The KL pulls σ_eff toward `PRIOR_SIGMA = 0.1`.
- **Measured on 8 frames, 40 noise draws:**
  - mean ∂(KL term)/∂log σ ≈ −0.002 per layer, pushing σ up toward the prior;
  - the data term is 4–40× smaller;
  - the data term shows no systematic response when the energy targets are shifted by 0, 5 or 50 kcal/mol (mixed signs, noise only).
- **Implication:** this is BLIP's MSE-run failure mode: nothing ties the spread to the error.

**5d, stale `_last_kl`: NOT CONFIRMED.**
- **The only production caller is fine.** `train.py:311` reads the KL right after the forward pass of the same batch.
- **Latent hazards:**
  - `_last_kl` starts as `tensor(0.0)` (`graph_stochastic.py:222`) and is returned silently if read before any forward pass;
  - `predict_distribution` overwrites it with the last MC sample's KL (`286-297` → `245`).
- **Fix:** an assertion or a per-call return is worth adding before a second caller appears.

### 6. Backend `/benchmarks`: CONFIRMED

- **The failure:** `backend/app/api/benchmarks.py:25` builds `BenchmarkResult(phase=payload["phase"], model=payload["model"], **payload["metrics"])`.
  - A real `evaluate.py` result has no top-level `model`.
  - Its `metrics` is the accumulator dict (`conventions`, `overall`, `per_key`, `schema_version`).
  - The §4 shape is already in `benchmark_row`.
- **The demonstration:** I copied `experiments/runs/phase1_painn/test_metrics.json` into a temporary results folder and called `GET /benchmarks` through FastAPI's TestClient. Result: **HTTP 500, `KeyError('model')`**.
- **Two related problems:**
  - the endpoint reads the gitignored `experiments/results/` (`benchmarks.py:12`), not the committed `experiments/runs/*/test_metrics.json`, so on a fresh clone it can only return `[]`;
  - `pydantic-settings` (declared in `backend/requirements.txt`) is not installed in the project `.venv`, so the backend cannot be imported there. The call above used a scratch-only stand-in for `pydantic_settings.BaseSettings`.
- **`inference.py` is still the deterministic stub** (`backend/app/services/inference.py:1-42`). See N6 for its catalogue.

### 7. Retrained Phase 1 checkpoints: CONFIRMED

| Run | `checkpoint_sha256` in run.json | Local `.pt` |
|---|---|---|
| phase1_baseline | `36a760377dbc` | `24b83e6bcabe` (differs) |
| phase1_dimenet | `9f2484b35baf` | `322a42c719a2` (differs) |
| phase1_painn | `be3c3fd74a86` | `0cc2c7205c36` (differs) |
| phase1_physnet | `1f24eb29d3e8` | `f56aa40d7467` (differs) |
| phase1_schnet | `17fba407f928` | `033a7323382a` (differs) |
| phase2_blip, phase2_blip_nll | `3489d273e42f`, `bf6c13fb152d` | missing |

- **What the local files are:** the pre-retrain checkpoints. Each matches the sha256 in the record from before 3498c08/b1c2d89.
- **The test doesn't check:** `tests/ml/test_painn_physics.py:274-283` only checks that the file exists and its version, then loads it. Today's "passing" trained-checkpoint physics checks therefore validated the old weights.
- **No re-score:** none of the five matches, so as instructed nothing was re-scored and no run record was touched.

### 8. Hand-edited NLL run record: CONFIRMED

- **What changed:** fdc2353 (22 Oct) changed only `experiments/runs/phase2_blip_nll/run.json`. It rewrote `protocol.loss` and added `loss_override`, `nll_min_energy_var`, `nll_min_force_var` and `nll_min_var_epoch`.
- **No trace in the record:** nothing in it says it was edited (0 matches for patch/edited/migrat/manual).
- **The values are inconsistent with its own commit.** Its `git_commit` is `fdada3b`, whose `train.py` never writes `loss_override` (0 occurrences).
- **The values are right:** they match `history.csv` at epoch 20.
- **The problem is provenance** (SCRUM-48: records are "written automatically, not by hand"). Left as instructed; you will raise the migration-script fix separately.

### 9. Test gaps: CONFIRMED (partly for `build_graph`)

- **Empty files:** `tests/test_md17.py`, `tests/test_models.py` and `backend/tests/test_api.py` are 0 bytes.
- **No tests at all** reference `silver_to_gold`, `bronze_to_silver`, `validate_sample` or `load_split_zip`. `trajectory_block_split` and `_PERIODIC_TABLE` appear only in comments.
- **`build_graph` is used, not tested:** it builds graphs for the PaiNN tests, but its edge set, symmetry and cutoff inclusivity are never checked. So the splitting, literature-split, unit-validation and loader code has no direct tests.
- **`tests/test_dimenet.py`:**
  - `setUpClass` has no skip guard and imports the superseded `ml.training.train_dimenet` (lines 10-19);
  - on 21 Oct, before `ethanol_ccsd_t.npz` was built on this machine, it produced 3 errors (`FileNotFoundError`), not skips.

## 3. New findings from the sweep

**How the sweep was done.** The three sub-agents launched for it stopped on their first step (account session limit) and returned nothing. I did the sweep myself instead: targeted checks per category rather than a line-by-line read of every file (see section 4).

### Should-fix

**N1. Validation data shapes training beyond checkpoint selection.**
- *File:* `ml/training/train.py:651-657`, `682`, `716`.
- *Evidence:* three things read validation data:
  - the NLL floor is calibrated from the validation MSE, so validation data sets a training-loss hyperparameter;
  - `ReduceLROnPlateau` steps on `val_loss`;
  - early stopping uses `val_loss`.

  For BLIP, `val_loss` is mostly KL (item 4), so the LR schedule follows KL rather than fit.
- *Ticket:* SCRUM-61 (`train.py`; the floor code came in fdada3b).
- *Fix:* calibrate the floor from training-pass error, or from a held-out slice of train. Schedule and stop on validation *data* loss only.

**N2. KL dominates the MSE-trained BLIP's objective.**
- *File:* `experiments/configs/phase2_blip.yaml` (`kl_weight: 0.01`); `train.py:309-313`.
- *Evidence:* `train_kl` is 68% of training loss at epoch 100 and 76% at epoch 200. A plausible contributor to BLIP's 2× energy MAE versus its own MPNN backbone, and possibly to item 3's offset. This is a hypothesis.
- *Ticket:* none found (BLIP: SCRUM-21).
- *Fix:* sweep `kl_weight`, or scale the KL by the number of training frames (minibatch-ELBO convention) rather than `n_batches`. Report the KL share in the run record.

**N3. A hard 5 Å cutoff makes the energy discontinuous in MPNN, BLIP, GraphStochastic and SchNet.**
- *File:* `ml/data/md17.py:242` (`dist <= cutoff`); no envelope in `mpnn.py`, `blip.py`, `graph_stochastic.py`, `schnet.py`.
- *Evidence:* a two-atom C–H pair moved from 4.99 to 5.01 Å (edges rebuilt by `build_graph`, models at initialisation):

  | Model | Energy jump | \|F\|·0.02 Å |
  |---|---|---|
  | MPNN | 8.0e-3 | 1.2e-5 |
  | BLIP | 8.4e-3 | 2.7e-5 |
  | GraphStochastic | 9.0e-4 | 5.9e-5 |
  | SchNet | 3.3e-3 | 1.7e-5 |

  PaiNN, PhysNet and DimeNet++ are continuous (jump ≤ 3e-8).

  This is harmless for ethanol (largest pair distance 4.43 Å, complete graph). It is not harmless for paracetamol (73% of pairs within 5 Å), azobenzene (59%) or MD22. All three phases share the discontinuous backbone.
- *Ticket:* none found (MPNN backbone SCRUM-51, SchNet SCRUM-53, BLIP SCRUM-21, Phase 3 SCRUM-32).
- *Fix:* multiply messages by a smooth cutoff envelope, as PaiNN and PhysNet do, before any phase trains on a molecule other than ethanol. Add the 4.99/5.01 check as a test.

**N4. The superseded standalone trainers are still runnable and write into shared folders.**
- *File:* `ml/training/train_schnet.py:58-60` (opens the test split), `181-186` and `247-248` (writes `experiments/checkpoints/<name>.pt` and `experiments/results/<name>.json`); `train_physnet.py`; `train_dimenet.py` (imported by `tests/test_dimenet.py`).
- *Evidence:* a second path reads the test split outside `evaluate.py`. It can overwrite a harness checkpoint of the same name, and writes into the folder the backend's `/benchmarks` reads.
- *Ticket:* none found (SchNet SCRUM-53, PhysNet SCRUM-55, DimeNet++ SCRUM-54).
- *Fix:* make them raise on import with a pointer to `train.py`, or delete them. Move `test_dimenet.py` onto the shared harness.

**N5. Backend gaps beyond item 6.**
- *File:* `backend/app/core/config.py:14`; `backend/app/services/inference.py:14-23`, `30`; `backend/requirements.txt`.
- *Evidence:*
  - the default `checkpoint_path` is `experiments/checkpoints/baseline.pt`, a name no run produces;
  - the stub's catalogue lists `naphthalene` and `salicylic_acid` (not in the data) and omits `azobenzene` and `paracetamol` (in the data);
  - an unknown molecule silently gets 10 atoms and a plausible-looking energy;
  - the backend's dependencies are not in the project `.venv`.
- *Ticket:* SCRUM-25 (`backend/tests/`: SCRUM-24).
- *Fix:*
  - point `/benchmarks` at `experiments/runs/*/test_metrics.json` and map `benchmark_row`;
  - build the catalogue from the gold roster and reject unknown molecules with a 404;
  - install the backend's requirements in CI and add API tests.

### Later

**N6. Validation metrics are averaged per batch, not per frame.**
- *File:* `ml/training/train.py:349`.
- *Evidence:* `run_epoch` returns means of batch means. The final 8-frame validation batch (5,000 = 156·32 + 8) weighs 4× per frame. This affects selection only, not reported scores.
- *Ticket:* SCRUM-24.
- *Fix:* accumulate sums and divide by frames.

**N7. Split-provenance config keys are recorded but never checked.**
- *File:* `ml/config.py:20-37`.
- *Evidence:* `dataset`, `train_split`/`val_split`/`test_split` and `use_literature_split` are read nowhere in `ml/` (grep). The gold file stores `used_literature_split`, and nothing compares the two. A config can misstate how its split was made, and that statement is copied into run.json and compared as the `data:` block.
- *Ticket:* SCRUM-24 (data side: SCRUM-28).
- *Fix:* check them against the gold file in `open_split`.

**N8. Stale documentation.**
- *Files and problems:*
  - `CLAUDE.md:19` ("Current state": `ml/models`, `ml/training`, `ml/utils` are implemented);
  - `glossary.md:4-5` ("structure only" in `ml/`);
  - `glossary.md:398` (graph_stochastic "not yet registered"; it is, at `ml/models/registry.py:88`);
  - glossary §3's owner for `train.py` disagrees with CLAUDE.md's 15 Oct agreement;
  - `train.py:560`'s pinned base-loss text says the KL term applies to "BLIP only", which GraphStochastic now makes false.
- *Ticket:* none found (`train.py` text: SCRUM-24).
- *Fix:* edit the docs. Changing the pinned loss text needs the same migration as item 8, or every existing record mismatches.

### Checked and found correct

- **Forces:** all seven configs give F = −dE/dpos.
  - Finite differences in float64 on a real ethanol frame agree to relative error ≤ 1e-6 (DimeNet++ with perturbed weights, since its output layer initialises to zero).
  - Net force is ≈ 0, and no model's energy reads the dataset's precomputed `edge_attr`.
- **Randomness:**
  - BLIP is batch-independent: molecule 0 alone and in a batch of 8 gives identical MC mean and std over 300 samples, so the earlier batch-coupling bug is fixed.
  - GraphStochastic is batch-independent: std ratio 0.985 over 2,000 samples, within one standard error.
  - Both are reproducible under a fixed seed.
- **Leakage:** `open_split` refuses the test split and checks molecule and theory against the gold file (`train.py:124-140`). Preprocessing asserts disjoint splits (`preprocessing.py:140-142`). Contiguous blocks come from `trajectory_block_split` (`preprocessing.py:107-112`).
- **Metrics:** `MetricAccumulator` accumulates float64 sums across the whole split, never per batch.
- **UQ formulas:** on synthetic data, a calibrated Gaussian gives coverage 0.498 / 0.799 / 0.900 / 0.950 and ECE 0.001. The closed-form CRPS matches Monte Carlo, the NLL matches by hand, Spearman separates informative from shuffled σ, and variance is converted to std correctly. (ECE here is the provisional coverage-gap definition, pending sign-off, SCRUM-39.)
- **Config:** unknown keys fail loudly at every level (`config.py:97-104`).

## 4. Not checked, and why

- **BLIP signed energy error (item 3):** no `phase2_blip*.pt` on this machine, and `--save-predictions` (PR #9) not on `main`. Needs the `phase2_blip*.pt` checkpoints (SCRUM-21).
- **Re-scoring the retrained Phase 1 models (item 7):** the local checkpoints are the pre-retrain ones. Needs the 19 Oct checkpoints.
- **BLIP trained-checkpoint physics checks:** 14 skips, for the same missing checkpoint.
- **The backend under its real dependencies:** `pydantic-settings` is not in the `.venv`. A scratch-only stand-in was used and nothing was installed.
- **Phase 3 noise dynamics during training (5c):** checked at initialisation only. No Phase 3 run exists.
- **Line-by-line read of every file:** the sub-agent sweep did not run (session limit). The sweep covered every Part 2 category with targeted checks and the evidence above, but not every function body. `ml/models/painn.py`, `physnet.py` and `dimenet.py` internals beyond forces, cutoff and batch independence were not read. `frontend/` was out of scope.

## Status on 31 Oct 2026 (main at `46b5f99`)

Appended on 31 Oct; everything above is the 22 Oct text, unedited. Since `fdc2353`, `ml/training/train.py`, `benchmark.py`, `config.py`, every model file, `ml/data/` and the core tests are unchanged on `main`; `evaluate.py` changed and was reverted (`a1c2ee8`, `6f38928`). So findings in that code are still open by construction. Commit `e332e1d` (SCRUM-70) is on `origin/alex`, **not on `main`**; where it changes a finding, that is noted below.

| # | Status on 31 Oct |
|---|---|
| 1 | **Partly fixed.** `phase2_blip_nll` now passes the benchmark: `7141609` re-scored it so its record carries the eval MC count. `phase2_blip` is still refused (`git_dirty=True` at `e581be8`), and it also still has no eval MC count. SCRUM-69. |
| 2a | **Still open** on `main`. **Also applies to Phase 3 once `e332e1d` lands:** `phase3_graph_stochastic_nll.yaml` (`train_num_mc_samples: 4`, `nll_warmup_epochs: 20`) takes the same `train.py` branch, so the floor is set once, at epoch 20, from validation MSE. For reference, BLIP's floor was 610× its final squared energy error and 83× its force error. |
| 2b | **Still open** on `main`. **`e332e1d`:** Phase 3 scoring adds no floor either (`ScaledEnergyModel.predict_distribution`, `GraphStochastic.predict_distribution`), and training uses the biased variance over K = 4 while scoring uses the unbiased variance over K = 20. Both of BLIP's mismatches carry over. |
| 2c | **Still open** on `main`. **`e332e1d`:** after epoch 20, Phase 3's `*_energy_mse`/`*_force_mse` history columns will hold NLL values. |
| 3 | **Can't check.** There's no `phase2_blip*.pt` on this machine, and the per-frame export (`a1c2ee8`, PR #9) was reverted by `6f38928`. |
| 4 | **Still open** (`train.py:309-313` unchanged). |
| 5a | **Still open** (`graph_stochastic.py` unchanged on `main`; `e332e1d` doesn't change the KL scaling). The scored `phase3_graph_stochastic` run (`7141609`) was trained with `kl_weight` 0.01 under this scaling. |
| 5b | **Fixed on `origin/alex` by `e332e1d`, not on `main`.** It adds `predict_training_distribution` and the `train_num_mc_samples`/`nll_warmup_epochs` hooks, and `ScaledEnergyModel.supports_training_distribution` is simply "has that method". This is what brings 2a, 2b, 2c and N1 to Phase 3. |
| 5c | **Still open** on `main`, now with trained-run evidence: `phase3_graph_stochastic`'s 90% intervals cover 99.5% of test energies but only 57.4% of force components (its `test_metrics.json`). `e332e1d` targets this with NLL training (SCRUM-70); no result yet. |
| 5d | **Latent on `main`; live on `origin/alex`.** In `e332e1d`, `predict_training_distribution` runs K forward passes, each overwriting `_last_kl` (`graph_stochastic.py:286`), and `batch_loss` then reads only the last one (`train.py:311`). `phase3_graph_stochastic_nll.yaml` sets `kl_weight: 0.01` with K = 4, the combination the new docstring calls "worth revisiting". |
| 6 | **Partly fixed by `b8a8ab7`:** `/benchmarks` maps `benchmark_row` (no `KeyError`), and inference is real, not the stub. **Still open:** it reads the gitignored `experiments/results/<checkpoint>.json`, not the committed run records. On this machine those files are older scorings: `phase1_baseline` energy MAE 0.0891 against 0.0779 in its run record, with a different data fingerprint. `pydantic-settings` is now installed in this machine's `.venv` (environment only). |
| 7 | **Still open.** The same five local Phase 1 checkpoints with the same mismatched sha256; no Phase 2/3 checkpoints here. `_trained_model` in `tests/ml/test_painn_physics.py` still checks only existence and version. |
| 8 | **Still open.** `experiments/runs/phase2_blip_nll/run.json` is unchanged since `fdc2353` and records no migration. (`7141609` re-scored the same run's `test_metrics.json` through `evaluate.py`.) |
| 9 | **Partly fixed:** `backend/tests/test_api.py` has 16 test functions (23 collected) since `b8a8ab7`. **Still open:** `tests/test_md17.py` and `tests/test_models.py` are 0 bytes; no test references `silver_to_gold`, `bronze_to_silver`, `validate_sample` or `load_split_zip`; `build_graph`'s edge set, symmetry and cutoff are untested; `tests/test_dimenet.py` still has no skip guard and still imports `ml.training.train_dimenet`. |
| N1 | **Still open** on `main`. **`e332e1d`:** applies to Phase 3 NLL. The floor comes from validation MSE, and the LR schedule and early stopping run on a `val_loss` that includes KL (`kl_weight: 0.01`), with item 4's per-loader scaling. |
| N2 | **Still open** (`phase2_blip.yaml` and the `phase2_blip` record unchanged). It bears on the SCRUM-69 retrain, which uses that config. |
| N3 | **Still open** (`md17.py:242` `<=`; no envelope in MPNN, BLIP, GraphStochastic or SchNet). It matters more since `dc0dfcc`: `POST /jobs/train` trains any of the 7 architectures on any molecule with gold data, including paracetamol and azobenzene. |
| N4 | **Still open.** `train_schnet.py`, `train_physnet.py` and `train_dimenet.py` are unchanged on `main`. |
| N5 | **Fixed by `b8a8ab7`:** no `baseline.pt` default; the catalogue is read from the gold files; an unknown molecule gets a 404 (tested). The backend's requirements are installed in this machine's `.venv` (environment only); the repo has no CI. |
| N6 | **Still open** (`train.py` unchanged). |
| N7 | **Still open** (`config.py` unchanged). |
| N8 | **Still open** on `main`: `CLAUDE.md:19` (corrected on the unmerged `fazin-uq-protocol-v1`, `319425a`); `glossary.md:4-5`; the "not yet registered" line, now `glossary.md:454`; glossary §3's `train.py` row still disagrees with `CLAUDE.md`; `train.py:560`'s pinned "BLIP only" text (changing it needs item 8's migration). |

**Also noted since 22 Oct:** where a trained checkpoint exists, 6 of `backend/tests/test_api.py`'s tests fail. Three fail because `TestClient(app)` is built without `with`, so `init_db()` never runs; three lack a skip guard for missing aspirin/azobenzene/paracetamol gold data. On a clean clone the module-level `skipif` skips all 23. Section 3's "Checked and found correct" covers code that is unchanged on `main`; it wasn't re-run.
