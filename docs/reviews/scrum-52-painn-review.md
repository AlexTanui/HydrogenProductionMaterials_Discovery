# Review: PaiNN Phase 1 bake-off entry (SCRUM-52)

| | |
|---|---|
| **Scope** | `ml/models/painn.py`, `experiments/configs/phase1_painn.yaml`, `docs/model_cards/painn.md`, and how they fit the shared harness (`ml/training/train.py`, `ml/config.py`, `ml/models/registry.py`, `ml/training/evaluate.py`, `ml/utils/metrics.py`) |
| **Reviewed commit** | `02def8a` on `fazin`. Every `file:line` below refers to that commit unless marked otherwise |
| **Reference** | Schütt, Unke & Gastegger (2021), *Equivariant message passing for the prediction of tensorial properties and molecular spectra*, ICML; schnetpack's `representation/painn.py` for implementation details |
| **Date** | 2026-10-11 |

## Verdict

PaiNN itself is sound. It matches the paper with no harmful deviations, and
every physics check passes on real TRAIN-split ethanol frames at float64
roundoff. Three other models (SchNet, MPNN, PhysNet) pass the same checks as
positive controls, and six deliberately broken mutants fail them by wide
margins, so the checks are discriminating.

What is not ready is the path around the model. PaiNN cannot currently be
trained by the shared `train.py`, and its energy normalisation convention
differs from `train.py`'s. Wired together as-is, that mismatch would cost
about 97,196 kcal/mol of energy error at evaluation. A short sanity run
shows PaiNN learning well ahead of SchNet under an identical loop, so
nothing points at a model problem.

PaiNN has never been trained, so no benchmark number exists for it yet.

## Fixes applied on `fazin`

| Fix | Commit |
|---|---|
| PaiNN physics and convention tests added (`tests/ml/test_painn_physics.py`, `tests/ml/test_painn_conventions.py`) | `c576fee` |
| H4: `evaluate.py` re-enables grad around the model call | `abdcb17` |
| `painn.py` filter-bias comment corrected (comment only) | `4d19ce7` |
| H8: run instruction in `phase1_painn.yaml` corrected | `9047f8e` |
| H8: `phase1_painn.yaml` header now says the config needs `train.py` support for PaiNN first (H1) | `7f81870` |
| H9: two stray files removed; TensorBoard event files ignored under `experiments/runs/` | `8cff88f` |

Full suite after these commits: 172 passed, 14 skipped, plus 3 errors in
`tests/test_dimenet.py` that predate them (see "Other observations"). The 14
skips are the trained-PaiNN variants, waiting for a checkpoint.

Everything else below needs a team decision first, mainly H1, H2, H3, H5 and
H6, which touch `train.py`, `config.py`, `registry.py` and other people's
models.

---

## Summary of checks

Physics numbers are float64 worst cases over four TRAIN-split ethanol frames
(positions 0, 148,024, 296,049 and 444,073 of `train_idx`). Tolerances:
energy 1e-9 kcal/mol, forces 1e-9 kcal/mol/Å, finite differences 1e-4
relative.

| Check | Status | Evidence |
|---|---|---|
| **Step 1: interface and conventions** | | |
| Method names (`predict_energy`, `predict_energy_and_forces`, `config`) | PASS | match the registry's required interface, [registry.py:68](../../ml/models/registry.py#L68) |
| `predict_energy` signature | WARN | takes `pos` and requires `batch` ([painn.py:333-340](../../ml/models/painn.py#L333-L340)); MPNN and SchNet take `(z, edge_index, edge_attr, batch=None)`. Deliberate: the vector channel needs bond directions. The 4-argument call raises rather than misreading its arguments (`test_shared_four_argument_call_fails_loudly`) |
| `predict_energy_and_forces` signature | PASS | `(z, pos, edge_index, edge_attr, batch)`, same as the other models |
| Shapes, dtypes, units | PASS | E `(B,)` and F `(N,3)` in float32 on a real loader batch, accepted by `contract.py` unchanged; output in the data's kcal/mol and Å, like every other model |
| Energy normalisation matches the other models | **FAIL** | per-atom `energy_shift`/`energy_scale` live inside the model ([painn.py:400](../../ml/models/painn.py#L400)) and no training loop sets them; `train.py` subtracts a total shift outside the model instead ([train.py:78](../../ml/training/train.py#L78), [:165](../../ml/training/train.py#L165)). Not applied twice today only because PaiNN has never been trained (H3) |
| Works with the shared `train.py` | **FAIL** | config rejected (`DataConfig` does not know `num_rbf`); the only builder is `mpnn` (H1) |
| Works with `evaluate.py`, no special-casing | PASS | shift and scale buffers survive registry → `evaluate_checkpoint`, and the MAE matches a direct computation (`test_energy_statistics_survive_registry_and_evaluate`) |
| **Step 2: fidelity to the paper** | | |
| s is F channels, v is 3×F, v starts at 0 | PASS | [painn.py:385-389](../../ml/models/painn.py#L385-L389) |
| Bessel basis with cosine cutoff; filter split three ways; vector message uses r̂ | PASS | [painn.py:116-125](../../ml/models/painn.py#L116-L125), [:173-179](../../ml/models/painn.py#L173-L179) |
| U and V have no bias; scalar update uses ‖Vv‖ and ⟨Uv, Vv⟩ | PASS | [painn.py:203-223](../../ml/models/painn.py#L203-L223) |
| Residual updates; block count and sizes match the config | PASS | [painn.py:391-397](../../ml/models/painn.py#L391-L397); 3 blocks, F = 128, 20 radial functions, 588,033 parameters (`test_config_builds_the_described_architecture`) |
| Readout uses scalars only, summed per molecule with `batch` | PASS | [painn.py:399-407](../../ml/models/painn.py#L399-L407) |
| Epsilon inside vector norms | PASS | `sqrt(Σx² + 1e-8)` ([painn.py:88](../../ml/models/painn.py#L88)); same eps as schnetpack |
| **Step 3: physics** | | |
| 1. Rotation (3 proper rotations + 1 reflection) | PASS | \|ΔE\| = 0, \|F(Qx) − QF(x)\| = 2.8e-16 |
| 2. Translation by (13.5, −7.25, 91) Å | PASS | \|ΔE\| = 0, \|ΔF\| = 0 |
| 3. Permutation, graph rebuilt from the permuted frame | PASS | \|ΔE\| = 1.5e-11 (1 ulp at 9.7e4), \|ΔF\| = 7.6e-17 |
| 4. Finite differences (h = 1e-4 Å, graph rebuilt at each displacement) | PASS | relative error 6.5e-6 |
| 5. Net force | PASS | \|ΣF\| = 1.9e-17 |
| 6. Batching through the real PyG `DataLoader` (batch sizes 2 and 4) | PASS | \|ΔE\| = 0, \|ΔF\| = 6.2e-17 |
| 7. Gradient flow from `pos` | PASS | zeroed, noisy or missing `edge_attr` changes E and F by exactly 0; with `edge_attr` frozen, moving one atom 0.05 Å changes E by 5.8e-4; loader-fed forces match finite differences; the loader's `edge_attr` has `grad_fn=None` (pinned) |
| Positive controls (SchNet, MPNN, PhysNet) | PASS | all checks pass at the same tolerances |
| Negative controls | PASS (each fails its check as designed) | see "Negative controls" below |
| Trained model | SKIP | no checkpoint exists; 14 tests skip automatically |
| Precision at float32 (for information) | — | float32 vs float64 energy differ by up to 0.006 kcal/mol (one float32 step at 97,196 is 0.0078); float32 rotation force error is 1.5e-6 relative |
| **Step 4: training setup** | | |
| `data:` section identical to the other configs | WARN | every shared key has the same value (test passes), but PaiNN adds `data.num_rbf: 16` ([phase1_painn.yaml:41](../../experiments/configs/phase1_painn.yaml#L41)) (H7) |
| Loss weights 1.0 / 100.0, seed 42 | PASS | all Phase 1 configs |
| Test split never used for training or selection | PASS | PaiNN has no trainer; the smoke script reads train only ([smoke_train_painn.py:62-71](../../scripts/smoke_train_painn.py#L62-L71)); `train.py` selects checkpoints on val ([train.py:211-225](../../ml/training/train.py#L211-L225)) |
| Shared training budget | **FAIL** | see H6 |
| PaiNN at least as good as SchNet | PASS, sanity check only | see "Sanity check" below |

### Negative controls

Each mutant breaks exactly one property and must fail the matching check by
at least 1,000× its tolerance. All do.

| Mutant | What it breaks | Result |
|---|---|---|
| Bias on U (vector channel) | rotation | ΔE = 2.2e-3, ΔF = 1.2e-2 |
| Term linear in absolute position | translation, net force | ΔE = 65 under translation; net force 0.72 |
| Atom-index-weighted term | permutation only | ΔE = 2.3, ΔF = 1.9 under permutation; rotation still passes |
| Radial filter reads the dataset's detached `edge_attr` | gradient path | **passes every symmetry check**; finite-difference error 26× the force; cannot run without `edge_attr` |
| Energy computed from `edge_attr` only | gradient path | zero forces; finite-difference error 1.0 |
| Readout ignores `batch` | batching | ΔE = 2.9e5 |

The stale-`edge_attr` mutant is the reason checks 4 and 7 exist: that bug
is invisible to rotation, translation and permutation tests. It only shows
when finite differences are taken over a graph rebuilt at each displaced
geometry.

### Deviations from the paper

None are harmful.

1. **No bias on `filter_net`** (harmless). schnetpack's `snn.Dense` defaults
   to `bias=True` with zero initialisation; the cutoff multiplies the filter
   afterwards in both codebases. The comment that attributed this choice to
   the reference implementation is corrected in `4d19ce7`.
2. **A √(2/r_c) factor on the Bessel basis** (harmless). The linear filter
   absorbs it; only the initial scale changes.
3. **eps inside the distance norm** (harmless). The resulting bias is at
   most 5e-9 Å, below float32 resolution.
4. **Xavier initialisation on the embedding**, where schnetpack uses N(0,1)
   (harmless; initialisation only).
5. **One filter layer per block**, where schnetpack uses one layer for all
   blocks by default (harmless; equivalent).
6. **Per-atom shift and scale as buffers set by the caller** (deliberate
   simplification). schnetpack does the same with ScaleShift buffers; the
   problem is the mismatch with the harness (H3).
7. **`pos` in the `predict_energy` signature** (deliberate).

One consequence of item 6: `energy_scale` also scales the forces
(F = −scale · ∂Σy/∂R), which `test_painn_conventions.py` pins.

---

## Results

There are no results JSONs or checkpoints in the repo, and no row below is
comparable with another. Numbers are taken from each model card.

| Model | Energy MAE | Force MAE | Energy RMSE | Force RMSE | Train time | Inference | Provenance |
|---|---|---|---|---|---|---|---|
| PaiNN | — | — | — | — | — | — | never trained |
| SchNet | 0.0873 | 0.1114 | 0.1026 | 0.1628 | 30,029 s | 0.163 ms/frame | **test** split; all 444k train frames, batch 256, `train_schnet.py` |
| PhysNet | 2.082 | 0.711 | 2.104 | 1.094 | 3,425 s | 2.96 ms/config | **val** split; 20k-frame subset, 20 epochs, **L1 loss** |
| DimeNet++ | 1.607 | 1.530 | 1.736 | 2.140 | — | — | **CCSD(T)** ethanol, a different dataset; 889 train frames |
| MPNN | — | — | — | — | — | — | no results and no model card |

Units: energy in kcal/mol, forces in kcal/mol/Å (component convention).

### Sanity check (not a benchmark number)

All three arms used the same loop, a copy of `train.py`'s recipe: Adam
lr 5e-4, weight decay 1e-6, gradient clipping 10, `ReduceLROnPlateau(0.5, 5)`
on val loss, MSE loss weighted 1/100, batch 32, seed 42. Training used the
first 2,000 frames of `train_idx` for 15 epochs. Scoring used the first
5,000 frames of `val_idx` at the final epoch, with no checkpoint selection.
The test split was never constructed. The script was a throwaway outside the
repo; this paragraph is its full specification.

| Arm | Val energy MAE | Val force MAE | Energy RMSE | Force RMSE | Training | Inference | Parameters |
|---|---|---|---|---|---|---|---|
| SchNet (its config, 64 channels) | 11.96 | 5.16 | 12.16 | 7.16 | 42 s | 0.13 ms | 61,569 |
| PaiNN, its own per-atom statistics | **0.83** | **2.00** | 1.02 | 2.91 | 105 s | 0.48 ms | 588,033 |
| PaiNN, `train.py`'s centred-target convention | 1.46 | 1.95 | 1.73 | 2.93 | 108 s | 0.50 ms | 588,033 |

- PaiNN's force error is 2.6× lower than SchNet's under either convention.
- The normalisation choice moves PaiNN's energy MAE (0.83 vs 1.46), so H3
  must be settled before any real training.
- Caveats: one seed, about 10× more parameters than SchNet at its config
  width, and batch 32 for both models (SchNet's config uses 256).

---

## Harness findings

Each finding has `file:line` evidence at `02def8a` and a suggested minimal
fix. H4, H8 and H9 are fixed on `fazin`; the rest are open pending a team
decision (one shared `train.py` vs per-model trainers).

### H1. `train.py` cannot train PaiNN, SchNet or PhysNet (open)

- `_MODEL_BUILDERS` contains only `mpnn` ([train.py:41-48](../../ml/training/train.py#L41-L48)).
- `config.py` has a fixed phase list ([config.py:38-40](../../ml/config.py#L38-L40)) and rejects unknown keys ([config.py:67-71](../../ml/config.py#L67-L71)).
- `phase1_painn.yaml` fails on `data.num_rbf`, `model.cutoff_radius`/`eps`/`max_atomic_number` and `train.train_subset*`. The SchNet and PhysNet configs fail too; only `phase1_baseline.yaml` loads.
- *Minimal fix (works for either team decision):* add `model.name` plus a pass-through `model.kwargs` to `config.py`, and build models with `registry.build_model(cfg.model.name, **kwargs)`.

### H2. `train.py` and `evaluate.py` use different checkpoint formats (open)

- `train.py` writes `{model_state_dict, energy_shift, config}` ([train.py:217-224](../../ml/training/train.py#L217-L224), [:237-241](../../ml/training/train.py#L237-L241)).
- `evaluate.py` requires `phase`/`model_config`/`state_dict` ([registry.py:145](../../ml/models/registry.py#L145)), so it rejects every `train.py` checkpoint.
- *Fix:* `train.py` calls `registry.save_checkpoint` ([registry.py:99](../../ml/models/registry.py#L99)).

### H3. Two energy-normalisation conventions are in use (open)

- The shift lives outside the model in `train.py` ([:78](../../ml/training/train.py#L78), [:123](../../ml/training/train.py#L123), [:165](../../ml/training/train.py#L165)), `train_schnet.py` and `train_dimenet.py` ([:40-41](../../ml/training/train_dimenet.py#L40-L41)).
- It lives inside the model in PaiNN ([painn.py:400](../../ml/models/painn.py#L400)) and PhysNet ([physnet.py:297](../../ml/models/physnet.py#L297)).
- `evaluate.py` adds nothing back ([evaluate.py:195-204](../../ml/training/evaluate.py#L195-L204)). Any model with an outside shift therefore scores about 97,196 kcal/mol off.
- PaiNN run through `train.py` as-is would hit the same problem. If a trainer both subtracts the mean and sets PaiNN's buffers, the shift is applied twice.
- *Fix:* the model owns the shift, as PaiNN and PhysNet do, so neither `evaluate.py` nor `backend/app/services/inference.py` needs a side channel. `train.py` then stops subtracting.

### H4. `evaluate.py`'s `torch.no_grad()` broke MPNN and SchNet (fixed, `abdcb17`)

- Reproducing the exact call from [evaluate.py:190-197](../../ml/training/evaluate.py#L190-L197), both models raised `RuntimeError: element 0 of tensors does not require grad` on the first batch. They failed loudly, not with silently wrong numbers.
- Cause: they call `pos.requires_grad_(True)` without `enable_grad` ([mpnn.py:92](../../ml/models/mpnn.py#L92), [schnet.py:148](../../ml/models/schnet.py#L148)). PaiNN and PhysNet re-enable grad internally and were unaffected.
- Fix: only the `predict_energy_and_forces` call is wrapped in `torch.enable_grad()`; the rest of the loop stays under `no_grad`.
- Verified with SchNet and MPNN registered in memory and scored through the real `evaluate_checkpoint`: both raised before the fix and score after it. Forces come back with `grad_fn=None`, so no graph is retained, and PaiNN/PhysNet results are unchanged.
- The bug stays latent in practice until H5, because MPNN and SchNet cannot be loaded by `evaluate.py` yet.

### H5. The registry contains only `painn` (open)

- [registry.py:61-63](../../ml/models/registry.py#L61-L63). MPNN and SchNet lack `config()`. PhysNet has it but is not registered ([train_physnet.py:100-101](../../ml/training/train_physnet.py#L100-L101) prints a note saying so).
- *Fix:* add `config()` to MPNN and SchNet, and register all four.

### H6. There is no shared training budget (open)

- `train_subset: 10000` ([phase1_painn.yaml:87-88](../../experiments/configs/phase1_painn.yaml#L87-L88)) is not read by any code.
- SchNet trained with batch 256 ([phase1_schnet.yaml:34](../../experiments/configs/phase1_schnet.yaml#L34)).
- PhysNet used a 20k-frame contiguous prefix ([train_physnet.py:60](../../ml/training/train_physnet.py#L60)) and an **L1** loss ([:130](../../ml/training/train_physnet.py#L130)); everything else uses MSE ([train.py:79-80](../../ml/training/train.py#L79-L80)).
- DimeNet++ trained on CCSD(T) ethanol with 889 frames.
- *Fix:* one subset, one batch size and MSE everywhere, all in the shared loop. The subset is the deterministic stride over `train_idx` defined in `phase1_painn.yaml` ([:79-89](../../experiments/configs/phase1_painn.yaml#L79-L89), current lines): `train_subset: 10000` frames at `train_subset_seed: 42`. Until `train.py` implements it, the reference is `stride_subset` in [scripts/train_painn.py:106](../../scripts/train_painn.py#L106) (added in `29a17a8`). It uses stride 444,074 // 10,000 = 44 and a seed-42 offset within the first stride (3), giving positions 3, 47, …, 439,959, spread evenly over the train block. It records those positions in the checkpoint so every model can reuse exactly the same frames. PhysNet's 20k contiguous prefix and DimeNet++'s 889 CCSD(T) frames do not match it, so both need re-running on ethanol/DFT with this subset.

### H7. `data.num_rbf` exists only in PaiNN's config (open)

- [phase1_painn.yaml:41](../../experiments/configs/phase1_painn.yaml#L41). It is content-identical to the loader default, but it makes the `data:` block differ from the other configs and is one of the keys `config.py` rejects.
- `train.py` sizes the dataset's `edge_attr` from `model.num_rbf` ([:65](../../ml/training/train.py#L65)), while `evaluate.py` uses the checkpoint's `data.num_rbf` ([:156](../../ml/training/evaluate.py#L156)): two sources for one value.
- *Fix:* remove the key, or add `num_rbf` to `DataConfig` and use it at `train.py:65`.

### H8. Wrong run instruction in the PaiNN config (fixed, `9047f8e`, `7f81870`)

- [phase1_painn.yaml:10](../../experiments/configs/phase1_painn.yaml#L10) showed `--config`, but `train.py` takes the config path positionally ([train.py:142](../../ml/training/train.py#L142)).
- Line 9 said the config was "Runnable as-is once ml/training/train.py lands", but `train.py` has landed and still cannot load it (H1). It now says the config needs `train.py` support for PaiNN first and points here (`7f81870`).

### H9. Stray files in commit `526ab8f` (fixed, `8cff88f`)

Commit `526ab8f` (message ".", on `fazin` only) added two files that were
never meant to be tracked.

- **`experiments/runs/painn_seed0/events.out.tfevents.1790210534.Fazinzz.37124.0`.**
  It holds only the header record `torch.utils.tensorboard.SummaryWriter`
  writes when it opens (host `Fazinzz`, PID 37124, 2026-09-24 10:12 ACST),
  with no scalars. Local shell history on that machine shows a
  `SummaryWriter("experiments/runs/painn_seed0")` snippet, then
  `pip install tensorboard`, then a run of `ml/training/dfsdf.py`. That
  scratch script was never committed and no longer exists. It is almost
  certainly the writer, and its loop failed before logging anything.
- **`tic evaluate harness, equivariance and gradient-path tests"`**, a
  colour-coded `git diff` of `evaluate.py`. Most likely a commit command was
  typed into the `less` pager of a preceding `git diff --cached`: `s` (the
  first lowercase "s", in "agnostic") is less's save-to-file key, so the rest
  of the message became the filename.

Both are removed. `.gitignore` now ignores
`experiments/runs/**/events.out.tfevents.*` only, not the directory, which
will hold SCRUM-48 run records.

### Other observations, for the owners

- **PhysNet (Shijin):** its two entry points disagree by 0.070 kcal/mol (untrained), because `predict_energy` drops the cosine cutoff ([physnet.py:336](../../ml/models/physnet.py#L336) vs [:363](../../ml/models/physnet.py#L363)).
- **Data layer (Shijin):** gold energies are stored as float32, and `MD17Dataset` loads them as float32 ([datasets.py:31](../../ml/data/datasets.py#L31)). At 97,196 kcal/mol that is a 0.0078 kcal/mol resolution floor for every model. It matters only once energy MAEs approach about 0.01.
- **DimeNet++ tests (Dongxiao):** `tests/test_dimenet.py` errors rather than skipping when `data/gold/md17/ethanol_ccsd_t.npz` is absent. Those are the 3 errors in a full `pytest` run.
- **MPNN and SchNet (Ruturaj, Alex):** `pos.requires_grad_(True)` mutates the caller's tensor in place ([mpnn.py:92](../../ml/models/mpnn.py#L92), [schnet.py:148](../../ml/models/schnet.py#L148)).

---

## Prioritised fixes, smallest first

| # | Fix | Changes numbers / needs retraining? | Status |
|---|---|---|---|
| 1 | `enable_grad` around the model call in `evaluate.py` (H4) | No | Done, `abdcb17` |
| 2 | Correct the filter-bias comment in `painn.py` | No | Done, `4d19ce7` |
| 3 | Correct the run instruction in the PaiNN config (H8) | No | Done, `9047f8e` and `7f81870` |
| 4 | Remove the stray files; ignore TensorBoard event files (H9) | No | Done, `8cff88f` |
| 5 | Remove `data.num_rbf`, or add it to `DataConfig` and use it at `train.py:65` (H7) | No | Open |
| 6 | Add `config()` to MPNN and SchNet; register all four models (H5) | No | Open, team decision |
| 7 | `train.py` writes registry checkpoints (H2) | No | Open, team decision |
| 8 | Model selection by name in `config.py`/`train.py` (H1) | No | Open, team decision |
| 9 | **Agree one energy-normalisation convention** (H3) | **Yes:** moves PaiNN's energy MAE (0.83 vs 1.46 in the sanity run). Settle before PaiNN's first training run | Open, team decision |
| 10 | **Shared budget, batch size and loss** (H6) | **Yes:** all five models need (re)training. SchNet's 0.087 / 0.111 is not comparable until then | Open, team decision |

PaiNN has never been trained, so "retraining" for PaiNN means its first run
has to come after fixes 9 and 10.

## Reproducing

```bash
pytest tests/ml/test_painn_physics.py      # 65 pass, 14 skip (trained variant)
pytest tests/ml/test_painn_conventions.py  # 13 pass
pytest                                     # full suite
```

`test_painn_physics.py` needs `data/gold/md17/ethanol_dft.npz`; run
`python -m ml.data.preprocessing` if it is missing. Its trained-model
variants run automatically once `experiments/checkpoints/phase1_painn.pt`
exists.
