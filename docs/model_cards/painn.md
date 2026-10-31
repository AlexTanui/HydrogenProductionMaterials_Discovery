# Model card — PaiNN (Phase 1 bake-off candidate)

**Jira:** SCRUM-52, under the Phase 1 bake-off epic SCRUM-50
**Owner:** Fazin Faizal (QA & Benchmarking Engineer)
**Status:** trained and scored under the Phase 1 protocol v1 (§7)
**Code:** [`ml/models/painn.py`](../../ml/models/painn.py)
**Config:** [`experiments/configs/phase1_painn.yaml`](../../experiments/configs/phase1_painn.yaml)
**Tests:** [`tests/ml/test_painn.py`](../../tests/ml/test_painn.py)

> **Results:** §7 holds the Phase 1 protocol v1 result, trained by the
> shared `ml/training/train.py` and scored once on the test split by
> `ml/training/evaluate.py`. The smoke-run numbers in §6 are not results.

---

## 1. What this model is

PaiNN — Polarizable Atom Interaction Neural Network (Schütt, Unke &
Gastegger, 2021). An **equivariant** message-passing network in which each
atom carries a paired feature set:

| Feature | Shape | Behaviour under rotation `R` |
|---|---|---|
| scalar `s_i` | `[F]` | invariant |
| vector `v_i` | `[3, F]` | equivariant (`v -> R v`) |

It predicts a per-atom energy contribution from the scalar channel, sums
them to a total energy, and obtains forces as `F = -∂E/∂R` by
differentiating that energy with respect to atomic positions. There is no
separate force head, so the model is a **conservative potential** by
construction.

### Why it is a bake-off candidate

Phase 1's default backbone (glossary.md §3) is a Gilmer-style MPNN over
RBF-expanded *distances*. Distances are rotation-invariant, so such a model
cannot represent directional information — it sees bond lengths but not
bond angles. PaiNN's vector channel does, at modest extra cost. The bake-off
exists to measure whether that buys enough accuracy on MD17 to justify
carrying it into Phases 2 and 3.

---

## 2. Architecture

```
embed:  s_i = Embedding(z_i)          v_i = 0
radial: Bessel(num_rbf) x cosine cutoff, recomputed from `pos`
num_layers x [ MessageBlock -> UpdateBlock ]
readout: Linear(F -> F/2) -> SiLU -> Linear(F/2 -> 1), summed over atoms
forces:  -autograd.grad(E.sum(), pos)
```

**MessageBlock** (interatomic):

```
ds_i = sum_j  m_ss
dv_i = sum_j [ v_j * m_vv  +  m_vs * r_hat_ij ]
(m_ss, m_vv, m_vs) = split( phi(s_j) * W(r_ij) )
```

**UpdateBlock** (per-atom):

```
dv_i = a_vv * U v_i
ds_i = a_ss + a_sv * <U v_i, V v_i>
(a_vv, a_sv, a_ss) = MLP([ s_i, ||V v_i|| ])
```

### The four rules that make it equivariant

1. Vectors reach scalars **only** through rotation-invariant contractions —
   `||v||` and `<u, v>`.
2. Vectors are **only** produced by scaling a vector by a scalar (an
   existing `v_j`, or the unit bond direction `r_hat_ij`).
3. **No bias on any linear map applied to a vector channel.** A bias adds a
   fixed vector, and `R(Wv + b) != R(Wv) + b` unless `b = 0`. This is the
   classic silent break — the model still trains and still looks plausible.
4. Only **relative** displacements `r_ij = pos_j - pos_i` are read, never an
   absolute coordinate, so translation invariance is structural.

Sum-aggregation over neighbours gives permutation equivariance; sum-pooled
per-atom energies give a size-extensive total.

### Two implementation choices worth knowing

**The dataset's `edge_attr` is deliberately ignored.** `MD17Dataset.get()`
builds it via `pos.numpy()`, so it arrives detached from the autograd graph
(`grad_fn=None`, verified in `test_dataset_edge_attr_is_detached_from_pos`).
A model that took its radial features from `edge_attr` would compute
`-∂E/∂pos` instead of the full `-(∂E/∂pos + ∂E/∂rbf · ∂rbf/∂pos)`, and would
report forces that are **not** the gradient of the energy it predicts — while
still training and still producing believable MAEs. PaiNN therefore
recomputes its own radial basis from `pos`. This applies to every
conservative model in the bake-off, not only this one.

**Norms are floored (`sqrt(sum(x^2) + eps)`).** `v_i` is exactly zero for an
atom with no neighbours in range, and wherever neighbour directions cancel —
every centrosymmetric environment. A bare `sqrt(sum(x^2))` returns a NaN
gradient there; `torch.linalg.norm` returns zero at the origin but 1.0 one
ulp away, and that discontinuity shows up as a spurious force on a perfectly
balanced atom. The floor does not affect equivariance at any `eps`, because
`sum(x^2)` is itself rotation-invariant.

---

## 3. Data

| | |
|---|---|
| Dataset | MD17, ethanol, **DFT** (theory level is part of the dataset identity) |
| Gold file | `data/gold/md17/ethanol_dft.npz` |
| Atoms | 9 |
| Configurations | 555,092 |
| Split | 444,074 train / 55,509 val / 55,509 test |
| Split method | Contiguous trajectory blocks (80/10/10), baked into the gold file as `train_idx`/`val_idx`/`test_idx`. Deterministic, no RNG. |
| Graph | 5.0 Å cutoff, neighbour list rebuilt per frame |
| Units | energy kcal/mol, length Å, force kcal/mol/Å |

Splits are **contiguous blocks, never random** — adjacent MD frames are
near-duplicates, so a random split leaks near-identical structures across
train and test and inflates apparent accuracy. `ethanol_dft` ships no
literature-standard split, so the trajectory-block split applies. Any
evaluation that re-splits gold data is a bug.

The test split is read in exactly one place: `ml/training/evaluate.py`.
Not for early stopping, not for monitoring, not for a sanity check.

---

## 4. Hyperparameters

| Parameter | Value | Note |
|---|---|---|
| `hidden_channels` | 128 | PaiNN's MD17 setting |
| `num_layers` | 3 | message+update blocks |
| `num_rbf` | 20 | PaiNN's own Bessel basis, separate from the dataset's 16-bin Gaussian `edge_attr` |
| `cutoff_radius` | 5.0 Å | must equal `data.cutoff_radius` |
| `max_atomic_number` | 100 | embedding table size |
| `eps` | 1e-8 | floor inside every vector norm |

Shared across all five bake-off models (do not vary):

| Parameter | Value |
|---|---|
| `seed` | 42 |
| `epochs` | 200 |
| `batch_size` | 32 |
| `lr` | 5e-4 |
| `energy_loss_weight` | 1.0 |
| `force_loss_weight` | 100.0 |
| `patience` | 20 |

`energy_shift` / `energy_scale` are **buffers**, not parameters. Ethanol's
total energies sit near −97,000 kcal/mol, which no freshly-initialised
readout can reach; the training loop must set them from the *training*
split's statistics before the first step. Being buffers, they travel in the
checkpoint and evaluation reproduces the training scale exactly.

---

## 5. Correctness verification

`tests/ml/test_painn.py` — 32 tests, all passing. Accuracy numbers cannot
detect a broken equivariant model; these assertions can.

| Property | Assertion |
|---|---|
| Rotation | `E` unchanged, `F -> R F`, over 6 random SO(3) matrices |
| Translation | `E` and `F` unchanged under a large rigid shift |
| Permutation | `E` unchanged, `F` permutes with the atoms |
| Conservative field | `F` equals `-autograd.grad(E, pos)` computed independently |
| Conservative field | `F` matches central finite differences of `E` (independent code path; ROADMAP.md Phase 1 DoD) |
| Batching | N molecules batched == N single passes, including unequal sizes |
| Gradient path | `pos.requires_grad` honoured; every atom receives a non-zero gradient |
| Size extensivity | two non-interacting copies in one graph have exactly 2× the energy |
| Newton's third law | forces sum to zero |
| Centrosymmetry | the central atom of a symmetric pair feels *exactly* zero force |
| Angular sensitivity | energy differs at 90° vs 120° with bond lengths held fixed |
| Cutoff continuity | no energy step as a pair crosses 5 Å |
| Message direction | a directed single edge distinguishes neighbour from centre |
| `edge_attr` isolation | corrupting or omitting `edge_attr` changes nothing |
| Loader contract | a real `MD17Dataset` PyG batch is accepted; outputs satisfy `ml/utils/contract.py` |

Tests run in **float64**, where the roundoff floor is ~1e-12 and the
tolerance is 1e-10. At float32 a rotation test needs ~1e-5, which is loose
enough to pass a model broken at the 1e-4 level.

Equivariance was separately confirmed at the **float32** precision training
actually runs in (`hidden_channels=128`, `num_layers=3`): energy exactly
invariant under rotation, forces equivariant to a relative error of 1.2e-6 —
float32 epsilon, not a modelling error. float64 gives 2e-15.

### Mutation testing

The suite was validated by deliberately breaking the model 13 ways and
confirming the right test fails. Ten mutations were caught. Three were not,
and each was investigated:

| Mutation | Caught by |
|---|---|
| Bias on the vector channel | rotation, centrosymmetry |
| Absolute instead of relative position | translation, size extensivity, force sum |
| Mean-pool instead of sum-pool readout | size extensivity |
| Scatter into the neighbour, not the centre | message direction |
| Gather from the centre, not the neighbour | message direction |
| `edge_attr` allowed into the energy | `edge_attr` isolation |
| `torch.linalg.norm` for the vector norm | centrosymmetry |
| Readout ignores the `batch` vector | batching (3 tests) |
| Vector message drops `r_hat` | angular sensitivity |
| Forces scaled by 1.0001 | conservative field, finite difference |
| Cutoff envelope removed | cutoff continuity |
| Bias added to `filter_net` | *not caught — and correctly so:* the envelope multiplies afterwards, so a bias there is harmless. The code comment claiming otherwise was wrong and has been corrected. |
| Readout also reads `\|\|v\|\|` | *not caught — not a bug:* still rotation-invariant, an architecture variant rather than a defect. |

Two genuine test gaps were found and closed this way: the original
size-extensivity test used two *separate* graphs (so mean-pooling divided
both equally and passed), and the original message-direction test let the
bond distance vary (so the radial filter alone made the energies differ).

---

## 6. Compute profile (SCRUM-49)

Measured on CPU (torch 2.13.0+cpu, Python 3.11.9, Windows 11), 9-atom
ethanol, `hidden_channels=128`, `num_layers=3`, `batch_size=32`.

| Quantity | Value |
|---|---|
| Parameters | ~590 k |
| Training step (fwd + double-backward + Adam) | 37 ms |
| Training throughput | ~780 frames/s |
| Inference (fwd + force backward) | 0.40 ms/config |
| Peak RSS | ~630 MB (+170 MB over baseline) |

Projected training wall-clock at 200 epochs:

| Train frames/epoch | Estimated total |
|---|---|
| 10,000 (config default) | ~45 min |
| 444,074 (full split) | ~32 h |

**No CUDA is available on the current dev machines.** The full train split
at 200 epochs is not reachable, so `train.train_subset: 10000` caps the
number of *training* frames; val and test are always used whole. The subset
is deterministic at seed 42 and **must be applied identically by all five
bake-off models** — that shared budget is what keeps the comparison fair.
For reference, the MD17 literature convention (PaiNN's own paper included)
trains on ~1,000 configurations, so this is closer to the published protocol
than the full split would be.

Profiling script: `scripts/smoke_train_painn.py` (throwaway; reads the train
split only, writes no checkpoint and no results — delete once SCRUM-51
lands). A 300-step smoke run confirms the model learns: loss 36,725 → 841,
force MAE 13.7 → 2.2 kcal/mol/Å. **Those are not results** and are not
comparable with the other bake-off models.

---

## 7. Phase 1 protocol v1 result

Trained by the shared `ml/training/train.py` under the [Phase 1 protocol v1](../phase1_protocol.md)
and scored once on the full test split by `ml/training/evaluate.py`. This is
this model's row in the [Phase 1 benchmark](../../reports/phase1_benchmark.md);
the run record is [`experiments/runs/phase1_painn/`](../../experiments/runs/phase1_painn/).

| | |
|---|---|
| Config | [`experiments/configs/phase1_painn.yaml`](../../experiments/configs/phase1_painn.yaml) - hidden_channels 128, num_layers 3, cutoff_radius 5.0, num_rbf 20, max_atomic_number 100, eps 1e-08 |
| Parameters | 588,033 |
| Trained at | `b1c2d89`, clean tree |
| Data | ethanol DFT: 10,000 train / 5,000 val frames (stride subsets, seed 42); test: all 55,509 frames, seed 0, batch 32 |
| Epochs | 200 run, best epoch 200; stopped by the 200-epoch cap, still improving (12 val improvements in the last 20 epochs, final lr 1.56e-05) |
| Val / train loss at the best epoch | 3.53x |
| Training time | 4,872 s (1.35 h), cpu, 4 threads |
| Physics checks | 14 of 14 passed on the earlier `d9f3814` checkpoint (`tests/ml/test_painn_physics.py`: rotation, reflection, translation, permutation, finite-difference forces, zero net force, batching, gradient path from `pos`); not yet re-run on this retrained one |

| Test metric | Value |
|---|---|
| **Energy MAE** (total) | **0.0557** kcal/mol |
| Energy RMSE (total) | 0.0711 kcal/mol |
| Energy MAE (per atom) | 0.00619 kcal/mol/atom |
| **Force MAE** (component) | **0.1117** kcal/mol/Å |
| Force RMSE (component) | 0.1858 kcal/mol/Å |
| Force MAE (atom norm) | 0.2236 kcal/mol/Å |
| Inference | 0.742 ms/frame (forward + force backward, batch 32) |

One seed, at the epoch cap; see the protocol's limitations before comparing
close rows.

Supersedes the first v1 run (`d9f3814`, energy MAE 0.0531, force MAE 0.1196), retrained with the same config and seed on another machine on 19 Oct for run-record schema consistency.

Both energy conventions (`total`, `per_atom`) and both force conventions
(`component`, `atom_norm`) are always computed and labelled - a bare MAE is
not interpretable without them. The benchmark table uses `energy_mae_total`
and `force_mae_component`, the MD17/SchNet/PaiNN reporting convention.

Reproduce with:

```bash
python -m ml.training.train    experiments/configs/phase1_painn.yaml
python -m ml.training.evaluate --checkpoint experiments/checkpoints/phase1_painn.pt
```

The checkpoint is gitignored; the run record is committed.

---

## 8. Uncertainty quantification

**None.** Phase 1 is deterministic by construction, so `ece` and
`uncertainty_correlation` are `null` in the results JSON rather than
invented. Those metric *definitions* (binning strategy, correlation
choice, what counts as calibrated) are UQ protocol v1.0 in
`ml/utils/uq_metrics.py`, Fazin's since 23 Oct 2026 (previously
Dongxiao's); Phase 2/3 runs are scored against them.

---

## 9. Limitations and known risks

- **One seed, at the 200-epoch cap.** The §7 result is a single run that was
  still improving when the cap stopped it; see the
  [protocol's limitations](../phase1_protocol.md#limitations).
- **Single molecule, single theory level.** Scored on ethanol/DFT only.
  Absolute energies are not comparable across molecules or theory levels —
  `ethanol` also exists at CCSD(T) — so results must stay keyed on
  (molecule, theory) and must never be pooled. `evaluate.py` refuses to
  score a checkpoint against a gold file whose identity disagrees.
- **Reduced training budget.** `train_subset: 10000` is a CPU concession,
  not a modelling choice. It is fair across the bake-off but means absolute
  MAEs will not match published MD17 numbers trained to convergence.
- **glossary.md §3/§7 still scope Phase 1 away from PaiNN.** §6 now lists
  `painn` as a registry key; the narrative sections predate the SCRUM-50
  bake-off.
- **`predict_energy` takes `pos`**, which the original SCRUM-52 signature
  omitted. PaiNN cannot compute an energy without positions, and no
  conservative model can compute correct forces without them. Flagged on
  SCRUM-50 for the other four candidates.
- **Cutoff is fixed at 5 Å.** Fine for 9-atom ethanol; MD22's larger systems
  (up to 370 atoms) would need this revisited.

---

## 10. References

- Schütt, Unke & Gastegger (2021), *Equivariant message passing for the
  prediction of tensorial properties and molecular spectra*, ICML.
- Gilmer et al. (2017), *Neural Message Passing for Quantum Chemistry* —
  the Phase 1 default backbone this is measured against.
- Chmiela et al. (2017), MD17 — the benchmark dataset.
