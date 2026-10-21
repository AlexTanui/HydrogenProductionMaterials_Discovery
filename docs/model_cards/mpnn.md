# Model card — MPNN (Phase 1 baseline)

**Jira:** SCRUM-51 · **Owner:** Ruturaj (model) · **Card:** Fazin, from the run record
**Status:** trained and scored under the Phase 1 protocol v1 (§6)
**Code:** [`ml/models/mpnn.py`](../../ml/models/mpnn.py)
**Config:** [`experiments/configs/phase1_baseline.yaml`](../../experiments/configs/phase1_baseline.yaml)
**Tests:** [`tests/ml/test_painn_physics.py`](../../tests/ml/test_painn_physics.py) (physics, all six models),
[`tests/ml/test_harness.py`](../../tests/ml/test_harness.py) (training path)

> `docs/model_cards/TEMPLATE.md` is empty, so this card follows the
> sections the existing cards share (what the model is, architecture, data,
> hyperparameters, verification, result, UQ, limitations, reproducing).

---

## 1. What this model is

A Gilmer-style message-passing neural network (Gilmer et al. 2017) that
predicts the total potential energy of a molecule, with forces taken as
F = -dE/dR by autograd. It is the project's Phase 1 backbone: Phase 2's
BLIP (`ml/models/blip.py`) keeps this exact layer structure and makes the
message and update weights stochastic, so any Phase 1 vs Phase 2 difference
is attributable to where the randomness lives (CLAUDE.md, three-phase plan).

---

## 2. Architecture

- **Embedding:** atomic number -> 128-dimensional feature (table of 101 rows).
- **Radial basis:** 50 Gaussians with centres evenly spaced over 0-5 Å and
  width equal to their spacing (about 0.10 Å), recomputed from `pos` on
  every call.
- **4 message-passing layers.** For each edge j -> i, a message
  MLP([h_j, rbf(d_ij)]) (Linear 178 -> 128, SiLU, Linear 128 -> 128) is
  summed into atom i, then h_i <- h_i + MLP([h_i, m_i]) (Linear 256 -> 128,
  SiLU, Linear 128 -> 128): a residual update.
- **Readout:** per-atom MLP (Linear 128 -> 128, SiLU, Linear 128 -> 1),
  summed over the molecule, so the energy is size-extensive.
- **Forces:** `predict_energy_and_forces` recomputes distances and the
  radial basis from `pos` and differentiates the energy with respect to it.
  The dataset's precomputed `edge_attr` is ignored on this path, because it
  is detached from `pos` and would give forces that are not the gradient of
  the reported energy.
- **Energy scale:** wrapped, like every harness model, in
  `ScaledEnergyModel` (E = scale * E_model + n_atoms * shift, per-atom
  statistics from the train subset; see `ml/models/scaling.py`).

384,897 parameters at this size.

---

## 3. Data

MD17 ethanol at DFT (`data/gold/md17/ethanol_dft.npz`), contiguous
trajectory-block split, 10,000 train and 5,000 val frames by stride
subsets, scored on all 55,509 test frames - identical for every Phase 1
model. Full detail and reasons: [docs/phase1_protocol.md](../phase1_protocol.md).

---

## 4. Hyperparameters

| Setting | Value |
|---|---|
| `hidden_channels` | 128 |
| `num_layers` | 4 |
| `num_rbf` | 50 (this model's own Gaussian basis; independent of `data.num_rbf`) |
| `cutoff` | 5.0 Å (constructor default; equals `data.cutoff_radius`) |
| Training | the locked protocol: Adam 5e-4, weight decay 1e-6, ReduceLROnPlateau (0.5, 5), clipping 10, MSE with weights 1 / 100, batch 32, up to 200 epochs |

---

## 5. Correctness verification

`tests/ml/test_painn_physics.py` runs 14 checks on real ethanol frames in
float64, each shown to fail on a deliberately broken model: invariance of the
energy under rotation, reflection, translation and atom permutation (forces
transform accordingly); forces equal to central finite differences of the
energy with the graph rebuilt at each displacement; zero net force;
batched predictions equal to single-frame ones; and forces that come from
`pos`, not from the detached `edge_attr`. MPNN passes **all 14 untrained
and all 14 on the trained checkpoint**.

---

## 6. Phase 1 protocol v1 result

Trained by the shared `ml/training/train.py` under the [Phase 1 protocol v1](../phase1_protocol.md)
and scored once on the full test split by `ml/training/evaluate.py`. This is
this model's row in the [Phase 1 benchmark](../../reports/phase1_benchmark.md);
the run record is [`experiments/runs/phase1_baseline/`](../../experiments/runs/phase1_baseline/).

| | |
|---|---|
| Config | [`experiments/configs/phase1_baseline.yaml`](../../experiments/configs/phase1_baseline.yaml) - hidden_channels 128, num_layers 4, num_rbf 50 |
| Parameters | 384,897 |
| Trained at | `755d57f`, clean tree |
| Data | ethanol DFT: 10,000 train / 5,000 val frames (stride subsets, seed 42); test: all 55,509 frames, seed 0, batch 32 |
| Epochs | 200 run, best epoch 199; stopped by the 200-epoch cap, still improving (6 val improvements in the last 20 epochs, final lr 7.81e-06) |
| Val / train loss at the best epoch | 2.81x |
| Training time | 2,261 s (0.63 h), cpu, 4 threads |
| Physics checks | 14 of 14 passed on the earlier `d9f3814` checkpoint (`tests/ml/test_painn_physics.py`: rotation, reflection, translation, permutation, finite-difference forces, zero net force, batching, gradient path from `pos`); not yet re-run on this retrained one |

| Test metric | Value |
|---|---|
| **Energy MAE** (total) | **0.0779** kcal/mol |
| Energy RMSE (total) | 0.0962 kcal/mol |
| Energy MAE (per atom) | 0.00866 kcal/mol/atom |
| **Force MAE** (component) | **0.1775** kcal/mol/Å |
| Force RMSE (component) | 0.2774 kcal/mol/Å |
| Force MAE (atom norm) | 0.3552 kcal/mol/Å |
| Inference | 0.337 ms/frame (forward + force backward, batch 32) |

One seed, at the epoch cap; see the protocol's limitations before comparing
close rows.

Supersedes the first v1 run (`d9f3814`, energy MAE 0.0891, force MAE 0.1791), retrained with the same config and seed on another machine on 19 Oct for run-record schema consistency.

---

## 7. Uncertainty quantification

None. Phase 1 is deterministic, so ECE and the uncertainty-error Spearman
correlation are empty in the benchmark table. Their definitions are
Dongxiao's; Phase 2 (BLIP) adds the uncertainty on this backbone.

---

## 8. Limitations and known risks

- **One seed, at the 200-epoch cap, still improving** - see the
  [protocol's limitations](../phase1_protocol.md#limitations).
- **A large train/val gap.** Val loss was 2.81x train loss at the best
  epoch (PaiNN 3.53x and DimeNet++ 3.45x are larger): with 10,000 training
  frames MPNN fits the training subset much better than it generalises.
- **No smooth cutoff envelope.** Unlike PaiNN, PhysNet and DimeNet++, MPNN's
  messages do not fade to zero at 5 Å, so the energy would jump when an atom
  pair crosses the cutoff. This does not arise for ethanol: across 20,000
  frames sampled over the whole trajectory the largest interatomic distance
  is 4.43 Å, so the graph is always fully connected. It would matter for
  larger molecules (MD22).
- **`pos.requires_grad_(True)` modifies the caller's tensor in place**
  ([mpnn.py](../../ml/models/mpnn.py)). Harmless in the harness, which
  passes a fresh tensor every batch, but surprising to other callers.
- **Single molecule, single theory level.** Ethanol DFT only; results are
  keyed on (molecule, theory) and never pooled.

---

## 9. Reproducing

```bash
python -m ml.training.train    experiments/configs/phase1_baseline.yaml
python -m ml.training.evaluate --checkpoint experiments/checkpoints/phase1_baseline.pt
```

The checkpoint is gitignored; the run record
([`experiments/runs/phase1_baseline/`](../../experiments/runs/phase1_baseline/))
is committed and holds the frames used, energy statistics, environment and
per-epoch history.

---

## 10. References

- Gilmer, Schoenholz, Riley, Vinyals & Dahl (2017), *Neural Message Passing
  for Quantum Chemistry*, ICML.
- Chmiela et al. (2017), *Machine learning of accurate energy-conserving
  molecular force fields*, Science Advances - MD17.
