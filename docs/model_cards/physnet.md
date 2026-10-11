# Model Card — PhysNet (Phase 1 bake-off)

**Ticket:** SCRUM-50 epic · **Owner:** Shijin Mathew · **Branch:** `shijin`
**Implementation:** [`ml/models/physnet.py`](../../ml/models/physnet.py) ·
**Config:** [`experiments/configs/phase1_physnet.yaml`](../../experiments/configs/phase1_physnet.yaml) ·
**Tests:** [`tests/test_physnet.py`](../../tests/test_physnet.py)

> `docs/model_cards/TEMPLATE.md` is still empty, so this card is
> improvising a structure. If it holds up, copy it into the template so
> the other four bake-off entries are directly comparable rather than
> each inventing their own layout.

---

## 1. Status

| | |
|---|---|
| Implementation | **Complete** |
| Unit tests | **Complete** — 22 checks, `pytest tests/test_physnet.py` |
| Trained checkpoint | **Not produced yet** |
| Results | **Not produced yet** — §5 below is blank on purpose |

Training is blocked on `ml/training/train.py` (Ruturaj, SCRUM-50 epic),
which is 0 bytes on `main`. `ml/training/train_physnet.py` is included as
standalone scaffolding so this entry can be trained before that lands; if
it is used, **say so in the results**, because a bake-off whose entries
were trained by different code is not a controlled comparison. Re-run
through `train.py` once it exists.

---

## 2. Architecture

PhysNet (Unke & Meuwly, *J. Chem. Theory Comput.* 2019), predicting total
energy as a sum of atom-wise contributions:

```
E = Σ_i ( scale[z_i] · E_i + shift[z_i] )
F = -∂E/∂R        (autograd)
```

| Component | Choice |
|---|---|
| Atom embedding | learned, per atomic number |
| Activation | shifted softplus, `ssp(x) = softplus(x) − ln 2` |
| Edge features | 16 Gaussian RBFs of interatomic distance, 5 Å cutoff |
| Envelope | cosine cutoff, smooth 1 → 0 across the radius |
| Structure | 3 modules × 2 interaction blocks, each with residual blocks and its own output head |
| Readout | per-module atomic contributions summed, then per-element scale and shift |
| Forces | autograd through recomputed interatomic distances |
| Parameters | ~`n` (printed by `repr(model)`; record the value with the results) |

Every module contributes additively to every atomic energy, which leaves a
short gradient path from the first module to the output — PhysNet's answer
to vanishing gradients in deep message passing.

Aggregation uses `index_add_` rather than torch-geometric, so the model
imports only `torch`. PyG is needed to *load* data, not to run the model.

---

## 3. Simplifications and deviations

### 3.1 Short-range term only — electrostatic and dispersion omitted

**Required by the ticket, and defensible on this data.** Full PhysNet adds
an explicit Coulomb term over predicted partial charges and a Grimme-D3
dispersion correction. Both are omitted.

MD17 ethanol is a single neutral molecule of fixed composition, sampled
near equilibrium. There is no composition variation for a charge model to
explain, and no long-range regime for a dispersion tail to act over — at
9 atoms within 5 Å, every atom pair is already inside the cutoff. Both
terms would fit constants.

**This limits what the result generalises to.** The conclusion supported
is "PhysNet's short-range architecture versus the other four, on
fixed-composition small-molecule data" — not "PhysNet versus the other
four". If the bake-off winner is later pointed at charged species,
multiple compositions, or the larger MD22 systems, the omitted terms
become relevant and this entry would need extending before it could be
compared there.

### 3.2 Repo RBF instead of PhysNet's exponential-Bernstein basis

The paper expands distance in exponential Bernstein polynomials. This
implementation uses the project's existing 16 Gaussian RBFs, matching
`ml/data/md17.py::build_graph`.

Chosen for the bake-off's sake: all five models then see identical edge
features, so a difference in results is attributable to architecture
rather than to featurisation. Worth revisiting if PhysNet underperforms —
the basis would be a confound to rule out before concluding anything
about the architecture.

### 3.3 Cosine cutoff envelope added

`build_graph` applies a hard 5 Å cutoff, so an atom drifting across the
boundary adds its contribution discontinuously and the force spikes.
Messages are multiplied by a cosine envelope that decays smoothly to zero
at the radius, making the predicted forces continuous. This follows
PhysNet and costs nothing.

---

## 4. The autograd-force trap (read before building another bake-off model)

`edge_attr` from `ml/data/datasets.py::MD17Dataset` is computed **in
numpy, at dataset-construction time**, by `ml/data/md17.py::build_graph`.
It carries no autograd history back to positions.

A model that predicts energy from `z`, `edge_index` and `edge_attr`, then
asks autograd for `-∂E/∂pos`, receives **a tensor of exact zeros**. No
error is raised. With `force_loss_weight: 100`, the force term becomes a
constant the optimiser cannot move: the run trains, the loss falls
because of the energy term, and the model learns nothing about forces.

`predict_energy_and_forces` therefore recomputes the radial basis from
`pos` inside the forward pass. The passed `edge_attr` is accepted for
interface compatibility and ignored on that path.
`predict_energy` — which consumes `edge_attr` — is for energy-only
evaluation and inference timing, and its docstring says it cannot be used
for forces.

`tests/test_physnet.py` pins this with a finite-difference check against
autograd, which is the force validation `ROADMAP.md` weeks 3–4 asks for.

**This affects all five bake-off entries, not just PhysNet.** Anyone whose
model reads `edge_attr` on the force path has silently-zero forces.

---

## 5. Results

**Subset run — not a full-dataset result.** See the scope note below the table.

| Metric | Value |
|---|---|
| Energy MAE (kcal/mol) | 2.082 total · 0.231 per atom |
| Energy RMSE (kcal/mol) | 2.104 total · 0.234 per atom |
| Force MAE (kcal/mol/Å) | 0.711 per component · 1.430 per atom norm |
| Force RMSE (kcal/mol/Å) | 1.094 per component · 1.895 per atom norm |
| Training time | 3425 s |
| Inference time (ms/config) | 2.9603 ms |
| Parameters | 808,746 |
| Epochs run | 20 |
| Hardware | CPU (no CUDA device available) |
| Trained by | `ml/training/train_physnet.py` (standalone script — the shared `ml/training/train.py` does not exist yet) |

Scored on 2,000 validation configs. Both conventions are reported for each
quantity because `ml/utils/metrics.py` produces both, and a bake-off table
mixing per-atom with total energy — or per-component with atom-norm force —
would compare different quantities under one column heading.

**Scope of this result.** Trained on **20,000 of 444,074** train configs
(4.5%) for 20 epochs, on CPU. A full-dataset run was not possible: at a
measured 0.22 s/batch that is ~57 minutes per epoch, so roughly 8 days for
the config's 200 epochs. `experiments/results/phase1_physnet.json` records
the subset explicitly.

These numbers are therefore **not comparable** with any bake-off entry
trained on the full dataset. For the comparison to mean anything, every
entry needs the same training budget — same configs, same epochs, and
ideally the same machine, since training time is itself a reported metric.

For context, the same model on a 2,000-config, 2-epoch smoke run gave
Energy MAE 18.44 and Force MAE 6.81 — so 10× the data and 10× the epochs
improved both by roughly 9×. The model is still far from converged, and a
longer run should improve these numbers substantially.

**Data (fixed across the bake-off):** `data/gold/md17/ethanol_dft.npz`,
9 atoms, 555,092 configs, contiguous 80/10/10 trajectory-block split
(444,074 / 55,509 / 55,509), seed 42.
**Loss:** `energy_loss_weight: 1.0`, `force_loss_weight: 100.0`.

Report validation numbers until the bake-off's final evaluation. The test
split stays untouched — it is 55,509 configs used once, and every look at
it before then spends a little of its value as a held-out set.
---

## 6. Reproducing

```bash
python -m ml.data.preprocessing --dataset md17      # produces the gold file
pytest tests/test_physnet.py                        # 22 checks, no data needed
python -m ml.training.train_physnet --config experiments/configs/phase1_physnet.yaml
```

Checkpoints and results are gitignored by design — regenerate rather than
pull them.

---

## 7. Known risks

- **Compute.** 444,074 training configs is a real training job. On CPU it
  is not realistic; `train_physnet.py` warns and falls back, and
  `--limit-train` gives a subset path for a working run. A subset result
  is **not** a bake-off result — if any entry ends up trained on a subset,
  every entry must be, or the comparison is meaningless.
- **Energy offset.** Ethanol sits near −97,000 kcal/mol while an untrained
  network outputs O(1). `init_shift_from_energies` anchors the per-element
  shift before step one. Without it the first epochs are spent travelling
  five orders of magnitude. Whether every bake-off entry handles this the
  same way is worth checking — if some do and some don't, the comparison
  partly measures initialisation.
- **Two `predict_energy` signatures are in circulation, and they
  disagree.** The SCRUM-50 ticket specifies
  `predict_energy(z, edge_index, edge_attr, batch)`; the docstring of
  `ml/models/registry.py` (SCRUM-52, `fazin` branch) specifies
  `predict_energy(z, pos, edge_index, edge_attr, batch)`. This model
  accepts both, telling them apart by the second argument's dtype and
  shape. **That shim is a symptom, not a solution** — the team should
  pick one signature and it should then be deleted.
  `predict_energy_and_forces` is not affected: both specs agree, and
  `ml/training/evaluate.py` calls it positionally as
  `(batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch)`,
  which is what this model implements and what
  `test_evaluate_harness_call_signature` pins.

- **PhysNet is not yet in `MODEL_REGISTRY`.** `ml/models/registry.py`
  lives on the `fazin` branch and hasn't merged. Once it does, add one
  line — `"physnet": PhysNet` — or `evaluate.py` cannot score this
  checkpoint, and `build_model` will raise `UnknownModel`. The class
  already satisfies the registry's `REQUIRED_INTERFACE`
  (`predict_energy`, `predict_energy_and_forces`, `config`) and exposes
  `cutoff_radius`, which `evaluate.py` probes by `getattr` to confirm a
  checkpoint is scored at the radius it was trained at — under the short
  name alone that check finds nothing and passes vacuously.

- **Test file location.** `tests/ml/test_painn.py` and
  `tests/ml/test_evaluate.py` sit under `tests/ml/` on the `fazin`
  branch; this one is at `tests/test_physnet.py`, matching what was on
  `main` at the time. Worth aligning on one convention before four more
  test files land in two places.
- **Undocumented in the project spec.** `glossary.md` §6 enumerates
  `mpnn | blip | graph_stochastic` only, and neither it nor `ROADMAP.md`
  mentions a five-model bake-off. The config here sets `phase: physnet`,
  which the schema does not yet allow.
