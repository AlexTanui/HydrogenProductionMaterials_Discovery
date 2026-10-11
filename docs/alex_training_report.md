---
title: "SchNet Training Report"
subtitle: "Phase 1 Baseline Bake-off — Sprint 3 (SCRUM-53)"
date: "2 October 2026"
---

**Prepared by:** Alex Tanui
**Role:** Scrum Master / Technical Lead
**Ticket:** SCRUM-53, under the Phase 1 bake-off epic (SCRUM-50)
**Status:** Trained, evaluated, committed (`alex` branch @ `1a09bc8`)

---

## Summary

SchNet — continuous-filter convolutions over Gaussian-expanded interatomic
distance (Schütt et al., 2017) — was trained to predict molecular energy
and interatomic forces for MD17 ethanol at the DFT level of theory, as one
of five architectures in the Phase 1 baseline bake-off (alongside MPNN,
PaiNN, DimeNet++, and PhysNet). This is currently the only entry in that
bake-off trained for the full epoch budget on the complete shared dataset,
making it the project's reference Phase 1 result as of this report.

## Setup

| | |
|---|---|
| Data | `data/gold/md17/ethanol_dft.npz` — 9 atoms, 555,092 configurations |
| Split | Fixed 80/10/10 contiguous trajectory blocks, baked into the gold file — deterministic, no randomness, identical for every bake-off entry |
| Architecture | `hidden_channels=64`, `num_layers=3`, `num_rbf=16`, `cutoff=5.0 Å` |
| Optimiser | Adam, `lr=5e-4`, `weight_decay=1e-6`, gradient clipping (`max_norm=10`), `ReduceLROnPlateau` |
| Batch size / seed | 256 / 42 |
| Loss weighting | `energy_loss_weight=1.0`, `force_loss_weight=100.0` — fixed across all five bake-off models |
| Code | `ml/models/schnet.py` (model) + `ml/training/train_schnet.py` (standalone train/eval script) |

## Key implementation decision: energy centering

Ethanol's raw total energy sits at approximately **−97,195.9 kcal/mol,
with a standard deviation of only ~4.2 kcal/mol** — the physically
meaningful signal is a small variation riding on an enormous constant.
Training directly against the raw value would spend most of early
optimisation simply learning that constant offset rather than anything
physically useful. Instead, the model is trained against
`E − mean(train_E)`, with the shift added back before any metric is
computed — so every number reported below is a genuine physical quantity,
directly comparable to the other four architectures in the bake-off, not
an artefact of this internal technique. The shift value
(`−97195.891`) is stored inside the saved checkpoint for reproducibility.

A second, unrelated correctness issue was identified and corrected during
this work: the original prototype this project's MPNN baseline was drafted
from computed forces from a precomputed edge feature with no gradient
connection back to atomic positions — meaning `torch.autograd.grad` would
have nothing to differentiate, and forces would silently come out as
exact zeros with no error raised. SchNet instead recomputes the relevant
distance features from the live atomic positions inside the
force-computation path, and this was verified correct with a
finite-difference check (0.004008 by finite difference vs. 0.003984 by
autograd, on a controlled synthetic test) before any training on real
data began.

## Training behaviour

The model was trained for the full 200-epoch budget and never triggered
early stopping (patience of 20 epochs) — validation loss continued to find
small improvements throughout the run. Train and validation loss tracked
each other closely for the entire run, rather than diverging, which is the
expected signature of a model that is still learning rather than
overfitting. The learning rate was reduced automatically four times by the
scheduler as progress plateaued (5e-4 → 2.5e-4 → 1.25e-4 → 6.25e-5 →
3.13e-5), with each reduction producing a further small drop in loss. The
metrics reported below use the weights from epoch 198 — the epoch with the
best validation loss — rather than epoch 200, since the final two epochs
were both marginally worse.

## Results

Evaluated on the held-out test split (55,509 frames, 499,581 atoms),
untouched until this final evaluation, scored via the project's shared
metric harness (`ml.utils.metrics.MetricAccumulator`), so these numbers
are directly comparable to the other bake-off entries without unit or
convention caveats.

| Metric | Value |
|---|---|
| **Energy MAE** | 0.0873 kcal/mol |
| Energy RMSE | 0.1026 kcal/mol |
| Energy MAE (per atom) | 0.0097 kcal/mol/atom |
| **Force MAE** (component) | 0.1114 kcal/mol/Å |
| Force RMSE (component) | 0.1628 kcal/mol/Å |
| Force MAE (atom-norm) | 0.2231 kcal/mol/Å |
| Force RMSE (atom-norm) | 0.2820 kcal/mol/Å |
| **Training time** | 30,029 s (~8h 20m), CPU |
| **Inference time** | 9.03 s total / 0.163 ms per frame, full test set |

The two charts below, regenerated directly from the saved checkpoint
against the real test set, show how tightly the model's predictions track
the true values — each point close to the dashed line indicates a correct
prediction.

![Energy parity — predicted vs. true, 1,280 test-set frames.](training_summary_assets/schnet_energy_parity.png){ width=48% }
![Force-component parity — predicted vs. true, 2,500 sampled components.](training_summary_assets/schnet_force_parity.png){ width=48% }

Force predictions show more spread than energy, which is expected — force
is the harder quantity to predict, since it is a per-atom, per-direction
derivative rather than a single scalar per molecule.

## Reproducibility

- Checkpoint: `experiments/checkpoints/phase1_schnet.pt` (not committed —
  gitignored by project convention; regenerate via
  `python -m ml.training.train_schnet`).
- Results: `experiments/results/phase1_schnet.json` (also gitignored).
- Configuration: `experiments/configs/phase1_schnet.yaml`.
- Fixed seed (42) and a deterministic data split mean a re-run should
  reproduce these results closely, floating-point non-determinism aside.

## Known limitations

- Trained and evaluated on ethanol at the DFT level only — not yet run
  against the other MD17 molecules or the larger MD22 systems.
- `ml/training/train_schnet.py` is a temporary, architecture-specific
  script. Once the shared `ml/training/train.py`/`evaluate.py` harness is
  complete for every architecture, this result should be reproduced
  through it, so that every bake-off entry has gone through identical code
  as well as identical data.
- This is a deterministic Phase 1 baseline with no uncertainty
  quantification — that is explicitly Phase 2/3 scope, not a gap in this
  entry.

## Context within the Phase 1 bake-off

As of this report, this is the only bake-off entry trained on the full
shared dataset for the full epoch budget. MPNN's shared training harness
exists but has only been smoke-tested; DimeNet++ was trained on a
different (CCSD(T)) dataset; PhysNet was trained on a 4.5% data subset for
compute reasons; PaiNN is implemented but not yet trained. This result is
therefore the project's current reference point for what a fully
converged Phase 1 baseline looks like, pending the remaining four entries
being brought onto the same footing.
