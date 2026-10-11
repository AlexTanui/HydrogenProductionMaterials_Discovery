# Model card — SchNet (Phase 1 bake-off)

**Ticket:** SCRUM-53 · **Owner:** Alex · **Status:** Trained, evaluated, committed (`alex` @ `1a09bc8`) · **Date:** 2026-09-13

## Summary

SchNet — continuous-filter convolutions over Gaussian-expanded interatomic
distance (Schütt et al. 2017) — trained to predict energy and forces for
MD17 ethanol at the DFT level, as one of five architectures in the Phase 1
bake-off (MPNN, SchNet, PaiNN, DimeNet++, PhysNet), compared head-to-head
on identical data, split, targets, and evaluation.

## Setup

- **Data:** `data/gold/md17/ethanol_dft.npz` — 9 atoms, 555,092 configs.
- **Split:** fixed 80/10/10 contiguous trajectory blocks, baked into the
  gold file (`train_idx`/`val_idx`/`test_idx`) — deterministic, no RNG, so
  identical across every bake-off entry.
- **Architecture:** `hidden_channels=64`, `num_layers=3`, `num_rbf=16`,
  `cutoff=5.0 Å`.
- **Training:** Adam (`lr=5e-4`, `weight_decay=1e-6`), gradient clipping
  (`max_norm=10`), `ReduceLROnPlateau`, `batch_size=256`, `seed=42`,
  `energy_loss_weight=1.0`, `force_loss_weight=100.0` — the last two fixed
  across all 5 bake-off models per the shared config schema.
- **Code:** `ml/models/schnet.py` (the model) + `ml/training/train_schnet.py`
  (standalone train/eval script — a temporary stand-in for
  `ml/training/train.py`/`evaluate.py`, both still empty at time of
  writing; retire this file once either lands).

## Key implementation decision: energy centering

Ethanol's raw total energy sits at **-97,195.9 kcal/mol with a std of only
~4.2 kcal/mol** — the physically meaningful signal is a tiny wobble on an
enormous constant. Training directly on the raw value would spend most of
early optimization just walking the readout bias down to -97,196 before
learning anything real. Instead, the model trains against
`E - mean(train_E)`, and the shift is added back before computing metrics
— so every number below is real, physical kcal/mol, comparable to the
other 4 architectures, not an artifact of the internal training trick.
The shift (`-97195.891`) is saved inside the checkpoint.

A second, unrelated correctness issue was caught and avoided while
building this: `raj_mpnn/MPNN.ipynb`'s `predict_energy_and_forces` sets
`pos.requires_grad_(True)` but then computes energy from a *precomputed*
`edge_attr` with no graph connection back to `pos` — `torch.autograd.grad`
would have nothing to differentiate. SchNet instead recomputes RBF
features from the live `pos` inside `predict_energy_and_forces`, verified
against a finite-difference check (0.004008 finite-diff vs. 0.003984
autograd, on a synthetic test) before ever training on real data.

## Training behavior

Full training curves, LR schedule, and parity plots:
**[SchNet Training Run dashboard](https://claude.ai/code/artifact/4ba8277f-0410-4df8-bf5e-fa77827f5366)**

- Ran the full 200-epoch cap — **never triggered early stopping**
  (`patience=20`); validation loss kept finding small improvements
  throughout, so there's likely headroom for a better result with a
  longer run later, not urgent for the Sprint 3 comparison.
- **No overfitting signature:** train and validation loss tracked each
  other closely for the entire run rather than diverging — the tell for
  overfitting (val worsening while train keeps improving) never appeared.
- Learning rate stepped down 4× via `ReduceLROnPlateau`
  (5e-4 → 2.5e-4 → 1.25e-4 → 6.25e-5 → 3.13e-5) as progress slowed at each
  plateau; each cut produced a further small drop in loss.
- **Reported metrics use epoch 198's weights** (best validation loss,
  3.4201), not epoch 200 — epochs 199–200 were both slightly worse, so the
  best checkpoint was kept rather than the last one.

## Final metrics (held-out test split, 55,509 frames / 499,581 atoms — untouched until this evaluation)

| Metric | Value |
|---|---|
| **Energy MAE** | 0.0873 kcal/mol |
| Energy RMSE | 0.1026 kcal/mol |
| Energy MAE (per atom) | 0.0097 kcal/mol/atom |
| **Force MAE** (component) | 0.1114 kcal/mol/Å |
| Force RMSE (component) | 0.1628 kcal/mol/Å |
| Force MAE (atom-norm) | 0.2231 kcal/mol/Å |
| Force RMSE (atom-norm) | 0.2820 kcal/mol/Å |
| **Training time** | 30,029s (~8h 20m), CPU |
| **Inference time** | 9.03s total / 0.163 ms per frame, full test set, CPU |

Scored via `ml.utils.metrics.MetricAccumulator` (Fazin, SCRUM-37) — same
harness every bake-off entry uses, so these numbers are directly
comparable without unit or convention caveats.

## Reproducibility

- Checkpoint: `experiments/checkpoints/phase1_schnet.pt` (gitignored, not
  committed — regenerate via `python -m ml.training.train_schnet`).
- Results: `experiments/results/phase1_schnet.json` (gitignored).
- Config: `experiments/configs/phase1_schnet.yaml`.
- Seed 42, deterministic split — rerunning should reproduce closely
  (float non-determinism aside).

## Known limitations

- Trained and evaluated on ethanol/DFT only — not yet run against the
  other MD17 molecules or MD22.
- `ml/training/train_schnet.py` is SchNet-specific and temporary; once
  `ml/training/train.py`/`evaluate.py` exist, this run should be
  reproduced through the shared harness for the final report so all 5
  models went through identical code, not just identical data/config.
- No uncertainty quantification — this is a Phase 1 (deterministic)
  baseline; UQ is Phase 2/3 scope.
