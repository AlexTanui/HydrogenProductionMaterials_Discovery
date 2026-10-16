# Phase 1 training protocol (v1)

Every Phase 1 model was trained and scored under the settings below. Only
each config's `model:` block differs. `ml/training/benchmark.py` refuses to
tabulate runs whose settings differ, and `tests/ml/test_harness.py` fails if
the six configs' `data:` and `train:` blocks drift apart.

**Results:** [reports/phase1_benchmark.md](../reports/phase1_benchmark.md) ([CSV](../reports/phase1_benchmark.csv)).
**Run records** (config, git commit, frames used, energy statistics, environment, per-epoch history, test metrics):
[SchNet](../experiments/runs/phase1_schnet/) ·
[MPNN](../experiments/runs/phase1_baseline/) ·
[PaiNN](../experiments/runs/phase1_painn/) ·
[PhysNet](../experiments/runs/phase1_physnet/) ·
[DimeNet++](../experiments/runs/phase1_dimenet/).

## Locked settings

| Setting | Value | Why |
|---|---|---|
| Data | MD17 ethanol, DFT: `data/gold/md17/ethanol_dft.npz` | One molecule and one level of theory for every model; theory levels are never pooled. |
| Split | The gold file's contiguous trajectory blocks: 444,074 / 55,509 / 55,509 frames (train / val / test), made without an RNG and never re-derived | Adjacent MD frames are near-duplicates, so a random split would leak near-identical structures into test. |
| Graph | 5.0 Å cutoff (`build_graph`); every model's own cutoff must equal it | A smaller model cutoff would silently drop messages; train.py checks. |
| Train subset | 10,000 frames of `train_idx`: stride 44, offset 3 (seed 42) | The full split at 200 epochs is out of reach on CPU. A stride spreads the frames over the whole trajectory, and every model sees the same frames. |
| Val subset | 5,000 frames of `val_idx`: stride 11, offset 0 (same seed) | Keeps validation cheap at every epoch while using the same frames for every model. |
| Uncovered tail | The stride rounds down, so the last 4,074 train frames and the last 509 val frames (0.92% of each block) are never drawn | A known property of the algorithm. Each subset still spans 99.1% of its block, and the undrawn frames sit at the end of each block, next to the following split. |
| Energy scale-and-shift | One wrapper for every model: E = scale · E_model + n_atoms · shift, F = scale · F_model. Per atom: shift = mean(E)/9 = −10,799.535 kcal/mol, scale = std(E)/9 = 0.4655 kcal/mol, both from the train subset only | Removes the ~97,196 kcal/mol offset a fresh network cannot reach. Stored in the checkpoint, so every consumer gets absolute energies; the loss is on absolute energies. |
| Loss | MSE(E) + 100 · MSE(F), in kcal/mol and kcal/mol/Å | Forces carry most of the information per frame (27 components against one energy); the 1 : 100 weighting is the glossary §6 convention. |
| Optimiser | Adam, lr 5 × 10⁻⁴, weight decay 10⁻⁶ | The recipe of the shared loop's original version, kept unchanged across models. |
| Scheduler | ReduceLROnPlateau on val loss, factor 0.5, patience 5 | Halves the step size when validation stalls. |
| Gradient clipping | Global norm 10 | Guards the early epochs of force training, where gradients come from a double backward pass. |
| Epochs and early stopping | Up to 200 epochs; stop after 20 without val improvement; keep the lowest-val-loss checkpoint | Model selection sees the val subset only, never test. |
| Batch | 32 | The glossary §6 default, the same for every model; SchNet's earlier batch of 256 was dropped for parity. |
| Seeds and batch order | `torch.manual_seed(42)` before the model is built; training batches shuffled by their own `torch.Generator(42)` | Every model sees the same batches in the same order, however much of the global RNG its initialisation consumes. |
| Hardware | CPU only: Intel64 Family 6 Model 198, 24 torch threads; torch 2.13.0+cpu, PyG 2.8.0.post1, Python 3.11.9, Windows 10.0.26200 | Recorded in every run record. Wall-clock times are comparable only on this machine, and its speed varied by about 2× between sessions on 15–16 Oct. |
| Test scoring | Once per model, after training, by `ml/training/evaluate.py`: all 55,509 test frames, batch 32, RNG seed 0 | Training never opens the test split. The fixed seed makes a stochastic model's score reproducible; deterministic models are unaffected. |
| Provenance | Each run records its git commit and a dirty-tree flag; every run was clean. SchNet, MPNN, PaiNN and PhysNet ran at `d9f3814`, DimeNet++ at `ff3e914` | The only change between those two commits is DimeNet++'s `model:` block. |

**DimeNet++ size.** DimeNet++ uses hidden 64, out_emb 128, int_emb 32,
basis_emb 8, 2 blocks, 7 spherical and 6 radial basis functions: 278,982
parameters, inside the 62k–809k range of the other four models. The size
previously used for CCSD(T) ethanol (7,491 parameters) was far below that
range, and the published default (1.89M) would be over twice the largest
model (PhysNet, 809k) and ran at about 97 s per epoch, against about 40 s
for the mid size.

## Limitations

1. **One seed per model.** There is no estimate of seed-to-seed variation.
   Models whose errors are close (energy MAE 0.0531 for PaiNN, 0.0567 for
   DimeNet++, 0.0590 for PhysNet) cannot be ranked against each other
   without repeated runs.
2. **Every model hit the 200-epoch cap while still improving.** None stopped
   early; the best epochs were 198–200, each model's val loss improved
   5–7 times in its last 20 epochs, and the learning rate had decayed to
   3.9 × 10⁻⁶ – 1.6 × 10⁻⁵. These are results at 200 epochs, not converged
   ones. Longer training would probably lower every error and could change
   the ordering.
3. **Parameter counts differ.** They range from 61,569 (SchNet) to 808,746
   (PhysNet), with DimeNet++ at 278,982. Each model is compared at its own
   configured size, not at matched capacity.
4. **Train/val loss gaps of 1.3–3.8× on 10,000 training frames.** At each
   model's best epoch, val loss divided by train loss was 1.27 (SchNet),
   1.77 (PhysNet), 2.60 (DimeNet++), 3.45 (PaiNN) and 3.77 (MPNN). Train
   loss is the epoch's mean over batches in training mode, so the ratio is
   approximate. Higher-capacity models fit the 10,000 training frames much
   better than they generalise, so more data or regularisation could change
   the comparison.
5. **BLIP (Phase 2) is excluded** until its batch coupling is fixed. Its
   noise scale is pooled over the whole batch, so a molecule's prediction
   depends on its batch-mates; see
   [docs/reviews/blip-batch-dependence.md](reviews/blip-batch-dependence.md).
