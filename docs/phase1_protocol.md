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
| Hardware | CPU only: Apple arm64, macOS 14.4, 4 torch threads; torch 2.13.0, PyG 2.8.0.post1, Python 3.11.9 | Recorded in every run record. All five runs trained on this machine on 18–19 Oct (UTC), so their wall-clock times are comparable with each other only. The first v1 runs (15–16 Oct) used a 24-thread Windows x86 CPU. |
| Test scoring | Once per model, after training, by `ml/training/evaluate.py`: all 55,509 test frames, batch 32, RNG seed 0 | Training never opens the test split. The fixed seed makes a stochastic model's score reproducible; deterministic models are unaffected. |
| Scoring-data fingerprint | Every score records sha256 of the test indices and of z, R, E, F at them; all five runs share `518af6e3…` | Molecule, theory and frame count cannot tell a different gold file of the same size apart; `benchmark.py` refuses runs whose fingerprints differ. |
| Provenance | Each run records its git commit and a dirty-tree flag; every run was clean. MPNN ran at `755d57f`; SchNet, PaiNN, PhysNet and DimeNet++ at `b1c2d89` | Those two commits differ only in run records, not in code. |

**DimeNet++ size.** DimeNet++ uses hidden 64, out_emb 128, int_emb 32,
basis_emb 8, 2 blocks, 7 spherical and 6 radial basis functions: 278,982
parameters, inside the 62k–809k range of the other four models. The size
previously used for CCSD(T) ethanol (7,491 parameters) was far below that
range, and the published default (1.89M) would be over twice the largest
model (PhysNet, 809k) and ran at about 97 s per epoch, against about 40 s
for the mid size.

**Reproducibility of scoring.** Scoring on a multithreaded CPU is not
bit-reproducible: the force backward pass sums in a thread-dependent order,
so the same checkpoint scores slightly differently from run to run. The
five current runs were scored by evaluate.py with their fingerprints
recorded at scoring time, so none needed a re-score, and none has been
re-scored yet (the report's last column). The first v1 runs, scored before
evaluate.py recorded fingerprints, were re-scored
(`scripts/fingerprint_run_records.py`, seed 0, batch 32) from the clean
commit `549e6dd`, which reproduced every stored metric to within 2.3 × 10⁻⁸
relative (energies bit-identical). The
documented tolerance is
**10⁻⁵ relative**, which leaves room for re-scoring on another machine or
thread count (about 1.2 × 10⁻⁶ measured) and is about 100 times below the 4
decimals reported. The stored metrics were kept unchanged. Re-scoring
reproduces numbers already reported and tunes or selects nothing, so the
"test scored once" rule still holds: no decision was made from the test
split.

## Limitations

1. **One seed per model.** There is no estimate of seed-to-seed variation.
   Models whose errors are close (energy MAE 0.0537 for DimeNet++, 0.0557
   for PaiNN, 0.0611 for PhysNet) cannot be ranked against each other
   without repeated runs. Retraining all five with the same configs and
   seeds on another machine (19 Oct) moved SchNet's energy MAE from 0.1173
   to 0.1714 and swapped PaiNN and DimeNet++ on energy MAE.
2. **Every model hit the 200-epoch cap.** None stopped early. Four were
   still improving: best epochs 199–200, val loss improved 6–12 times in
   their last 20 epochs, and the learning rate had decayed to
   3.9 × 10⁻⁶ – 1.6 × 10⁻⁵. DimeNet++ last improved at epoch 191 (3 times
   in its last 20), with its learning rate down to 4.9 × 10⁻⁷. These are
   results at 200 epochs, not converged ones. Longer training would
   probably lower every error and could change the ordering.
3. **Parameter counts differ.** They range from 61,569 (SchNet) to 808,746
   (PhysNet), with DimeNet++ at 278,982. Each model is compared at its own
   configured size, not at matched capacity.
4. **Train/val loss gaps of 1.2–3.5× on 10,000 training frames.** At each
   model's best epoch, val loss divided by train loss was 1.23 (SchNet),
   1.77 (PhysNet), 2.81 (MPNN), 3.45 (DimeNet++) and 3.53 (PaiNN). Train
   loss is the epoch's mean over batches in training mode, so the ratio is
   approximate. Higher-capacity models fit the 10,000 training frames much
   better than they generalise, so more data or regularisation could change
   the comparison.
5. **BLIP (Phase 2) is excluded** until its batch coupling is fixed. Its
   noise scale is pooled over the whole batch, so a molecule's prediction
   depends on its batch-mates; see
   [docs/reviews/blip-batch-dependence.md](reviews/blip-batch-dependence.md).
