# Dongxiao DimeNet++ — ethanol (Phase 1 protocol v1 on DFT; earlier CCSD(T) baseline)

## Phase 1 protocol v1 result

Trained by the shared `ml/training/train.py` under the [Phase 1 protocol v1](../phase1_protocol.md)
and scored once on the full test split by `ml/training/evaluate.py`. This is
this model's row in the [Phase 1 benchmark](../../reports/phase1_benchmark.md);
the run record is [`experiments/runs/phase1_dimenet/`](../../experiments/runs/phase1_dimenet/).

| | |
|---|---|
| Config | [`experiments/configs/phase1_dimenet.yaml`](../../experiments/configs/phase1_dimenet.yaml) - hidden_channels 64, out_channels 1, num_blocks 2, int_emb_size 32, basis_emb_size 8, out_emb_channels 128, num_spherical 7, num_radial 6 |
| Parameters | 278,982 |
| Trained at | `ff3e914`, clean tree |
| Data | ethanol DFT: 10,000 train / 5,000 val frames (stride subsets, seed 42); test: all 55,509 frames, seed 0, batch 32 |
| Epochs | 200 run, best epoch 199; stopped by the 200-epoch cap, still improving (6 val improvements in the last 20 epochs, final lr 3.91e-06) |
| Val / train loss at the best epoch | 2.60x |
| Training time | 8,021 s (2.23 h), cpu, 24 threads |
| Physics checks on the trained checkpoint | 14 of 14 pass (`tests/ml/test_painn_physics.py`: rotation, reflection, translation, permutation, finite-difference forces, zero net force, batching, gradient path from `pos`) |

| Test metric | Value |
|---|---|
| **Energy MAE** (total) | **0.0567** kcal/mol |
| Energy RMSE (total) | 0.0721 kcal/mol |
| Energy MAE (per atom) | 0.00630 kcal/mol/atom |
| **Force MAE** (component) | **0.1254** kcal/mol/Å |
| Force RMSE (component) | 0.2045 kcal/mol/Å |
| Force MAE (atom norm) | 0.2508 kcal/mol/Å |
| Inference | 1.010 ms/frame (forward + force backward, batch 32) |

One seed, at the epoch cap; see the protocol's limitations before comparing
close rows.

The mid size (278,982 parameters) was chosen to sit inside the 62k-809k range of the
other four Phase 1 models; see the protocol's DimeNet++ size note.

## Earlier result (2026-09-17, different protocol)

> **The sections below describe the earlier CCSD(T) experiment:** a
> different dataset and level of theory (ethanol CCSD(T), 889 train frames),
> a 7,491-parameter model and the standalone `ml/training/train_dimenet.py`.
> CCSD(T) and DFT energies sit on different absolute scales, so these
> numbers are **not comparable** with the Phase 1 protocol v1 result above
> or with any DFT row.


## Scope
Completed the user-authorized local ethanol CCSD(T) deterministic baseline on 2026-09-17. The user explicitly authorized completion using existing information without Jira access. This does not assert compliance with unavailable Jira requirements or completion of other molecules. Uses only existing Gold data and saved splits: 889 training / 111 validation / 1000 held-out test frames. No test-based tuning or offset correction.

## Selected experiment
Run: dongxiao_dimenet_ethanol_lr1e4_100. Best epoch: 100.
Selection: minimum validation energy MSE + 100 * force-component MSE, across the original 20-epoch run and two predeclared lower-learning-rate runs. The 32-sample smoke run is excluded. Runs use the same small DimeNet++ architecture, CPU and seed 42.

## Held-out test result
Energy total MAE: 1.607129 kcal/mol
Energy total RMSE: 1.736099 kcal/mol
Force component MAE: 1.529615 kcal/mol/Angstrom
Force component RMSE: 2.140271 kcal/mol/Angstrom
All normalisations and per-atom force norm metrics appear in dimenet_final_test.json. Mean signed energy error is +1.576001 kcal/mol; this systematic offset is reported, not corrected using test data.

## Reproduction
Read notebooks/dongxiao_dimenet/Dongxiao_DimeNet_Lab.ipynb for code, actual outputs, plots and historical-run labelling. Its exported HTML can be read without Jupyter. Existing .venv contains training dependencies but not ipykernel; notebook outputs remain readable. For interactive re-execution select/configure an environment with the recorded dependencies and a Jupyter kernel. Paths in the notebook refer to this local checkout and should be adjusted on another machine.
Training entry: python -B -m ml.training.train_dimenet --config experiments/configs/dimenet_ethanol_lr1e4_100.yaml
The runner protects existing output directories. Use a new output_dir for a new experiment. It does not resume optimizer state.
The historical runner labels its purpose as smoke; this card explicitly documents its reuse for baseline exploration without modifying old logs.

## Limits
One seed, one molecule, small architecture and fixed epoch budget; not a claim of convergence, scientific superiority, paper reproduction, UQ or other-molecule completion. Adapter passes physics checks but has not been numerically compared against the compiled-extension PyG forward path. Quadratic triplet enumeration is suited to these small MD17 batches, not large MD22 systems. Shared metrics conventions were read from the locally cached origin/fazin version: total/per-atom energy and component/atom-norm forces; no branch merge or edits to teammate code occurred.

## Provenance
Gold SHA-256: 8e6571e1c0f0ac5a4fbab8a3236c26abff0095f54772866ad6ffa7f525deb441
Checkpoint SHA-256: aa560b100f943232b8701c8ad43ac94bb05ed4a01565542896a492ded12bbc46
Source hashes: dimenet_source_manifest.json. Raw Gold files are not bundled.
No Git commit/push or Jira status change performed.
