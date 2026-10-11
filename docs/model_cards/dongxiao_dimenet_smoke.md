# Dongxiao DimeNet++ smoke checkpoint

Purpose: verify Gold -> DimeNet++ -> energy/force loss -> backward -> validation -> checkpoint. This is not a benchmark or a converged scientific result.

## Scope and provenance

- Branch: dongxiao. No existing shared source files modified.
- Gold source: data/gold/md17/ethanol_ccsd_t.npz. SHA-256 and exact selected indices are recorded in experiments/results/dongxiao_dimenet_smoke/metrics.json.
- Used the first 32 entries of saved train_idx and first 16 of saved val_idx. No resplitting; no test evaluation. Energy centering uses these 32 training samples only, computed in float64 before casting centered targets to float32.
- PyTorch 2.14.0+cpu; PyG 2.8.0.post1; CPU, seed 42, three epochs.
- Config: experiments/configs/dimenet_smoke.yaml. The small architecture and loss weights are smoke settings, not approved final comparison hyperparameters.

## Implementation

ml/models/dimenet.py inherits the installed PyG DimeNetPlusPlus layers, retaining its radial/spherical basis, embedding, interaction and output blocks and DimeNet++ angle convention. The forward adapter consumes the unchanged MD17Dataset radius graph and enumerates directed k->j->i triplets in PyTorch. No compiled radius/sparse extension or global PyG patch is needed. There is no neighbor truncation in the dataset; the smoke uses nine-atom ethanol, below PyG's default neighbor limit.

The edge-pair enumeration is quadratic in batch edge count. This is intended for small MD17 batches and is not suitable for MD22 or large batches without revisiting graph construction. The caller must supply the configured cutoff graph. Distances/angles are recomputed from differentiable positions; stored RBF edge attributes are not used by DimeNet++.

Forces are -dE/dR; training retains their derivative graph for the combined energy/force MSE loss. MAE/RMSE accumulate global element sums: one scalar per molecular energy and all Cartesian force components. Shared metrics.py is currently empty, so metric code is local to this runner; confirm conventions against the team's eventual metric harness before formal comparisons.

## Result and checks

Completed all three epochs with finite losses and gradients. Epoch 3 (best weighted validation MSE): energy MAE 3.489723 kcal/mol, force component MAE 20.210859 kcal/mol/Angstrom. These errors do not demonstrate useful predictive accuracy. The force error decreased, while energy error increased over these short epochs.

Three unittest cases pass: finite-difference energy/force consistency and parameter gradients; translation/rotation consistency and batch isolation; saved Gold split selection, train-only centering and rejection of Bronze paths.

Checkpoint: experiments/results/dongxiao_dimenet_smoke/best.pt. It contains model_state, config and energy_mean. The network predicts centered total energy; add energy_mean to restore kcal/mol energies. The constant offset does not affect forces. Use the same adapter, architecture config and cutoff when loading the checkpoint.

## Commands (repository root, existing .venv)

    python -B -m unittest discover -s tests -p test_dimenet.py -v
    python -B -m ml.training.train_dimenet --config experiments/configs/dimenet_smoke.yaml

The completed run directory is intentionally protected from overwrite. To run again, copy the configuration and choose a new train.output_dir under experiments/results. The runner refuses to run outside dongxiao. This module is a smoke runner, not the team's shared trainer; no optimizer resume or final test evaluation is implemented.

PyG reference: https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.models.DimeNetPlusPlus.html
