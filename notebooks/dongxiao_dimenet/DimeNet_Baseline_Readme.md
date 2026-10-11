# Dongxiao DimeNet++ — ethanol CCSD(T) baseline

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
