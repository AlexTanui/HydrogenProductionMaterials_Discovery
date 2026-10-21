# SCRUM-34 — Phase 3 candidates and UQ verification

Date: 22 October 2026 (ACDT). Demo: 23 October 2026.

## Status and evidence

The common backbone is Raj's MPNN. GitHub was read on 22 October into an isolated checkout at `work/team_repository`, from `https://github.com/AlexTanui/HydrogenProductionMaterials_Discovery.git`. Fetched main/alex HEAD: `e52e174`; Fazin: `195eec3`. `origin/ruturaj` is an old branch, so it is not the current integration source. Main contains the integrated MPNN, BLIP, shared UQ evaluation and Phase 3 draft. No remote push, merge into the user's original checkout, or long training was performed.

Original Windows checkout: `C:/Users/64512/HydrogenProductionMaterials_Discovery`, HEAD points to `dongxiao` (last logged commit `dc35abe`). Its MPNN, BLIP, graph stochastic model, shared metrics and Phase 2 config are 0-byte placeholders. Git status/fetch there failed (work-tree/permission errors), so its full dirty state is NOT certified. The fresh isolated checkout avoids modifying those files. Original DimeNet experimental code and raw prediction archives are present and readable.

`docs/literature_review.md` is still 0 bytes on fetched main. Latest `docs/phase2_remaining_work.md` explicitly says the NLL run has not been validated. SCRUM-33 reopened, SCRUM-34 assigned/due today, and the 09:30 ACDT expected run time are supplied screenshot context, not live Jira verification. A scheduled finish time is not evidence of completion.

## Short Phase 3 mechanism note for Alex

These are candidate mechanisms, not validated gains or claims of novelty. All must extend the same MPNN and use identical molecule/theory/splits, accuracy conventions and evaluation MC budget.

| Candidate | Minimal mechanism | Physics and force requirements | Current recommendation |
|---|---|---|---|
| Edge dropout | Bernoulli mask on edge messages; optional inverse-keep scaling | Couple reverse-edge masks for an undirected-interaction design; do not claim this alone guarantees physical forces. Keep the sampled mask fixed through one energy/gradient calculation. Hard graph changes and cutoff crossings can make geometry sensitivity discontinuous. | Useful simple ablation; avoid making it the sole demo method. |
| Continuous edge feature Gaussian noise | Add small Gaussian perturbation to RBF/message features, or multiply messages by `1 + sigma*epsilon`; state which | Preserve actual distances, angles, atomic identity and graph construction. Derive features from live coordinates. Noise/gates should preserve rotational symmetry; fixed latent draws must follow edges under permutation. Differentiate through coordinate-dependent gates. | Best direct continuation of Dongxiao's earlier baseline, but smaller noise is a hypothesis, not an established improvement. |
| Node feature perturbation | Perturb learned scalar node embeddings after each update, never atomic numbers | For scalar invariant embeddings, additive scalar-channel noise can preserve rotation invariance; vector channels require an equivariant noise construction. Fix draws while differentiating energy, and permute them with node identity for coupled symmetry checks. | Reuse Alex's existing `RepresentationNoise` first draft instead of implementing a duplicate. |

For every candidate, compute each sampled force as `F_s = -grad_R E(R, epsilon_s)` using the SAME draw as its energy. Redrawing randomness between energy and force invalidates this check. Check fixed-draw finite differences, translation/rotation behavior, atom permutation, and cutoff sensitivity before reporting physical consistency. A conservative sampled force does not by itself imply calibrated uncertainty or accurate dynamics. Zero-mean hidden noise does not guarantee an unbiased output after nonlinear layers.

Main already implements input-dependent additive node Gaussian noise: `h + softplus(gate(h))*softplus(log_sigma)*eps`, plus a KL regularizer. It is marked a first draft, not the team's final mechanism. A Gaussian-noise KL penalty alone does not establish equivalence to the full BLIP variational framework. BLIP background: https://arxiv.org/abs/2508.14022 and official code https://github.com/dario-coscia/blip .

## Rechecked earlier DimeNet++ evidence

Recomputed directly from saved predictions and targets, using the existing `metrics`/`uq_metrics` functions (no retraining). Dataset: ethanol CCSD(T) Gold, 111 validation structures, 9 atoms each. One shared pretrained checkpoint, fine-tuning seeds 42/43/44, 20 epochs per arm, sigma 0.05, 32 MC draws for the noisy arm. These are validation-selected results, not independent test evidence or three independent pretraining runs.

| Seed | Control E MAE | Noisy E MAE | Control F MAE | Noisy F MAE | Force UQ/error Spearman |
|---|---:|---:|---:|---:|---:|
|42|1.128260|0.460043|1.130940|1.270389|0.347016|
|43|0.837351|1.550670|1.133179|1.303981|0.354923|
|44|0.714532|0.426197|1.096088|1.276697|0.375132|

Units: E kcal/mol; F kcal/mol/Angstrom, averaging Cartesian components. Energy improved in 2/3 comparisons; force MAE worsened in all three. Mean paired relative force increase: **14.6269%**. Correlation type is confirmed **Spearman** in code and by recomputation.

Legacy force correlation is structure-level: mean MC component variance versus mean squared force error per structure. It is not the pooled force-component correlation used by the shared harness. Legacy coverage uses empirical MC energy quantiles at 50/80/90%, with equal-weight mean absolute coverage gap; it is not the same diagnostic as the shared Gaussian interval ECE. Do not compare their numbers as if identical.

Archive shapes/finite values and unique target frame IDs were checked. Prediction archives contain no frame IDs; code uses the same unshuffled dataset for every draw, but independent ID-level matching of saved predictions cannot be certified from those archives alone. Full recomputed values and file hashes are in `verified_uq.json`.

## Phase 2 — what is actually available

Fetched main/fazin contain `experiments/configs/phase2_blip_nll.yaml`, but **no committed `phase2_blip_nll` run directory, evaluated metrics, checkpoint or raw MC draws**. Therefore NLL prediction-target alignment, MC variability, coverage, ECE and Spearman remain **unverified / unavailable**, not zero. No before/after NLL improvement can be claimed.

Available baseline is the older **`phase2_blip`** committed `test_metrics.json`: ethanol DFT, test split, 55,509 frames, 9 atoms, model config MC=20. These values were read from the record; only the ECE arithmetic can be rechecked without its checkpoint/raw arrays.

| Quantity | MAE | Regression interval ECE | UQ/error Spearman | Mean predictive std |
|---|---:|---:|---:|---:|
|Energy total/frame|0.157671|0.703342|0.004670|0.027596|
|Force/component|0.206466|0.675730|0.273047|0.023092|

| Nominal central interval | Energy observed coverage | Force component observed coverage |
|---|---:|---:|
|50%|3.8156%|5.4510%|
|80%|7.5375%|10.3259%|
|90%|9.9660%|13.2238%|
|95%|12.3439%|15.7072%|

Under this stated Gaussian interval diagnostic, old BLIP intervals severely under-cover. This is not a result of the NLL retrain. Nonzero recorded spread is not proof that individual samples are correctly aligned or that MC estimates are stable.

## Definitions / SCRUM-35 reuse and integration

Use existing `ml/utils/uq_metrics.py`, `UQAccumulator`, `evaluate.py` and `benchmark.py`; do not create a competing module. Its definitions are explicitly provisional awaiting Dongxiao/team agreement. This note documents a proposed reporting convention, not sign-off on the user's behalf.

- **Regression interval ECE:** `mean_p |observed_p - p|`, p in {0.50,0.80,0.90,0.95}; central Gaussian interval `mu +/- Phi^-1((1+p)/2)*sqrt(max(var,1e-12))`. Equal weighting over levels; energy and force reported separately. This is an interval coverage gap, not classification ECE. Report interval assumption/grid alongside the name.
- **Spearman:** predictive std versus absolute error; energy one value/frame, force flattened frame/atom/Cartesian components. Constant std or error returns null. Do not treat correlated force components as independent inferential evidence. Add a separate structure-level force metric when selecting whole structures; keep both names explicit.
- **Coverage:** fraction of targets in inclusive central intervals. Distinguish interval coverage from retained fraction on a risk-coverage curve.
- **MC variability:** `predict_distribution` returns mean and unbiased sample variance across repeated energy/force predictions (`ddof=1`). Report MC count, seed, per-quantity std summaries, zero/tiny-variance fraction and stability versus MC count; positive variance alone does not establish useful UQ. Training K=4 and evaluation MC=20 are different settings.
- **Prediction-target alignment:** current contract checks counts and units, not frame identities. For the NLL export include frame IDs, molecule/theory, atom ordering, split/fingerprint and units beside predictions and references. Verify exact identity/order before computing metrics. Separately show parity/residuals and MAE; correlation alone is insufficient.
- **Important mapping:** `benchmark_row.ece` and `benchmark_row.uncertainty_correlation` currently select ENERGY metrics only. Include explicit energy/force columns in the demo note; do not label these generic fields as force UQ.
- **Fair before/after:** evaluate both checkpoints under the same gold fingerprint, split, batch/seed/MC policy and metric implementation. The old DimeNet CCSD(T) validation and team BLIP DFT test numbers must not be used as a head-to-head comparison.

Reusable `recheck_uq.py` runs the saved-array recheck and recomputes ECE from the committed coverage table. It imports the current shared ECE implementation and extracts the legacy metrics unchanged. It does not produce or simulate an NLL result.

## Demo-ready message for Alex / Fazin

I have completed a candidate-mechanism note and audited the available UQ evidence. The shared baseline is Raj's MPNN. Main already contains Alex's node-representation Gaussian-noise draft and Fazin's reusable UQ module, so I recommend reusing those rather than duplicating them. My earlier DimeNet++ edge-noise runs improved validation energy in two of three seeds but increased force MAE in all three, by 14.63% on average; the 0.347–0.375 force correlations are confirmed structure-level Spearman. They are supporting evidence, not MPNN results.

The uploaded Phase 2 metrics belong to the old phase2_blip run, not phase2_blip_nll. Old energy/force interval ECE is 0.7033/0.6757; Spearman is 0.0047/0.2730 under the shared Gaussian, component-level convention. NLL results are still pending verification because the finished checkpoint and evaluation outputs are not in the fetched repository. Please provide its run.json, checkpoint, test_metrics.json and aligned mean/variance or MC exports. The benchmark's generic ECE/correlation fields currently show energy only. We can demonstrate the offline audit and mechanism design now; we should not claim a validated NLL improvement or a real-checkpoint web demo yet.

## Completion boundary

Verification: the existing UQ, shared evaluation and graph stochastic model suites passed **57 tests** on the isolated current-main checkout. Initial attempts hit the sandbox temporary-folder restriction and a missing file in the sparse checkout; setting a workspace temp folder and including the required existing script resolved both. This checks code behavior, not the missing NLL experiment. The saved-array recheck also passed and reproduced all three reported Spearman values.

SCRUM-34 candidate note and available-evidence audit are delivered. NLL acceptance and final calibration-definition approval remain open. No Jira statuses were changed; no message was sent to Alex/Fazin. A Codex handoff may be synchronized to the existing capstone chat under the user's request.
