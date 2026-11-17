# Graph-space stochasticity: candidate review (SCRUM-34)

Draft for team review, 22 October 2026. Common baseline: Raj MPNN. Candidate design guidance; no Phase 3 accuracy claims.

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
