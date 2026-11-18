# Literature Review: Molecular Message Passing and Bayesian Interatomic Potentials

Prepared for SCRUM-33 | Dongxiao Wu | 15 November 2026

## Scope and reading status

This review provides Phase 1 background from Gilmer et al. (2017) and Schutt et al. (2017), followed by an initial methodological reading of BLIPs. It separates published methods from project recommendations. It does not certify a complete reproduction, introduce new experimental results, or change Jira status.

## 1. Gilmer et al. (2017): the MPNN framework

Gilmer and colleagues unify graph-based molecular predictors through a message-passing phase and a graph-level readout. At each layer, a node aggregates information from neighbours and updates its representation:

```text
m_i^(t+1) = sum_j M_t(h_i^t, h_j^t, e_ij)
h_i^(t+1) = U_t(h_i^t, m_i^(t+1))
y_hat = R({h_i^T})
```

Here, `h` denotes node features, `e` denotes edge attributes, and the readout must be invariant to node ordering. MPNN describes a family of architectures rather than one fixed implementation. The study evaluates molecular property prediction on QM9, including settings with molecular geometry. Its benchmark results should not be interpreted as evidence of performance on our molecular dynamics trajectories. [1]

**Project implication:** Use the message, update, and readout decomposition when explaining Raj's baseline and locating proposed stochastic modifications. Record the actual implementation and configuration before calling two experiments the same backbone.

## 2. Schutt et al. (2017): SchNet

SchNet uses continuous-filter convolutions to represent interactions between atoms at continuous positions. Distance-dependent filters allow geometric information to influence learned atomic representations without discretising positions onto an image grid. Atomic contributions produce a total energy, and forces are obtained by differentiating that energy:

```text
F_i = -partial E_hat / partial r_i
```

The architecture is designed for invariant energies, equivariant forces, and a smooth, conservative force field. The paper examines equilibrium molecules and molecular dynamics settings, including chemical and conformational variation in ISO17. SchNet is a specific architecture, whereas MPNN is a broader framework. [2]

**Project implication:** Evaluate both energy and force errors. Improving scalar energy predictions alone is insufficient evidence that the learned energy surface has better derivatives. For stochastic experiments, keep a sampled latent draw fixed during an energy/force calculation and test geometric behaviour explicitly.

## 3. Coscia et al.: initial BLIPs reading

The relevant paper is **BLIPs: Bayesian Learned Interatomic Potentials**, by Coscia, de Haan, and Welling; this review uses arXiv version 1, dated 19 August 2025. It is unrelated to the vision-language model sharing the BLIP abbreviation.

BLIPs introduces variational Bayesian message and update networks. Input-dependent coefficients control Gaussian weight uncertainty; local reparameterisation permits activation sampling. Training balances data fit and a posterior-prior KL term through an evidence lower bound. Multiple stochastic predictions support uncertainty estimation. The paper reports improvements in its studied systems, which do not establish calibration for our dataset automatically. Its symmetry discussion distinguishes distributional behaviour from individual noisy forward passes and states restrictions on supported equivariant layers. [3]

**Project implication:** Adding fixed Gaussian feature noise is not sufficient to claim a BLIP reproduction. Review the posterior parameterisation, inference network, prior, KL scaling, likelihood, and sampling procedure against the paper and author implementation before making that claim.

## 4. Application to our project

The following are proposed evaluation requirements, not experimental findings from the papers.

| Project component | Review requirement |
|---|---|
| Phase 1 baseline | Identify the exact model, checkpoint, preprocessing, energy units, and saved sample IDs. |
| Phase 2 uncertainty | Describe where randomness enters and distinguish predictive error, uncertainty ranking, and interval coverage. |
| Phase 3 graph perturbations | State whether the intervention changes edges, scalar node features, or edge features; do not label every noise mechanism Bayesian. |
| Cross-phase comparison | Use matching molecule, theory level, splits, units, metric definitions, and evaluation sampling budget; disclose remaining configuration differences. |

Retain the existing approved Gold splits. Fit preprocessing on training data only. Do not combine the earlier ethanol CCSD(T) DimeNet++ study with the team's ethanol DFT comparison as if they were one benchmark. Preserve sample and trajectory identifiers: non-overlapping IDs alone do not remove correlation between nearby molecular dynamics frames.

Report energy MAE and RMSE separately from force-component metrics. For uncertainty, report nominal versus empirical coverage and interval width, with the interval construction stated explicitly. A useful uncertainty/error correlation is not a coverage guarantee. Any later conformal calibration experiment needs a suitable held-out calibration set and a documented treatment of trajectory dependence; this review does not establish those conditions.

Use existing saved evidence for the Phase 3 comparison and explicitly report cases where the method does not outperform BLIP. Avoid attributing an improvement to noise alone when model size, loss weighting, optimisation, or checkpoint selection also changed.

## 5. Follow-up and evidence boundaries

- The foundational review and initial BLIP method summary are documented here.
- A line-by-line reproduction audit against the BLIP author code remains separate work.
- This documentation change does not train models or validate new performance claims.
- Review the terminology with the team before marking SCRUM-33 accepted.

Related project records:

- [Graph-space candidate review and earlier DimeNet++ evidence](dongxiao_sprint4/literature_review.md)
- [Phase comparison technical report](technical_report.md)

## References

[1] Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., and Dahl, G. E. (2017). *Neural Message Passing for Quantum Chemistry*. ICML, PMLR 70, 1263-1272. [Proceedings and paper](https://proceedings.mlr.press/v70/gilmer17a.html).

[2] Schutt, K. T., Kindermans, P.-J., Sauceda, H. E., Chmiela, S., Tkatchenko, A., and Muller, K.-R. (2017). *SchNet: A continuous-filter convolutional neural network for modeling quantum interactions*. Advances in Neural Information Processing Systems 30. [Proceedings and paper](https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html). This citation refers to the 2017 conference paper.

[3] Coscia, D., de Haan, P., and Welling, M. (2025). *BLIPs: Bayesian Learned Interatomic Potentials*. arXiv:2508.14022v1. [Version used for this review](https://arxiv.org/html/2508.14022v1). [Author implementation](https://github.com/dario-coscia/blip).
