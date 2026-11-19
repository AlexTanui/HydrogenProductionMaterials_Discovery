# Literature Review: Molecular Message Passing and Bayesian Interatomic Potentials

Prepared for SCRUM-33 | Dongxiao Wu | Updated 19 November 2026

## Scope and conclusion

This final SCRUM-33 review grounds Phase 3's mechanism in the literature and evaluates that reasoning against the project's actual outcomes. It covers what worked, what failed, concrete improvements, and delivery status. The paper summaries below are followed by a retrospective; proposed mechanisms are not presented as if they had been selected prospectively before implementation.

The supported conclusion is a trade-off, not universal Phase 3 superiority: graph-space stochasticity is a defensible extension of the molecular message-passing backbone, but its placement and training objective materially affect energy accuracy and force uncertainty. The project identified a calibration failure, tested a training mitigation, and implemented a structural message-level repair. Research limitations remain part of the conclusion rather than reasons to leave this literature-review deliverable indefinitely open.

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

## 3. Coscia et al.: Bayesian grounding and limits

The relevant paper is **BLIPs: Bayesian Learned Interatomic Potentials**, by Coscia, de Haan, and Welling; this review uses arXiv version 1, dated 19 August 2025. It is unrelated to the vision-language model sharing the BLIP abbreviation.

BLIPs introduces variational Bayesian message and update networks. Input-dependent coefficients control Gaussian weight uncertainty; local reparameterisation permits activation sampling. Training balances data fit and a posterior-prior KL term through an evidence lower bound. Multiple stochastic predictions support uncertainty estimation. The paper reports improvements in its studied systems, which do not establish calibration for our dataset automatically. Its symmetry discussion distinguishes distributional behaviour from individual noisy forward passes and states restrictions on supported equivariant layers. [3]

**Project implication:** Adding fixed Gaussian feature noise is not sufficient to claim a BLIP reproduction. Review the posterior parameterisation, inference network, prior, KL scaling, likelihood, and sampling procedure against the paper and author implementation before making that claim.

## 4. Phase 3 mechanism: rationale and evolution

The project asks whether uncertainty can be represented in learned graph features rather than sampled weights while retaining an MPNN energy/force model. Gilmer's separation of message and update functions provides explicit intervention points; SchNet motivates geometry-aware interactions and forces obtained from an energy gradient; BLIPs motivates learning input-dependent uncertainty rather than attaching a constant error bar. These ideas motivate an experiment, not a theorem that feature noise improves calibration. The project's MPNN BLIP implementation and the separately reported paper-faithful PaiNN reproduction should not be treated as identical implementations.

The original Phase 3 perturbed node representations after aggregation and update:

```text
h_noisy = h + s(h) * epsilon
```

This was economical to integrate into the shared backbone, with a learned input-dependent scale and a KL term covering the gated scale. However, energy reads the resulting representation while force differentiates the energy with respect to coordinates. A noisy representation therefore need not generate useful force uncertainty.

The subsequent structural repair moves noise to messages before aggregation:

```text
m_noisy = m * (1 + s(m) * epsilon)
```

Messages depend on live interatomic distances. Holding a draw fixed, the perturbation derivative includes `s * epsilon * dm/dR` as well as `m * epsilon * ds/dR`. This supplies an explicit multiplicative coupling to the message Jacobian. It is the reason to prefer this repair over simply inflating the previous scale. The old additive formulation still has `epsilon * ds/dR`, and downstream nonlinearities matter: `d(epsilon)/dR = 0` does not mean all force randomness vanishes, nor prove that the remaining derivative must be weak. The mechanism diagnosis should be evaluated through experiments, not an overstated derivative argument.

Of the alternatives discussed in SCRUM-34, continuous message noise preserves the intended graph-space experiment without deleting interactions. Edge dropout remains an ablation rather than the chosen repair because discrete edge changes complicate geometry sensitivity. A separate force-uncertainty head could provide more flexibility, but would change the interpretation of one energy-derived stochastic model. The message-level choice is therefore a reasonable scoped repair; it is not evidence that the alternatives are universally inferior. [P1, P3, P4]

## 5. What worked, and why

**Shared interfaces and evaluation infrastructure.** Phase 1 established five architectures; Phases 2 and 3 then extended the MPNN family through common data, registry, training, and scoring interfaces. This made differences inspectable and allowed review to catch mismatched training hooks. Keeping molecule/theory identity, Gold splits, units, and evaluation MC counts explicit was more defensible than comparing unrelated notebook outputs. A shared harness improves comparability without making unequal model capacities a one-factor causal experiment. [P2, P5]

**Energy and force were evaluated separately.** This exposed a failure that an energy-only review would miss. On the recorded full-test comparison, moving from Graph base to Graph NLL reduced force ECE from approximately 0.2969 to 0.0277; force coverage at nominal 90% rose from approximately 57% to 86%. Learning a distribution with the NLL objective was useful because mean-error optimisation alone does not directly reward appropriate predicted variance. This was a training mitigation, not the later architectural repair. [P2]

**The review led to a concrete structural change.** The diagnosis was followed by multiplicative message noise in commit `633d1f2`, rather than only post-hoc rescaling. The team reports that this version was implemented and scored, and that it restored direct force coupling while replacing the original undercoverage with milder overcoverage. The commit records 446 passing ML tests, two skips and three expected failures. These are upstream-reported checks, not tests rerun for this document. Exact new calibration values are not reproduced here because a corresponding structural-fix metrics JSON was not located in the inspected tracked files. [P3, P4]

**Auditable results and a working platform.** Saved metrics, checkpoint fingerprints, rescore records, and a real dashboard/API made the work demonstrable and reviewable. Integration also caught a stale benchmark report relative to a backend-launched run. Correcting the reporting path was valuable because a live interface is only useful if the experiment displayed is the experiment actually executed. [P4, P5]

## 6. What failed, and why

**Noise alone did not produce calibrated uncertainty.** Graph base overcovered energy and undercovered forces. A KL penalty did not guarantee coverage, and increasing MC draws could estimate a poor predictive distribution more precisely without fixing its misspecification. The MC study did not establish its stability criterion over the tested seeds and K values; this is not proof that larger K can never help. [P2, P6]

**Better force calibration did not imply better energy prediction.** In the historical matched NLL comparison, Graph NLL energy MAE was 0.1292 versus BLIP NLL's 0.0750 kcal/mol, and force-component MAE was 0.1846 versus 0.1787 kcal/mol/Angstrom. Thus, Graph NLL's much lower raw force ECE came with worse point accuracy. These figures belong to the additive-node NLL checkpoints, not the new multiplicative-message repair. The experiment establishes an observed trade-off; it does not establish an unavoidable physical law. [P2]

**The early comparison was not fully matched.** BLIP NLL used four training draws and 20 warmup epochs, whereas Graph base used one draw and no warmup. The later matched NLL comparison changed the interpretation substantially. Comparing the same test frames was necessary but insufficient for attributing differences to noise placement. [P2]

**Post-hoc calibration was not an architectural repair.** Scalar and CQR corrections changed interval coverage and width, not the learned point predictor or the force pathway. Reusing validation data for calibration after model selection also does not demonstrate an independent calibration protocol. MD trajectory dependence and pooled force-component dependence limit exchangeability claims. Observed near-nominal coverage must not be presented as a proven finite-sample guarantee for this setup. [P2]

**Earlier noise evidence did not transfer automatically.** Dongxiao's separate ethanol CCSD(T) DimeNet++ fine-tuning comparison improved energy in two of three seeds but worsened force MAE in all three (mean paired increase about 14.6%). This justified investigating both channels, not assuming Gaussian noise would help the MPNN DFT experiment. The architectures, theory levels, and evaluation conventions differ. [P1]

**Documentation lagged implementation.** The literature review initially described the papers without resolving their implications for the implemented mechanism. Some source comments and reports also retained superseded status or metric wording. This made handoffs harder and confused a closed ticket with a merged or deployed model. This retrospective explicitly repairs that gap rather than claiming the completed review drove every earlier decision.

## 7. What could be done better next time

| Improvement | Concrete practice | Purpose |
|---|---|---|
| Agree mechanisms before building them | Record a short decision with formula, injection location, expected derivative behaviour, owner, and acceptance checks before training. | Connect literature to a testable design rather than retroactive justification. |
| Test physics and uncertainty together | Add coupled-draw rotation/translation/permutation and finite-difference checks, plus checks of non-degenerate force variance; evaluate calibration separately. | Detect gradient-path problems without confusing symmetry tests with calibration guarantees. |
| Predeclare controlled comparisons | Fix training draws, warmup, loss, data, selection rules and compute budget; record any intended deviations. | Reduce confounding and avoid changing the headline after results are known. |
| Separate data roles | Reserve calibration data before model selection, preserve exact IDs and trajectory boundaries, and assess dependence or independent-trajectory evaluation. | Make calibration assumptions and leakage checks auditable. |
| Quantify uncertainty in comparisons | Use independent training seeds and aligned per-frame predictions; choose dependence-aware block analysis only after checking trajectory structure. | Avoid treating MC draws or Cartesian components as independent replications. |
| Improve calibration validation | Test finite-sample quantile ranks and small-sample edge cases; compare coverage and width per quantity; keep scalar and CQR evaluation populations labelled. | Prevent numerically plausible intervals from masking invalid guarantees. |
| Make reports traceable to deployments | Generate tables from immutable run records and display model version/checkpoint identity in the platform; regenerate reports when runs change. | Prevent stale tables and accidental attribution of old scores to new code. |
| Close documentation with implementation | Update the review, model card, handoff and ticket evidence together; distinguish implementation, scoring, merge and deployment status. | Preserve contribution attribution and avoid overdue documentation becoming the final delivery blocker. |

These are follow-up lessons, not a requirement to perform another research programme before accepting SCRUM-33.

## 8. Where the project stands

As reported by the team on 19 November, all other tickets are closed and SCRUM-33 is the remaining documentation/sign-off item. Phases 1-3 have been implemented and scored: Phase 1 covers MPNN, SchNet, PaiNN, DimeNet++ and PhysNet; Phase 2 studies weight-space uncertainty; Phase 3 studies graph-space uncertainty. A working integrated platform supports real prediction, benchmarking, training/jobs and report generation. This is a methodology evaluated on molecular benchmarks, not a validated hydrogen-catalyst screening system.

The Phase 3 calibration issue was diagnosed and addressed in two stages: NLL training mitigated the observed force-calibration failure; the later multiplicative-message change repaired the explicit coupling mechanism. The team's revised Sprint 6 record reports that the structural version was scored and SCRUM-70 closed. It also states that the demonstration still uses the earlier deployed checkpoint pending the architecture owner's review. Consistently, the inspected `alex` snapshot contains the structural change, while the inspected `main` snapshot still contains additive node noise. Ticket closure therefore must not be described as proof that the repaired checkpoint is already deployed everywhere. [P3, P4]

The contribution is a completed three-phase implementation and evaluation, an operational demonstration platform, and a documented negative result leading to a concrete repair. The evidence does not support universal Phase 3 superiority, perfect calibration, or generalisation to unseen molecules and catalyst systems. Those scientific boundaries can coexist with completion of this review.

## 9. Evidence and acceptance mapping

Reviewed 19 November 2026: `alex` at `8e1024195c8b8cb341f8787cc95fa4a457ccb768`; `main` at `17c7e7fa623e524f68814a7c58b76d5b25648c8b`. Source narrative, directly inspected implementation, saved metrics, and current team status are distinguished above. No new training, scoring, deployment, or Jira mutation was performed for this update.

| Requested closure content | Location |
|---|---|
| What worked and why | Sections 4-5 |
| What failed and why | Section 6 |
| Concrete improvements | Section 7 |
| Actual project state | Section 8 |
| Paper grounding and traceable evidence | Sections 1-3 and references |

This document completes the requested SCRUM-33 write-up and is ready for the team's closure/sign-off. It does not claim a complete author-code reproduction audit of BLIPs.

### Project evidence

- [P1: Graph-space candidate review and DimeNet++ evidence](dongxiao_sprint4/literature_review.md).
- [P2: Verified Phase 3 comparison and calibration review](dongxiao_scrum36/alex_review_2026_11_16/Review.md), with source hashes and original run records; [technical report](technical_report.md).
- [P3: Structural repair commit and recorded test summary](https://github.com/AlexTanui/HydrogenProductionMaterials_Discovery/commit/633d1f26bfca43d9ac7441e276d7faa0a63bf5ae).
- [P4: Revised Sprint 6 delivery and deployment narrative](https://github.com/AlexTanui/HydrogenProductionMaterials_Discovery/blob/8e1024195c8b8cb341f8787cc95fa4a457ccb768/docs/reviews/sprint6-presentation-script-2026-11-16.md).
- [P5: Upstream technical report](https://github.com/AlexTanui/HydrogenProductionMaterials_Discovery/blob/8e1024195c8b8cb341f8787cc95fa4a457ccb768/docs/technical_report.md). Some historical status wording is stale; newer dated records take precedence.
- [P6: MC stability study](https://github.com/AlexTanui/HydrogenProductionMaterials_Discovery/blob/8e1024195c8b8cb341f8787cc95fa4a457ccb768/docs/reviews/scrum-49-mc-stability-2026-11-14.md).

## References

[1] Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., and Dahl, G. E. (2017). *Neural Message Passing for Quantum Chemistry*. ICML, PMLR 70, 1263-1272. [Proceedings and paper](https://proceedings.mlr.press/v70/gilmer17a.html).

[2] Schutt, K. T., Kindermans, P.-J., Sauceda, H. E., Chmiela, S., Tkatchenko, A., and Muller, K.-R. (2017). *SchNet: A continuous-filter convolutional neural network for modeling quantum interactions*. Advances in Neural Information Processing Systems 30. [Proceedings and paper](https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html). This citation refers to the 2017 conference paper.

[3] Coscia, D., de Haan, P., and Welling, M. (2025). *BLIPs: Bayesian Learned Interatomic Potentials*. arXiv:2508.14022v1. [Version used for this review](https://arxiv.org/html/2508.14022v1). [Author implementation](https://github.com/dario-coscia/blip).
