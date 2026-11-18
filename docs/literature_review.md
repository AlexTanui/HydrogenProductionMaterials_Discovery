# Literature Review and Project Findings — PG-S2-47 (Hydrogen Production Materials Discovery / Adaptive Graph-Space Uncertainty Modelling for MD17)

Prepared for SCRUM-33, extended beyond its original Phase-1-grounding scope
(Gilmer/SchNet/BLIP, Dongxiao Wu, 15 Nov 2026) to cover the full roster of
architectures actually trained (§1), the stochastic-GNN and calibration
literature the team drew on for Phase 2/3 and post-hoc UQ (§2-3), and the
project's own findings end to end — data preparation through Phase 3
(§4-8). §1.1-1.3, §2.3 and the "project implication" notes under them are
Dongxiao's original text (15-16 Nov), reused verbatim where the project has
not moved past what she wrote; everything else is new as of 18 Nov 2026.

**What this is and isn't.** This is a review of the methods the project
used and a record of what was found when they were applied to this data,
not a new, independent research contribution — the papers are read for
what they let the team claim and what they don't, and the project findings
are reported with their own caveats intact rather than smoothed into a
single clean narrative. Several findings below are genuinely unresolved
(flagged as such); this document doesn't resolve them, it states them
precisely enough that resolving them is a well-defined task.

---

## 1. Architecture grounding — the five Phase 1 models

Phase 1's job was a fair bake-off of five architectures on one shared
protocol (`ml/training/train.py`, same data/splits/optimiser/budget), to
settle which backbone to carry into Phases 2/3. The five models and their
grounding:

### 1.1 Gilmer et al. (2017): the MPNN framework

Gilmer and colleagues unify graph-based molecular predictors through a
message-passing phase and a graph-level readout. At each layer, a node
aggregates information from neighbours and updates its representation:

```text
m_i^(t+1) = sum_j M_t(h_i^t, h_j^t, e_ij)
h_i^(t+1) = U_t(h_i^t, m_i^(t+1))
y_hat = R({h_i^T})
```

Here, `h` denotes node features, `e` denotes edge attributes, and the
readout must be invariant to node ordering. MPNN describes a family of
architectures rather than one fixed implementation. The study evaluates
molecular property prediction on QM9, including settings with molecular
geometry. Its benchmark results should not be interpreted as evidence of
performance on our molecular dynamics trajectories. [1]

**Project implication:** Use the message, update, and readout
decomposition when explaining Raj's baseline and locating proposed
stochastic modifications. Record the actual implementation and
configuration before calling two experiments the same backbone.

`ml/models/mpnn.py` is this project's own Gilmer-style instance: message
and update MLPs over a Gaussian/RBF-expanded distance edge feature (16
bins, §4 below), total energy from a sum-pooled readout, forces via
autograd (`F = -∂E/∂R`) rather than a second output head — so forces
inherit whatever the energy surface's smoothness is, which matters
directly for §7's Phase 3 findings.

### 1.2 Schütt et al. (2017): SchNet

SchNet uses continuous-filter convolutions to represent interactions
between atoms at continuous positions. Distance-dependent filters allow
geometric information to influence learned atomic representations without
discretising positions onto an image grid. Atomic contributions produce a
total energy, and forces are obtained by differentiating that energy:

```text
F_i = -partial E_hat / partial r_i
```

The architecture is designed for invariant energies, equivariant forces,
and a smooth, conservative force field. The paper examines equilibrium
molecules and molecular dynamics settings, including chemical and
conformational variation in ISO17. SchNet is a specific architecture,
whereas MPNN is a broader framework. [2]

**Project implication:** Evaluate both energy and force errors. Improving
scalar energy predictions alone is insufficient evidence that the learned
energy surface has better derivatives. For stochastic experiments, keep a
sampled latent draw fixed during an energy/force calculation and test
geometric behaviour explicitly.

This project's `ml/models/registry.py` `"schnet"` entry is a faithful
continuous-filter implementation per this paper, used as Phase 1's
accuracy floor — it is clearly the weakest model on every accuracy metric
in §5's table despite being fastest to train, which is consistent with its
design intent (a simple, efficient baseline) rather than a defect.

### 1.3 PaiNN — Schütt, Unke and Gastegger (2021)

*Equivariant message passing for the prediction of tensorial properties
and molecular spectra* [4]. PaiNN extends the continuous-filter idea with
explicit equivariant vector (not just scalar) node features, updated by
message-passing layers that mix scalar and vectorial channels while
preserving rotational equivariance exactly (not just approximately, as a
scalar-only model relying on invariant distance features does implicitly
through the energy-gradient construction). This buys more expressive
geometric representations per layer than SchNet's purely scalar filters,
at the cost of a larger parameter count per layer.

**Why this matters here:** PaiNN had the lowest force MAE/RMSE of all five
Phase 1 models in this project's bake-off (§5) — consistent with the
paper's own emphasis on PaiNN's suitability for force-sensitive,
spectroscopy-adjacent properties, which force prediction on MD17 directly
is. It was not, however, picked as the Phase 2/3 backbone (§5 explains
why): winning the accuracy bake-off and being the right base for the
project's stochasticity design are different questions.

### 1.4 DimeNet / DimeNet++ — Gastegger et al. (2020)

*Fast and Uncertainty-Aware Directional Message Passing for
Non-Equilibrium Molecules* [5], extending the original DimeNet (Gasteiger,
Groß and Günnemann, 2020, *Directional Message Passing for Molecular
Graphs*). DimeNet's distinguishing idea is using **directional**
information — angles between triplets of atoms, not just pairwise
distances — via spherical Bessel and harmonic basis functions, so the
model can distinguish geometrically different configurations that have
the same set of pairwise distances (a real ambiguity pairwise-distance-only
models like plain SchNet cannot resolve in general). DimeNet++ is an
efficiency rework of the same directional idea (faster envelope/basis
functions, a bilinear rather than Hadamard interaction layer) aimed
specifically at non-equilibrium geometries — i.e. molecular dynamics
trajectories away from the energy minimum, which is exactly MD17's
regime, not just relaxed structures.

**Why this matters here:** DimeNet++ had this project's lowest energy MAE
of all five Phase 1 models (§5) — plausibly because angular information
disambiguates exactly the kind of near-equilibrium geometric variation MD
trajectories explore, which the paper's own stated design target predicts.

### 1.5 PhysNet — Unke and Meuwly (2019)

*PhysNet: A Neural Network for Predicting Energies, Forces, Dipole
Moments, and Partial Charges* [6]. PhysNet is a SchNet-family continuous-
filter model extended with explicit physical terms added to the learned
short-range energy: a damped electrostatic (Coulomb) term over learned
partial charges and a D3-style dispersion correction, so the total
predicted energy is `E = E_short-range(NN) + E_electrostatic + E_dispersion`
rather than a single opaque NN output. The paper's stated motivation is
that long-range electrostatics and dispersion are physically well
understood and poorly learned by a purely local message-passing stack
with a finite cutoff radius.

**Why this matters here:** this project's MD17 molecules are small
(9-24 atoms, §4) and single-conformer, so long-range electrostatic/
dispersion effects are a much smaller share of the total energy variance
than they would be for e.g. a larger catalytic system — consistent with
PhysNet landing mid-pack in the Phase 1 bake-off (§5) rather than showing
a decisive advantage: the physical terms it adds are most valuable exactly
where this benchmark's small, short-range molecules provide the least
signal to show it off. This is a useful caveat for the project's stated
long-term aim (CLAUDE.md: validating methodology before pointing it at
real catalyst screening) — catalytic systems are larger and more likely to
benefit from PhysNet's explicit long-range terms than MD17 does.

---

## 2. Stochastic / Bayesian GNN literature — Phase 2 and 3 grounding

### 2.1 Coscia, de Haan and Welling: initial BLIPs reading

The relevant paper is **BLIPs: Bayesian Learned Interatomic Potentials**,
by Coscia, de Haan, and Welling; this review uses arXiv version 1, dated
19 August 2025. It is unrelated to the vision-language model sharing the
BLIP abbreviation.

BLIPs introduces variational Bayesian message and update networks.
Input-dependent coefficients control Gaussian weight uncertainty; local
reparameterisation permits activation sampling. Training balances data fit
and a posterior-prior KL term through an evidence lower bound. Multiple
stochastic predictions support uncertainty estimation. The paper reports
improvements in its studied systems, which do not establish calibration
for our dataset automatically. Its symmetry discussion distinguishes
distributional behaviour from individual noisy forward passes and states
restrictions on supported equivariant layers. [3]

**Project implication:** Adding fixed Gaussian feature noise is not
sufficient to claim a BLIP reproduction. Review the posterior
parameterisation, inference network, prior, KL scaling, likelihood, and
sampling procedure against the paper and author implementation before
making that claim.

`ml/models/blip.py` (Phase 2, §6 below) follows this mechanism: Gaussian,
input-dependent weight noise in the message/update layers, with local
reparameterisation and a KL term, evaluated via multiple stochastic
forward passes — this project's own implementation, not the authors' code,
so §6's findings are a test of this project's BLIP, not a reproduction
claim against the paper's own reported numbers.

### 2.2 The graph-stochasticity design space considered for Phase 3

CLAUDE.md and glossary.md frame Phase 3 as moving stochasticity from BLIP's
*weights* into node/edge *representations*, with the exact mechanism left
as "a team decision from the literature review." The design space
surveyed, beyond BLIPs itself:

- **DropEdge** (Rong et al., 2020) and **DropConn/DropNode**-style methods
  randomly remove edges or connections during training/inference as a
  regulariser; repeated stochastic forward passes with different dropped
  structure give a cheap ensemble-like uncertainty signal, but the
  stochasticity is structural (which edges exist) rather than a
  continuous perturbation of representation *values* — a fundamentally
  coarser knob than a Gaussian perturbation.
- **GRAND** (Feng et al., 2020, *Graph Random Neural Network*) randomly
  propagates/augments node features before a shared, deterministic
  classifier, combined with a consistency-regularisation training
  objective across augmentations — closer in spirit to "noise in
  representations, not weights" than DropEdge, but designed for node
  classification robustness, not regression uncertainty quantification;
  its consistency loss doesn't directly transfer to an energy/force NLL
  objective.
- **Edge-variational GCNs** treat edge weights/features as random
  variables with a learned variational posterior (structurally the
  edge-space analogue of BLIP's weight-space posterior), which is the
  closest prior-art shape to what `ml/models/graph_stochastic.py`
  actually does — the team's implementation (perturbing the *message*,
  itself a function of edge features, rather than the edge feature or the
  weight directly) is a variant of this family rather than a direct
  instance of any single one of the four methods above.

**What the literature review actually settled, honestly stated:** no
single paper above specifies "inject input-dependent Gaussian noise into
the node representation after aggregation" (the project's original Phase
3 mechanism, §7) or "into the message before aggregation" (the SCRUM-70
structural fix, §7) — the team's Phase 3 design is informed by this
literature's general shape (stochasticity somewhere in the forward pass
other than the weights) rather than a direct port of any one method, which
is also why the project had to discover empirically, rather than read off
a paper, that *where* in the forward pass the noise enters changes which
output channel (energy vs. force) it reaches at first order (§7).

### 2.3 Scope boundary (Dongxiao, 15 Nov)

- The foundational review and initial BLIP method summary (§1.1, §1.2,
  §2.1) are the only parts of this document in that original scope.
- A line-by-line reproduction audit against the BLIP author code remains
  separate work, not attempted here or elsewhere in the project as of
  this writing.
- No documentation change in this file trains models or validates new
  performance claims by itself — §4-8's findings are each sourced to a
  specific run record, report, or test, cited inline.

---

## 3. Calibration and conformal prediction literature — post-hoc UQ

Both post-hoc calibration methods used in `ml/training/calibrate.py` and
reported in §8 come from the conformal-prediction and calibration
literature rather than being bespoke to this project:

- **Scalar/Gaussian recalibration** (Kuleshov, Fenner and Ermon, 2018,
  *Accurate Uncertainties for Deep Learning Using Calibrated Regression*):
  rescales a model's predicted Gaussian standard deviation by a single
  learned scalar factor, fit on a held-out set, so that the *empirical*
  coverage at each nominal confidence level matches the *nominal* level as
  closely as a one-parameter correction can. It assumes the predictive
  distribution's *shape* (Gaussian) is already right and only its *width*
  is systematically off — a strong assumption this project's own findings
  repeatedly show is not satisfied (§6-8: every raw model here is either
  badly under- or over-covered on at least one output, in a way a single
  scalar can mask the average of without being correct to every input).
- **Conformalised Quantile Regression / CQR** (the project's working
  reference is Bao et al., 2025; building on the original CQR of Romano,
  Patterson and Candès, 2019, *Conformalized Quantile Regression*):
  distribution-free — it does not assume Gaussianity — and instead
  calibrates an interval's *width* directly from the empirical quantiles of
  calibration-set residuals, with a finite-sample marginal coverage
  guarantee under exchangeability. The project's own §3-caveat-1 and
  §8 findings note that this project's MD frames are *not* exchangeable
  (adjacent trajectory frames are near-duplicates, the same reason splits
  are block-contiguous rather than random, §4) — so even CQR's
  distribution-free guarantee is not rigorously applicable here without
  a block-aware calibration procedure the project has not yet built; CQR
  is used because it's a useful cross-check against a different set of
  assumptions than the scalar method, not because its guarantee formally
  holds on this data.

Running both methods against the same checkpoints, and finding that they
disagree by 13-33% on Phase 3's energy interval width (§8), is itself one
of this project's more useful findings: it means at least one of the two
methods' assumptions is being meaningfully violated on this data, and a
single calibration method's numbers alone would have hidden that.

---

## 4. Project findings — data preparation

Full detail: `docs/data_dictionary.md`, `docs/eda.md`
(notebook: `notebooks/eda_md17.ipynb`).

**Pipeline.** Bronze (raw, as-downloaded, Git-LFS-tracked) → silver
(validated, unit-checked, literature-split-preserving) → gold (final
train/val/test index arrays) → `MD17Dataset` (PyG graphs built at train
time: 5 Å cutoff, 16-bin RBF-expanded edge distances, never raw scalar
distance — matching Gilmer et al.'s MPNN design, §1.1). All 9 current MD17
bronze sources (ethanol×2 theories, benzene, azobenzene, paracetamol,
aspirin, malonaldehyde, toluene, uracil) — originally introduced by
Chmiela et al. [12] as a benchmark for energy-conserving force fields —
process cleanly end to end.
Theory level is part of dataset identity (`ethanol_dft` and
`ethanol_ccsdt` are never pooled); splits are contiguous trajectory blocks
or the literature-shipped split where one exists, never random — adjacent
MD frames are near-duplicate geometries, so a random split would leak and
inflate apparent accuracy.

**Two real bugs caught and fixed, not just theoretical risks:**

- **`__MACOSX` zip junk broke every zip-sourced dataset** (aspirin,
  malonaldehyde, toluene, CCSD(T)-ethanol) — macOS-zipped archives carry
  `__MACOSX/._<name>` AppleDouble entries that match a naive
  `endswith(".npz")` filter and crash `UnpicklingError` on load. Fixed in
  `ml/data/md17.py::_is_real_member`.
- **Unknown element symbols used to silently become atomic number 0**
  (not a real element) instead of raising — corrupts the graph silently
  rather than failing loudly. Fixed to raise deliberately.

**EDA findings** (`docs/eda.md`, all read from gold-stage data only, the
same discipline training uses): energy/force distributions across all 9
datasets are clean (unimodal, no corruption or outlier clusters beyond
what the programmatic NaN/shape/unit checks already catch) — a visual
confirmation of what the validation code asserts numerically, not a new
finding by itself. The one genuinely new finding from this pass: **the
default 5 Å cutoff radius is close to a no-op on MD17's small molecules**
— 100% of atom pairs in the 9-atom molecules (ethanol, malonaldehyde) fall
within 5 Å, meaning the "neighbor graph" is just the complete graph for
those systems and the cutoff provides zero locality/sparsification there;
it only starts doing real sparsification work on the larger MD17 molecules
(59% of pairs within cutoff for the 24-atom azobenzene). This matters
beyond MD17 itself: the same default **cannot be assumed to transfer
sensibly to MD22** (up to 370 atoms in the double-walled nanotube bronze
source, not yet processed by the pipeline), where a cutoff that's
essentially inert on a 9-atom molecule could be meaningfully — possibly
too — sparse, or fragment the graph outright, on something 40x larger.
Flagged as an open item (`docs/data_dictionary.md` §6) to resolve
empirically before any Phase 1-3 training touches MD22, not to inherit by
default. Split-integrity and the DFT-vs-CCSD(T) ethanol sanity check both
passed without surfacing new issues (full detail in `docs/eda.md` §2).

**Reading across to the modelling choices above:** the cutoff-radius
finding is a concrete instance of why §1.4's DimeNet++ argument (angular
information disambiguates near-equilibrium geometric variation) and §1.1's
RBF edge-feature argument (raw distance is a poor edge feature near
equilibrium bond lengths) matter in practice on *this* data specifically —
MD17's small molecules are exactly the regime where the graph is nearly
complete and the *edge features*, not graph sparsity, carry essentially
all of the geometric information the model has to work with.

---

## 5. Project findings — Phase 1 (deterministic bake-off)

Full table: `reports/phase1_benchmark.md`; model cards: `docs/model_cards/`.
All five architectures (§1) retrained under one shared protocol (fixed
data/splits/optimiser/200-epoch budget, `ml/training/train.py`), closing a
gap the Sprint 3 retrospective had flagged (prior numbers weren't
protocol-matched).

| Model | Parameters | Energy MAE | Energy RMSE | Force MAE | Force RMSE | Training time |
|---|---|---|---|---|---|---|
| MPNN | 384,897 | 0.0779 | 0.0962 | 0.1775 | 0.2774 | 0.63 h |
| DimeNet++ | 278,982 | **0.0537** | **0.0688** | 0.1142 | 0.1994 | 2.07 h |
| PaiNN | 588,033 | 0.0557 | 0.0711 | **0.1117** | **0.1858** | 1.35 h |
| PhysNet | 808,746 | 0.0611 | 0.0779 | 0.1738 | 0.2596 | 1.10 h |
| SchNet | 61,569 | 0.1714 | 0.2247 | 0.4270 | 0.6091 | 0.29 h |

Energy in kcal/mol, force in kcal/mol/Å per Cartesian component, both on the
shared 55,509-frame ethanol (DFT) test split. **DimeNet++ and PaiNN won on
raw accuracy, not MPNN** — matching §1.3-1.4's grounding (angular features
and equivariant vector channels both plausibly help on MD17's
near-equilibrium regime). SchNet is clearly weakest on every accuracy
metric despite being fastest and smallest, consistent with its role as a
simple baseline rather than the strongest model in this family (§1.2).
PhysNet lands mid-pack, consistent with §1.5's argument that its added
long-range physical terms have the least to contribute on small,
single-conformer MD17 molecules.

**MPNN was nonetheless chosen as the Phase 2/3 backbone** — not because it
won the bake-off (it didn't, on either metric), but for architectural
suitability to the two kinds of stochasticity the plan requires: a
Gilmer-style message-passing backbone gives a clean, well-understood place
to inject weight noise (Phase 2) and later representation/message noise
(Phase 3) without redesigning the base model twice, so that any
cross-phase difference in results is attributable to *where*
stochasticity lives rather than to three unrelated architectures. This is
a legible, stated trade — accuracy headroom traded for a controlled
scientific comparison — not an oversight, but it does mean Phase 2/3's
accuracy numbers (§6-7) start from a backbone that is not the most
accurate one available in this project's own bake-off.

---

## 6. Project findings — Phase 2 (BLIP, weight-space stochasticity)

`ml/models/blip.py`, grounded in §2.1. Two scored configurations: plain
MSE on a single stochastic forward pass (`phase2_blip`), and Gaussian NLL
training over multiple MC draws (`phase2_blip_nll`, `train_num_mc_samples
=4`, `nll_warmup_epochs=20` — intended as a calibration fix, mirroring the
training-time-NLL idea later also applied to Phase 3, §7).

| Variant | Energy MAE | Force MAE | Energy ECE | Force ECE | Energy cov@90% | Force cov@90% |
|---|---|---|---|---|---|---|
| BLIP MSE | 0.0920 | 0.1826 | 0.5964 | 0.6715 | 22.63% | 13.73% |
| BLIP NLL | 0.0750 | 0.1787 | 0.6470 | 0.7379 | 16.62% | 5.86% |

**NLL training improved BLIP's point accuracy** (both energy and force MAE
drop) but **made its calibration worse, not better, on both quantities** —
the opposite of its intended purpose. Root cause (Fazin, SCRUM-70 comment
10117, 31 Oct): the NLL loss used here applies a *variance floor* — a
minimum predicted spread, set once from epoch 20's training error and never
revisited for the remaining ~180 epochs — and for BLIP that floor is
**~610x the eventual squared energy error and ~83x the eventual squared
force error**, meaning the floor itself, not the model's learned
uncertainty, supplies almost all of the predicted spread for nearly the
whole training run. A fixed floor sized to an early, high training error
systematically *undersizes* predicted uncertainty once the model has
converged further than it had at epoch 20 — exactly the badly-undercovered
pattern above (16.62%/5.86% coverage against a 90% nominal target, i.e.
roughly 5-15x too narrow). This is a training-mechanism finding, not a
BLIP-architecture-specific one — §7 finds the same floor mechanism
operating on Phase 3, with a different (and informative) quantitative
effect.

Separately, `phase2_blip`'s own checkpoint was trained from a **dirty
working tree** and is excluded from official comparison tables by the
harness's clean-commit guard (SCRUM-48/69) pending a clean retrain — an
open item, not yet resolved as of this document (CLAUDE.md "current
state").

---

## 7. Project findings — Phase 3 (graph-space stochasticity)

`ml/models/graph_stochastic.py`, grounded in §2.2. Three scored
configurations, each isolating one variable against the last:

| Variant | Energy MAE | Force MAE | Force ECE | Force cov@90% |
|---|---|---|---|---|
| Graph MSE (original mechanism) | 0.0897 | 0.2088 | 0.2969 | 57.19% |
| Graph NLL (+ training-time fix) | 0.1292 | 0.1846 | 0.0277 | 86.31% |
| Graph structural fix (+ mechanism fix) | 0.1389 | 0.2370 | 0.0424 | 91.61% |

**Step 1 — training-time NLL fix (SCRUM-70, "fix 1").** Applying the same
Gaussian-NLL training idea used for BLIP (§6) to Phase 3: force
calibration improves dramatically (ECE 0.2969 → 0.0277, a >10x
improvement; coverage@90% 57.19% → 86.31%, roughly closing the gap to
nominal) and force MAE actually *improves slightly* alongside it — unlike
BLIP, where NLL training made calibration worse. But energy MAE regresses
sharply (+44%, later found to be +72.3% once protocol-matched against
BLIP NLL for cross-model comparison, §1 cross-ref `reports/phase2_vs_phase3_nll.md`)
and energy's already-poor overcoverage (99.46% → 99.99% at nominal 90%)
gets worse, not better.

**Why energy and force move in opposite directions under the same
training change — the mechanism, confirmed by two independent lines of
evidence:**

1. **The variance-floor mechanism from §6 recurs here, with a telling
   asymmetry.** Phase 3's NLL energy variance floor (`nll_min_energy_var
   = 27.47`) is ~2.2x *higher* than BLIP's equivalent floor (`12.56`) at
   the same epoch-20 boundary, while the *force* floors are close between
   the two models (6.81 vs 6.33, 1.08x) — it is specifically the energy
   floor that differs this much, not a general "Phase 3 trains worse"
   effect. A diagnostic run doubling the warmup period
   (`nll_warmup_epochs=40`, giving the pre-NLL MSE phase twice as long to
   converge before the floor is fixed) recovered about two-thirds of the
   energy regression (+72.3% → +26.4%) but gave back roughly a third of
   the force-calibration win (ECE improvement over BLIP fell from 26.6x to
   9.2x) — confirming the floor-convergence hypothesis directly: warmup
   length is a real dial trading energy accuracy against force
   calibration sharpness on this architecture, not a bug with one correct
   setting (run: `phase3_graph_stochastic_nll_warmup40`, trained from a
   dirty tree so not citable as an official number, but sufficient to
   confirm the mechanism directionally).
2. **The noise *pathway* to forces was structurally weak in the original
   mechanism, independent of training recipe.** The original
   `RepresentationNoise` added its stochastic term *additively* to the
   post-aggregation node state (`h_new + scale(h_new)·sigma·eps`), where
   `eps` is a fresh draw with zero functional dependence on atomic
   position (`d(eps)/dR = 0`). Energy reads directly off the noisy
   representation, so the noise reaches energy uncertainty at full
   strength; forces are computed as `F = -∂E/∂R` via autograd, and
   differentiating a term with zero position-gradient kills its *direct*
   contribution to the force Jacobian entirely — only a weaker,
   second-order pathway through the spatial derivative of the gating term
   survives. This explains the original (pre-NLL) numbers directly: force
   coverage@90% 57.19% vs. energy's 99.46%, and explains *why* NLL
   training can only partially fix it — `scale(h)·sigma` is one shared
   knob feeding both the full-strength energy pathway and the structurally
   weak force pathway, so turning it up to compensate force necessarily
   overshoots energy. Full argument:
   `docs/reviews/scrum-70-structural-root-cause-2026-11-12.md`.

**Step 2 — structural fix (SCRUM-70, "fix 2", 17-18 Nov 2026).** Addressing
the pathway asymmetry directly: noise now perturbs each *message*
multiplicatively, before aggregation (`m_out = m·(1 + scale(m)·sigma·eps)`),
where the message is itself a function of the live inter-atomic distance —
so `d(m)/dR` is generically nonzero and the noise now multiplies a real,
position-dependent Jacobian rather than a position-independent constant,
mirroring how BLIP's multiplicative weight noise reaches its force channel
(§2.1). Verified correct by the project's finite-difference test (forces
remain exactly `-∂E/∂R` under the new mechanism) before being trained;
this specific mechanism was independently prototyped by Ruturaj in his own
exploratory notebook before being ported to the production model, which is
a point of convergent validation for the idea, separate from whether the
quantitative outcome below is a net improvement.

**Result: the fix does what it was mechanically designed to do, and that
alone is not sufficient for a calibration win.** Force coverage@90% clears
the 90% nominal target for the first time (91.61%) — confirming forces now
have a genuine first-order noise pathway — but force ECE, which integrates
miscalibration across *all* coverage levels rather than one point, gets
*worse* (0.0277 → 0.0424): the model has moved from under-covering to
over-covering at most levels (e.g. 50% nominal → 60.0% actual), and both
energy and force MAE regress further on top of the NLL-only run's own
accuracy cost. **The lesson generalises beyond this one fix**: coverage at
a single nominal level and integrated calibration error are different
properties that do not necessarily move together, and a fix that
correctly identifies and repairs a *mechanism* (the missing first-order
derivative pathway) does not automatically produce better *calibration* —
the model still has to land its predicted variance at roughly the right
*magnitude*, not just restore the right *order* of dependence on position.
This mechanism is not merged to `main` and awaits Ruturaj's review per the
project's model-ownership convention (CLAUDE.md); it is reported here, and
tracked in the project's Jira (SCRUM-70) and reports, as an additional,
clearly-labelled experiment rather than a replacement for the NLL-only
numbers above.

---

## 8. Project findings — calibration methodology and reproducibility

**Post-hoc correction (§3) narrows the gap to nominal for every variant**,
as it's designed to — scalar and CQR correction both bring BLIP NLL, Graph
MSE and Graph NLL to within ~2 points of 90% nominal coverage on a held-out
5,000-frame evaluation split. But **post-hoc correction is not a
substitute for fixing raw calibration** (the point of §7's SCRUM-70 work):
Graph NLL's scalar-corrected force ECE (0.0415) is actually slightly worse
than its own *raw* force ECE (0.0277, on a different subset) — correcting
coverage toward nominal and lowering integrated error across all levels
are not the same improvement, the same lesson §7 draws from the structural
fix. The two post-hoc methods also disagree with each other: Phase 3
energy interval width diverges 13-33% between the scalar and CQR methods
depending on configuration, unresolved as of this document — per §3, this
is evidence that at least one method's assumptions (Gaussianity for
scalar; exchangeability for CQR, which this project's trajectory-correlated
data does not strictly satisfy) is being meaningfully violated, not just
reporting noise.

**Independent re-score (SCRUM-71, 14 Nov):** every checkpoint with a local
`.pt` file reproduces its recorded accuracy/benchmark metrics to within
~1e-9 relative or better when re-scored independently — confirming the
*recorded* numbers are reproducible from the *same trained checkpoint*.
This is a narrower claim than reproducibility of *training*: no variant in
this project has a repeated-seed training replication, so none of the
cross-model deltas reported in §5-7 carry a formal statistical-significance
claim — they are single-run descriptive comparisons, stated as such
throughout, not evidence of a reliably reproducible ranking.

---

## 9. Synthesis — what the literature predicted, and what had to be found empirically

Three things in §4-8's findings were essentially *predictable* from the
literature in §1-3 before any training ran: that DimeNet++/PaiNN's extra
geometric expressiveness (angles, equivariant vectors) would help on
MD17's near-equilibrium regime (§1.3-1.4, confirmed §5); that a cutoff
radius default needs checking against the actual system size rather than
assumed (standard GNN practice, confirmed concretely for MD17 vs. the
MD22 risk, §4); and that neither post-hoc calibration method's assumptions
(§3) are rigorously satisfied by trajectory-correlated MD data, which
predicts the two methods *should* disagree somewhat rather than being a
surprise when they did (§8).

Two things were not predictable from reading the papers alone and had to
be found empirically: that moving stochasticity from weights to
representations would produce the *specific, asymmetric* energy/force
trade-off documented in §7 (no paper in §2.2's design space specifies the
exact mechanism, so its first-order behaviour had to be derived and then
confirmed against the actual training runs); and that a *training-time*
mechanism (the NLL variance floor, §6-7) and a *forward-pass* mechanism
(the noise pathway, §7) would turn out to be two separate, independently-
contributing causes of the same symptom (force under-coverage) on the same
model — disentangling them required the two-step fix-and-measure process
in §7, not a single literature-informed design decision.

**Open going into the report:** Ruturaj's review of the SCRUM-70
structural fix; a repeated-seed training replication for at least the
headline Phase 2/3 variants (nothing in §5-8 currently supports a
significance claim without one); resolving the scalar-vs-CQR energy-width
disagreement; and extending §4's cutoff-radius check to MD22 before any
future Phase 1-3 work touches it.

---

## References

[1] Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., and Dahl,
G. E. (2017). *Neural Message Passing for Quantum Chemistry*. ICML, PMLR
70, 1263-1272. [Proceedings and paper](https://proceedings.mlr.press/v70/gilmer17a.html).

[2] Schütt, K. T., Kindermans, P.-J., Sauceda, H. E., Chmiela, S.,
Tkatchenko, A., and Müller, K.-R. (2017). *SchNet: A continuous-filter
convolutional neural network for modeling quantum interactions*. Advances
in Neural Information Processing Systems 30.
[Proceedings and paper](https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html).

[3] Coscia, D., de Haan, P., and Welling, M. (2025). *BLIPs: Bayesian
Learned Interatomic Potentials*. arXiv:2508.14022v1.
[Version used for this review](https://arxiv.org/html/2508.14022v1).
[Author implementation](https://github.com/dario-coscia/blip).

[4] Schütt, K. T., Unke, O. T., and Gastegger, M. (2021). *Equivariant
message passing for the prediction of tensorial properties and molecular
spectra*. Proceedings of the 38th ICML, PMLR 139.

[5] Gasteiger, J., Giri, S., Margraf, J. T., and Günnemann, S. (2020).
*Fast and Uncertainty-Aware Directional Message Passing for
Non-Equilibrium Molecules*. NeurIPS 2020 Machine Learning for Molecules
Workshop. Building on Gasteiger, J., Groß, J., and Günnemann, S. (2020),
*Directional Message Passing for Molecular Graphs*, ICLR 2020.

[6] Unke, O. T., and Meuwly, M. (2019). *PhysNet: A Neural Network for
Predicting Energies, Forces, Dipole Moments, and Partial Charges*. Journal
of Chemical Theory and Computation, 15(6), 3678-3693.

[7] Rong, Y., Huang, W., Xu, T., and Huang, J. (2020). *DropEdge: Towards
Deep Graph Convolutional Networks on Node Classification*. ICLR 2020.

[8] Feng, W., Zhang, J., Dong, Y., Han, Y., Luan, H., Xu, Q., Yang, Q.,
Kharlamov, E., and Tang, J. (2020). *Graph Random Neural Networks for
Semi-Supervised Learning on Graphs*. NeurIPS 2020.

[9] Kuleshov, V., Fenner, N., and Ermon, S. (2018). *Accurate Uncertainties
for Deep Learning Using Calibrated Regression*. ICML, PMLR 80.

[10] Romano, Y., Patterson, E., and Candès, E. (2019). *Conformalized
Quantile Regression*. NeurIPS 32.

[11] Bao, et al. (2025). Working reference for the project's CQR
implementation — see `docs/reviews/cqr-calibration-report-2026-11-05.pdf`
for the exact citation and the method as applied here.

[12] Chmiela, S., Tkatchenko, A., Sauceda, H. E., Poltavsky, I., Schütt,
K. T., and Müller, K.-R. (2017). *Machine learning of accurate
energy-conserving molecular force fields*. Science Advances, 3(5).
[Paper](https://www.science.org/doi/10.1126/sciadv.1603015). Origin of
the MD17 benchmark dataset this entire project trains and evaluates on.

Related project records: [SCRUM-33 lineage and earlier DimeNet++ evidence](dongxiao_sprint4/literature_review.md) ·
[Technical report](technical_report.md) · [Data dictionary](data_dictionary.md) ·
[EDA write-up](eda.md) · [SCRUM-70 structural root-cause analysis](reviews/scrum-70-structural-root-cause-2026-11-12.md) ·
[SCRUM-36 methodological review](dongxiao_scrum36/SCRUM36_Report.md).
