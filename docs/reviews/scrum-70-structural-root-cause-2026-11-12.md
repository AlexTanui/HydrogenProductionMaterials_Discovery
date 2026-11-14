# SCRUM-70: is the force-uncertainty bug structural? Yes — and it's the same mechanism driving the warmup-diagnostic energy/force trade-off

Written for Ruturaj (SCRUM-70 assignee) and the team. Builds directly on comment 10116's root-cause note (`d(eps)/dR = 0`) and comment 10176/10177's warmup diagnostic — this doc ties the two together into one structural explanation and proposes fixes. No change made to `ml/models/graph_stochastic.py`; per CLAUDE.md, that file's maths is Ruturaj's.

## Short answer

**Yes, it's structural.** `RepresentationNoise` injects noise *additively* into a representation after it has already been computed, with no multiplicative coupling to anything position-dependent. `StochasticLinear` (BLIP, Phase 2) injects noise *multiplicatively* onto the activation it's transforming — the same activation whose spatial derivative drives the force. That one difference in *how* noise enters the computation graph, not a training-recipe gap, is why force uncertainty needed training-time intervention at all, and why fixing it via NLL training drags energy accuracy down with it.

## The argument

**Phase 3 (`RepresentationNoise.forward`, `ml/models/graph_stochastic.py:139-143`):**
```python
scale = F.softplus(self.gate(h))
sigma = F.softplus(self.log_sigma)
eps = torch.randn_like(h)
return h + scale * sigma * eps
```
Differentiate the noise term w.r.t. `pos` (writing `s = scale(h)*sigma` for brevity, `eps` fixed per forward call):
```
d(s * eps)/dR = eps * ds/dR
```
`ds/dR` is the derivative of a scalar `softplus(Linear(h))` through `h`'s own (already weak, nonlinear) dependence on `pos` — a second-order, indirect pathway. The *direct* term `s * d(eps)/dR` is exactly zero, because `eps` is a constant draw for that forward call, not a function of `pos`.

**Phase 2 (`StochasticLinear.forward`, `ml/models/blip.py:127-132`):**
```python
weight_sigma = F.softplus(self.weight_rho)
eps = torch.randn_like(self.weight_mu)
row_scale = F.softplus(self.noise_gate(x))
return F.linear(x, self.weight_mu, self.bias) + row_scale * F.linear(x, weight_sigma * eps)
```
The noise term is `row_scale * (x @ (weight_sigma*eps).T)` — **linear in `x`**, the activation itself. Differentiate w.r.t. `pos`:
```
d(x @ W_noise.T)/dR = (dx/dR) @ W_noise.T    [+ a row_scale gating term, same order as Phase 3's]
```
`dx/dR` is *the same quantity* that drives the deterministic forward path's contribution to force — first-order, full-strength, not a second-order correction. BLIP's force channel inherits a stochastic contribution of the same differential order as its mean signal. Phase 3's does not; it only gets the weak gating term, because the direct term vanishes identically.

**This is a property of where/how the noise is injected, not of training.** No choice of loss, warmup length, or MC sample count changes which terms survive differentiation — that's fixed by the forward-pass algebra. It explains:
- The original SCRUM-70 numbers (comment 10116): force coverage@90% 57.19% vs energy's 99.52%, before any training change.
- Why NLL training (comments 10162, 10176, 10177) can *partially* compensate: pushing `scale(h)*sigma` up via gradient descent inflates the weak `ds/dR` pathway until its apparent force variance matches real force residuals — but `scale(h)*sigma` is one shared quantity feeding **both** the full-strength energy pathway and the weak force pathway. Inflating it to fix force's weak signal necessarily overshoots energy's already-strong signal. **This is the same mechanism as the warmup-diagnostic trade-off** (longer warmup → lower variance floor → less inflation needed → less energy overshoot, but also less force-channel compensation) — one shared coupling constant, two different gains, no way to set it independently per output. The warmup knob changes how hard training pushes on that shared constant; it can't decouple the two channels.

## Why the existing mitigation (NLL training, already shipped) is a real improvement but not a structural fix

It is not "wrong" — it measurably narrows the force gap (ECE 0.298→0.028, 10x+) by giving the weak pathway an actual gradient signal it never had under plain MSE (comment 10116's framing is correct: SCRUM-68's post-hoc scale genuinely couldn't do this). But it's fighting the structural gain asymmetry with optimization pressure on a shared parameter, which is exactly why it can't get force calibration for free — every unit of fix costs energy accuracy, at a rate set by the gain ratio between the two pathways, not by anything tunable in `train.py`.

## Proposed fixes, ranked by how much they touch the mechanism

All three are maths changes to `graph_stochastic.py` and are proposed to Ruturaj, not implemented here, per CLAUDE.md's ownership rule.

1. **Smallest change, worth trying first: shrink `log_sigma`'s init** (currently `softplus(-3) ≈ 0.049`, ~7x BLIP's `softplus(-5) ≈ 0.0067`) to match BLIP's scale. This doesn't fix the structural gain asymmetry — energy still gets full-strength noise, force still gets the weak derivative-mediated term — but it reduces how much early-training noise the energy pathway has to fight before `scale`/`log_sigma` have been shaped by gradient, independent of warmup length. Cheap to test (one constant), low risk, but likely only dampens the trade-off rather than resolving it, since the underlying additive-vs-multiplicative gain gap is untouched.

2. **Moderate change, addresses the gain asymmetry directly: make the injected noise multiplicative with a position-dependent quantity**, mirroring how BLIP's noise multiplies `x`. Concretely: inject the stochastic term into the *message* computation (`message_mlp`'s output, which is a function of `edge_rbf(dist)` and hence of `pos`) multiplicatively — e.g. `message_noisy = message * (1 + scale(message)*sigma*eps)` — rather than additively into the already-aggregated node state after message passing. This gives force a first-order noise pathway analogous to BLIP's, without abandoning "noise lives in representations/messages, not weights" as the Phase 3 mechanism.

3. **Most invasive, cleanest decoupling: a separate, explicit force-uncertainty head**, e.g. a per-atom or per-edge learned scale trained directly against force residuals via its own NLL term, independent of `RepresentationNoise`'s `scale(h)*sigma`. This fully breaks the shared-parameter coupling — force calibration could be tuned without moving energy's noise magnitude at all — but it's a bigger departure from "one noise mechanism, read two ways" and probably needs sign-off from Dongxiao too, since `graph_stochastic.py`'s own module docstring already flags the exact mechanism as "not yet a team decision" pending the SCRUM-33/34 literature review. This is squarely what that review should be weighing in on.

## Caveat

This is an architectural/algebraic argument about which gradient terms survive differentiation, not a new experiment — it explains the existing numbers (comments 10116, 10162, 10176, 10177) rather than producing new ones. Options 2 and 3 above are untested; before adopting either, the same before/after protocol used for the NLL fix (`ml/training/calibrate.py` against a retrained checkpoint, clean commit) should be run to confirm the predicted effect actually shows up in force ECE/coverage without a proportional energy cost.
