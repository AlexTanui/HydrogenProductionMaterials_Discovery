# Model card — GraphStochastic (Phase 3, first draft)

**Ticket:** SCRUM-32 · **Owner:** Alex (kickoff; mechanism pending team review) · **Status:** Trained, evaluated, committed (`alex`/`main` @ `fdc2353`) · **Date:** 2026-10-22

## Summary

Phase 1's MPNN backbone with input-dependent Gaussian noise added to node
representations after each message-passing layer, in place of BLIP's
stochastic weights — the "stochasticity in node/edge representations"
CLAUDE.md specifies for Phase 3. Implementation: `ml/models/graph_stochastic.py`.

**This is explicitly a first draft, not the team's final mechanism
decision.** CLAUDE.md says Phase 3's exact mechanism should come from the
literature review (SCRUM-33/34), which hadn't landed when this was
trained. `RepresentationNoise` is isolated so it is a one-piece swap once
that review lands; nothing else here or in the shared harness needs to
change for that. Everything below describes this one draft, not a
validated choice.

## Result

Trained by the shared `ml/training/train.py`, scored once on the full test
split by `ml/training/evaluate.py`, and compared against `phase2_blip_nll`
under the harness's fairness check (same data, same train/val/test frames,
same eval MC sample count): [`reports/phase2_vs_phase3.md`](../../reports/phase2_vs_phase3.md).
Run record: [`experiments/runs/phase3_graph_stochastic/`](../../experiments/runs/phase3_graph_stochastic/).

| | |
|---|---|
| Config | [`experiments/configs/phase3_graph_stochastic.yaml`](../../experiments/configs/phase3_graph_stochastic.yaml) — hidden_channels 128, num_layers 4, num_rbf 50, kl_weight 0.01 |
| Parameters | 385,925 |
| Trained at | `fdc2353`, clean tree |
| Data | ethanol DFT: 10,000 train / 5,000 val frames (stride subsets, seed 42); test: all 55,509 frames, seed 0, batch 32 |
| Epochs | 200 run, best epoch 197; stopped by the 200-epoch cap |
| Training time | 2,641.7 s (0.73 h), cpu — about half `phase2_blip_nll`'s per-epoch cost, consistent with half the parameter count and no multi-sample (K=1) training |
| Training loss | plain MSE throughout (no NLL hooks wired up yet — see *Known limitations*) |

| Test metric | Value |
|---|---|
| **Energy MAE** (total) | **0.0897** kcal/mol |
| Energy RMSE (total) | 0.1126 kcal/mol |
| **Force MAE** (component) | **0.2088** kcal/mol/Å |
| Force RMSE (component) | 0.3264 kcal/mol/Å |
| Inference | 6.554 ms/frame (20 MC passes, batch 32) |

## Uncertainty — a real finding, not a clean win

| | `phase2_blip` (K=1, MSE) | `phase2_blip_nll` (K=4, NLL) | `phase3_graph_stochastic` (K=1, MSE) |
|---|---|---|---|
| Energy ECE | 0.703 | 0.772 | **0.154** |
| Energy Spearman | 0.0047 | 0.0026 | 0.0021 |
| Energy coverage @ 90% nominal | 0.0997 | 0.0187 | **0.9946** |
| Energy sharpness (predicted std) | 0.0276 | 0.0127 | 0.2143 |
| Force ECE | 0.676 | 0.741 | **0.297** |
| Force Spearman | 0.273 | 0.300 | 0.262 |
| Force coverage @ 90% nominal | 0.1322 | 0.0551 | 0.5737 |
| Force sharpness (predicted std) | 0.0231 | 0.0090 | 0.1137 |

ECE looks dramatically better than either BLIP variant — **but the full
picture is not "Phase 3 is calibrated."** Both BLIP runs are badly
*overconfident*: their 90%-nominal intervals actually cover only 2–13% of
the true values, because their predicted variance (sharpness ≈ 0.01–0.03)
is far too small. Phase 3's energy uncertainty goes the other way —
90%-nominal coverage of 99.5% means its intervals are too *wide*
(sharpness ≈ 0.21, roughly 8–17× BLIP's), i.e. honestly uncertain to the
point of being uninformative, not accurately calibrated. Force uncertainty
sits in between: still under-covering (57% at 90%-nominal) but far less
severely than either BLIP run.

So: ECE punishes severe overconfidence harder than it punishes being
honestly over-cautious in this binning, which is most of why Phase 3's
number looks so much better — not clear evidence the representation-noise
mechanism is intrinsically better calibrated. Energy Spearman is weak
everywhere (0.002–0.005 across all three models) — none of them produce
energy uncertainty that tracks per-sample error. Force Spearman is modest
and similar across all three (0.26–0.30).

**Honest summary:** Phase 3's first draft is more accurate (energy MAE
0.090 vs. 0.158–0.172) and avoids BLIP's severe overconfidence, trading it
for mild-to-moderate underconfidence on force and real overdispersion on
energy. Neither phase has uncertainty that reliably tracks its own error on
energy. This is one run of a first-draft, untuned mechanism — not a
verdict on which phase's approach is better once both are tuned.

## Known limitations

- **Mechanism is provisional** (see Summary) — pending SCRUM-33/34.
- **`kl_weight=0.01` was copied from BLIP's starting value, not tuned** for
  this mechanism — same caveat as `docs/phase2_remaining_work.md` item 4,
  now applying to Phase 3 too.
- **NLL training hooks (`train_num_mc_samples`, `nll_warmup_epochs`) are
  not wired up** — this run is plain single-sample MSE, deliberately, to
  avoid debugging the new mechanism and NLL training at once (see the
  model's module docstring). Nothing here says NLL training would or
  wouldn't help this mechanism; it hasn't been tried.
- **One seed, one run, one molecule (ethanol/DFT).** No variance estimate
  across seeds, no other MD17 molecules, no MD22.
- **Not yet in the extended physics test suite** — `tests/ml/test_graph_stochastic.py`
  checks forces = -dE/dR by finite difference and basic registry/KL
  sanity, but not `tests/ml/test_painn_physics.py`'s full six-model
  negative-control matrix (rotation/translation/permutation mutants).

## Reproducibility

- Checkpoint: `experiments/checkpoints/phase3_graph_stochastic.pt` (gitignored — regenerate via `python -m ml.training.train experiments/configs/phase3_graph_stochastic.yaml`).
- Results: `experiments/results/phase3_graph_stochastic.json` (gitignored).
- Run record (committed): `experiments/runs/phase3_graph_stochastic/`.
- Seed 42, deterministic split.
