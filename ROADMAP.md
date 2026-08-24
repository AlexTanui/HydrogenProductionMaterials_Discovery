# Roadmap — PG-S2-47, 10-Week Plan

Companion to [glossary.md](glossary.md) (team structure, architecture,
contracts). That document says *what* goes where; this one says *when* and
*in what order*, and why the plan looks the way it does given real
constraints.

**Ambition level:** full — foundation-model fine-tuning, rigorous
leakage-aware evaluation, causal discovery cross-validated against
explainability, an active-learning platform, and surrogate-guided
candidate proposal. Nothing here is descoped by default; the fallback
order in §5 exists for if — not when — time runs short.

**Constraints this plan is built around:**
- **Timeline:** 10 weeks, hard deadline.
- **Compute:** personal GPUs / Colab only — no university cluster or cloud
  credits. This rules out training large models from scratch and drives
  the two methodological calls below.

**Two adjustments made to keep "very ambitious" realistic on this compute:**
1. **Phase 1 uses adapter/partial fine-tuning**, not full fine-tuning or
   training from scratch. Load a pretrained checkpoint (CHGNet or
   MACE-MP-0-small), freeze most of it, fine-tune only the output head /
   last few layers. Fits Colab session limits, and is arguably the more
   defensible methodological choice on a small downstream dataset anyway
   (lower overfitting risk than full fine-tuning).
2. **Phase 4 uses surrogate-guided search** (genetic algorithm or Bayesian
   optimization over the Phase 1 model + its uncertainty as the fitness
   function), not a trained deep generative model (CDVAE/DiffCSP-style).
   Same goal — propose novel candidates, not just rank an existing pool —
   at a compute cost that fits personal hardware.

---

## 1. Phase overview

| Phase | Goal | Depends on |
|---|---|---|
| 0 — Foundations | Multi-source data fusion, canonical schema, leakage-aware splits | — |
| 1 — Modeling | Fine-tuned foundation model + calibrated uncertainty | Phase 0 |
| 2 — Causal validation | Causal discovery over structure→property, cross-checked against GNN explainability | Phase 1 (needs model embeddings/predictions) |
| 3 — Platform | Active-learning ranking wired into backend/frontend | Phase 1 (needs real predictions + uncertainty) |
| 4 — Discovery loop | Surrogate-guided search proposing new candidates | Phase 1 (needs trusted uncertainty) |

Phases 1–4 run as **parallel tracks**, not a strict relay — each person
starts in week 1 against the contracts in `glossary.md`, using stub/mock
data until the phase they depend on lands for real. Integration checkpoints
(§3) are where tracks actually connect.

---

## 2. Week-by-week plan

| Week | Shijin — data | Ruturaj — model | Dongxiao — causal | Fazin — QA/eval | Alex — platform |
|---|---|---|---|---|---|
| 1–2 | Fuse Catalysis-Hub / Materials Project / Open Catalyst into canonical schema; leakage-aware (family-based) splits | Select + load pretrained checkpoint (CHGNet / MACE-MP-0-small); baseline eval | Literature review on causal priors (d-band center, Sabatier principle) | Define eval protocol + metrics (§4 DoD) | Scaffold backend/frontend against real schema, not stubs |
| 3–4 | Data quality/bias audit; δ-correction for DFT-functional inconsistency across sources | Adapter fine-tuning (frozen backbone, tuned head); add ensemble/conformal uncertainty | Causal discovery (NOTEARS or PC algorithm) on Phase-1 embeddings | Build benchmark harness; regression tests against Phase 0 splits | Wire `POST /predictions` to the real model (replace stub) |
| 5–6 | Support feature extraction as needed by Dongxiao/Ruturaj | Finalize model; calibrate uncertainty | GNNExplainer / integrated-gradients cross-validation vs. causal graph | Run full benchmark suite; catch leakage/overfit before it's load-bearing elsewhere | Build active-learning ranking endpoint (predicted activity + uncertainty) |
| 7–8 | — | Support the search fitness function (Phase 4) | Write up causal findings for the technical report | Validate uncertainty calibration — this gates whether Phase 4 can be trusted | Implement GA/BO surrogate-guided search; live candidate queue UI |
| 9 | **Integration week — everyone.** Wire all tracks together; end-to-end test the full pipeline (data → model → causal validation → ranked/searched candidates → UI) | | | | |
| 10 | **Buffer + report/demo polish.** Assume integration week surfaces at least one real bug — this week absorbs it, not the deadline | | | | |

---

## 3. Integration checkpoints

Don't wait for week 9 to discover two tracks don't fit together. Treat
these as hard sync points:

- **End of week 2:** Phase 0 schema is frozen. Every other track builds
  against it from here — a schema change after this point is a
  cross-team blocker, raise it immediately rather than working around it
  locally.
- **End of week 4:** Phase 1 produces real predictions + uncertainty for
  the first time. Alex swaps the backend off the stub predictor here;
  Dongxiao starts causal discovery on real embeddings, not placeholders.
- **End of week 6:** Uncertainty calibration is validated by Fazin. This
  is the actual gate for starting Phase 4 — a surrogate-guided search
  built on miscalibrated uncertainty will confidently propose bad
  candidates, which is worse than not having Phase 4 at all.
- **Week 9:** full pipeline integration, as above.

---

## 4. Definition of done, per phase

- **Phase 0:** canonical schema documented in `docs/data_dictionary.md`;
  splits are by material family, not random; a written note on which
  confounders were identified and how (δ-correction, dataset flag, etc.).
- **Phase 1:** fine-tuned checkpoint beats a naive baseline (e.g. mean
  predictor) on the Phase 0 test split by a stated margin; uncertainty
  estimates exist and are calibrated (not just present).
- **Phase 2:** a causal graph exists over the structural features
  considered; a written comparison of causal-graph findings vs.
  GNNExplainer output, including where they *disagree* — disagreement is
  a finding, not a failure.
- **Phase 3:** `/predictions` and `/benchmarks` serve real (not stub)
  data; the frontend queue reflects live ranking, not a static table.
- **Phase 4:** the search proposes at least one candidate outside the
  original training pool, with a predicted-activity + uncertainty
  estimate attached, and a documented fitness function.

---

## 5. Fallback order (decide now, not in week 8)

If a track falls behind, descope in this order — decided now so it's a
plan, not a panic call under deadline pressure:

1. **Phase 4 first** — shrink the candidate search space, or cut the
   search down to a simpler heuristic (e.g. greedy ranking instead of
   full GA/BO).
2. **Phase 3's UI polish** — the ranking/search logic matters more for
   the report than a polished frontend; a working endpoint with a plain
   table beats a broken interactive queue.
3. **Phase 2 depth** — causal discovery is the most research-heavy piece;
   fall back to ATE estimation on the top few hand-picked features
   (the original, simpler scope) rather than full causal graph discovery.
4. **Phase 0–1 are non-negotiable.** Everything else depends on them
   being correct; there is no cheaper fallback version of "the data and
   base model are trustworthy."

---

## 6. Ownership recap

Full role descriptions and folder-level ownership: `glossary.md` §1 and §3.
This roadmap sequences that ownership in time; it doesn't change who owns
what.
