# Roadmap — PG-S2-47, 10-Week Plan

Companion to [glossary.md](glossary.md) (team structure, architecture,
contracts). That document says *what* goes where; this one says *when* and
*in what order*. This plan follows the supervisor-issued research proposal
directly — three phases, each building on the last, using MD17 as the
benchmark dataset (see `glossary.md`'s note on how that relates to the
project's title).

**Constraints:**

- **Timeline:** 10 weeks, hard deadline.
- **Compute:** personal GPUs / Colab only. Unlike the platform's earlier
  ambition (fine-tuning a large pretrained foundation model), this plan is
  comfortably within that budget — a from-scratch MPNN on a single MD17
  molecule at a time is a small model on small graphs (~10–20 atoms). No
  compute-driven scope adjustment is needed here.

Phases 1–3 are **sequential by nature** — Phase 2 extends the Phase 1 model,
Phase 3 is informed by findings from both — but data pipeline work,
benchmarking-harness design, literature review, and platform work can all
start in week 1 in parallel, against the contracts in `glossary.md`.

---

## 1. Phase overview

| Phase | Goal | Depends on |
|---|---|---|
| 1 — Basic MPNN | Deterministic message-passing baseline predicting energy + forces on MD17 | — |
| 2 — BLIP reproduction | Reproduce BLIP's weight-space Gaussian stochasticity for uncertainty | Phase 1 model |
| 3 — Graph-space stochasticity | The team's own contribution: stochasticity in node/edge representations instead of weights | Phase 1 + 2 findings, literature review |
| Platform | Backend/frontend serving predictions + the three-phase benchmark comparison | Each ml phase, as it lands |

Per the proposal itself, Phase 3 is expected to be the most open-ended —
its exact mechanism (node dropout, edge dropout, Gaussian perturbation,
learnable perturbation strength) is a team decision made from the
literature review and Phase 1/2 results, not fixed in advance.

---

## 2. Week-by-week plan

| Week | Shijin — data | Ruturaj — model | Dongxiao — research/eval | Fazin — QA/benchmarking | Alex — platform |
|---|---|---|---|---|---|
| 1–2 | Bronze → silver: dedupe (aspirin/malonaldehyde/toluene/ccsd_t-ethanol format duplicates, azobenzene/paracetamol/AT-AT-CG-CG npz-vs-zip redundancy), unify format, tag every file with molecule + theory level (see `glossary.md` §3 inventory) | Phase 1 Gilmer-style MPNN backbone (RBF-expanded distance edges) — shared by Phases 2 and 3, not rebuilt per phase | Literature review: Gilmer 2017, Schütt 2017 (SchNet) — Phase 1 grounding; start BLIP paper | Define eval protocol; energy/force MAE metric implementation | Scaffold backend/frontend against the MD17-based contracts in `glossary.md` §4–5 |
| 3–4 | Silver → gold: apply trajectory-block splits (preserving the literature-standard splits that ship with aspirin/malonaldehyde/toluene/ccsd_t-ethanol, per §5); validate no train/test leakage | Train Phase 1 baseline; validate autograd forces (finite-difference check against `F = -∂E/∂R`) | Finish BLIP paper deep-dive; document the core mechanism to reproduce | Regression-test Phase 1 reproducibility; benchmark harness skeleton | Wire `POST /predictions` to the real Phase 1 checkpoint (replace stub) |
| 5–6 | Extend gold pipeline to the larger MD22 molecules if scope allows (stretch — 370-atom nanotube needs the cutoff-radius question from `glossary.md` §5 resolved first) | Implement Phase 2 (BLIP-style stochastic weights); multiple MC forward passes for uncertainty | Implement/validate UQ metrics (ECE, uncertainty–error Spearman correlation); calibration analysis | Extend benchmark harness with UQ metrics; run Phase 1 vs Phase 2 comparison | Wire uncertainty into `/predictions` + `/benchmarks`; update Predict/Benchmarks pages |
| 7–8 | — | Implement Phase 3 (graph-space stochasticity), per the team's chosen approach | Evaluate Phase 3 vs Phase 1/2; statistical analysis; start writing up findings | Run full three-phase benchmark suite; verify a fair comparison protocol (same splits, same MC sample counts) | Finalize dashboard's three-phase comparison view |
| 9 | **Integration week — everyone.** Wire all three phases into `/benchmarks`; end-to-end test data → model → metrics → dashboard | | | | |
| 10 | **Buffer + report/demo polish.** Assume integration week surfaces at least one real bug — this week absorbs it, not the deadline | | | | |

---

## 3. Integration checkpoints

- **End of week 2:** MD17 data pipeline + splitting strategy frozen. Every
  other track builds against it from here.
- **End of week 4:** Phase 1 produces real energy/force predictions.
  Alex swaps the backend off the stub predictor here.
- **End of week 6:** Phase 2 uncertainty is calibrated and validated by
  Fazin/Dongxiao. This is the credibility gate before Phase 3 comparisons
  mean anything.
- **Week 9:** full three-phase integration, as above.

---

## 4. Definition of done, per phase

- **Phase 1:** energy MAE and force MAE reported on the held-out
  (trajectory-block) test split; forces verified against a finite-difference
  check of the energy gradient, not just "the model runs."
- **Phase 2:** BLIP's core mechanism (input-dependent Gaussian stochasticity
  in message/update weights) is reproduced; ECE and uncertainty–error
  correlation are computed and compared against Phase 1's (lack of) UQ.
- **Phase 3:** at least one graph-space stochasticity variant is
  implemented and evaluated on the same metrics and splits as Phase 1/2,
  with a written comparison — including where it *doesn't* beat BLIP,
  since that's a legitimate finding for RQ2/RQ3, not a failure.
- **Platform:** `/predictions` and `/benchmarks` serve real (not stub)
  data for whichever phases have landed; the dashboard's three-phase
  comparison reflects live benchmark results.

---

## 5. Fallback order (decide now, not in week 8)

1. **Phase 3 scope first** — the proposal itself flags this as the most
   open-ended phase; fall back to the simplest variant (e.g. plain node
   feature dropout) rather than exploring multiple perturbation strategies.
2. **Platform polish** — a working `/benchmarks` endpoint with a plain
   table beats a broken interactive comparison view.
3. **Phase 2 depth** — fall back to fewer MC samples / a simpler
   calibration analysis rather than the full ECE + Spearman treatment.
4. **Phase 1 is non-negotiable.** Everything else depends on a working,
   validated baseline.

---

## 6. Ownership recap

Full role descriptions and folder-level ownership: `glossary.md` §1 and §3.
This roadmap sequences that ownership in time; it doesn't change who owns
what.
