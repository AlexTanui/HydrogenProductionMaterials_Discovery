# Hydrogen Production Materials Discovery Using Deep Neural Networks

**PG-S2-47**, Adelaide University, College of Engineering & IT
Agency supervisor: Henry Li (<henry.li@adelaide.edu.au>) · Academic supervisor: Dhika Pratama (<dhika.pratama@adelaide.edu.au>)

**Live dashboard:** <https://hydrogen-materials-discovery.vercel.app>, a read-only build. Training and data upload only work when you run it locally.

> **In one sentence:** we built graph neural networks that predict a molecule's energy and the forces on its atoms, and asked where randomness should be added to such a network so it can also say *how much to trust* each prediction.

---

## Contents

1. [Why this project exists](#1-why-this-project-exists)
2. [The research question and the three phases](#2-the-research-question-and-the-three-phases)
3. [What we found](#3-what-we-found)
4. [How we made the results trustworthy](#4-how-we-made-the-results-trustworthy)
5. [Team and contributions](#5-team-and-contributions)
6. [The dashboard](#6-the-dashboard)
7. [Repository layout](#7-repository-layout)
8. [Getting started](#8-getting-started)
9. [Limitations and open work](#9-limitations-and-open-work)
10. [Further reading](#10-further-reading)

---

## 1. Why this project exists

Discovering better materials for hydrogen production, such as catalysts, needs fast, accurate simulation of how atoms interact. Quantum-chemistry methods such as DFT are accurate but far too slow to screen many candidates. **Machine-learning interatomic potentials (MLIPs)** learn to reproduce those energies and forces at a fraction of the cost.

An MLIP is only useful for discovery if it **knows when it doesn't know**. A model that gives a confident wrong answer on an unfamiliar structure is worse than useless in a screening pipeline. So we need *uncertainty quantification (UQ)*: alongside each prediction, an honest estimate of its error.

Following the supervisor-issued technical plan (*Adaptive Graph-Space Uncertainty Modelling for Machine Learning Interatomic Potentials*), this project develops and validates that methodology on **MD17**, a standard molecular-dynamics benchmark. It does so before the methodology is ever pointed at real catalyst screening. The deliverable is a **validated method**, proven on data whose behaviour is well understood. [glossary.md](glossary.md) explains how this relates to the project title.

## 2. The research question and the three phases

**Where should stochasticity live in a molecular graph neural network: in the model's *weights*, or in its *graph representations*?**

The work is three sequential phases. Each extends the same backbone rather than introducing a new architecture, so any difference in results comes from *where the randomness is*, not from comparing unrelated models.

| Phase | What it is | Code |
|---|---|---|
| **1. Deterministic baseline** | A Gilmer-style message-passing neural network (MPNN). It predicts total energy, and forces follow exactly as `F = -∂E/∂R` (autograd), so the potential conserves energy. Four other published architectures were also trained under the same protocol as a bake-off. | [ml/models/mpnn.py](ml/models/mpnn.py) |
| **2. Weight-space uncertainty (BLIP)** | A reproduction of BLIP: input-dependent Gaussian noise on the message/update **weights**. Uncertainty comes from repeating stochastic forward passes (Monte Carlo sampling). | [ml/models/blip.py](ml/models/blip.py) |
| **3. Graph-space uncertainty (our contribution)** | The noise moves out of the weights and into the node/edge **representations**. | [ml/models/graph_stochastic.py](ml/models/graph_stochastic.py) |

All experiments share one dataset and protocol: **ethanol, DFT level of theory**. That means 555,092 MD frames split into contiguous trajectory blocks (444,074 train / 55,509 val / 55,509 test). Each model trains on a 10,000-frame subset for up to 200 epochs. Phase 2 and 3 uncertainty uses 20 MC samples. Energies are in kcal/mol; forces are in kcal/mol/Å per Cartesian component.

## 3. What we found

### Phase 1: architecture bake-off

| Model | Parameters | Energy MAE | Force MAE | Training time |
|---|---|---|---|---|
| DimeNet++ | 278,982 | **0.0537** | 0.1142 | 2.07 h |
| PaiNN | 588,033 | 0.0557 | **0.1117** | 1.35 h |
| PhysNet | 808,746 | 0.0611 | 0.1738 | 1.10 h |
| **MPNN (chosen backbone)** | 384,897 | 0.0779 | 0.1775 | 0.63 h |
| SchNet | 61,569 | 0.1714 | 0.4270 | 0.29 h |

**DimeNet++ and PaiNN were the most accurate, not MPNN.** We still chose MPNN as the Phase 2/3 backbone, deliberately. It gives a clean, well-understood place to inject weight noise (Phase 2) and representation noise (Phase 3) without redesigning the model twice. We traded accuracy headroom for a controlled scientific comparison, and we say so openly. Full table: [reports/phase1_benchmark.md](reports/phase1_benchmark.md).

### Phase 2 vs Phase 3: protocol-matched comparison

| | Phase 2: BLIP (weights) | Phase 3: graph-space (representations) |
|---|---|---|
| Energy MAE | **0.0750** | 0.1292 |
| Force MAE | **0.1787** | 0.1846 |
| Energy ECE (lower is better) | 0.647 | **0.201** |
| Force ECE (lower is better) | 0.738 | **0.028** |
| Force uncertainty–error Spearman | 0.284 | **0.295** |
| Parameters | 740,953 | **385,925** |
| Inference (ms/frame, 20 MC passes) | 12.9 | **6.3** |

Source: [reports/phase2_vs_phase3_nll.md](reports/phase2_vs_phase3_nll.md).

**The answer is real but not clean: it's a trade-off.**

- Moving uncertainty into the graph representations gives **far better-calibrated force uncertainty**: force ECE is about 26× lower. Force accuracy is nearly matched, and the model is half the size and twice as fast.
- It **costs energy accuracy**: energy MAE is about 72% higher than BLIP's.
- BLIP's uncertainty is badly **overconfident**: its "90% intervals" contain the true force only about 6% of the time. Phase 3's force intervals land close to the nominal level, while its energy intervals are too wide (underconfident).

We traced *why* the trade-off happens to the forward-pass algebra ([SCRUM-70 analysis](docs/reviews/scrum-70-structural-root-cause-2026-11-12.md)):

- Phase 3 adds noise *after* a representation is computed. Its derivative with respect to atomic positions therefore only reaches the forces through a weak, second-order path.
- BLIP's noise multiplies the activation itself, so it reaches the forces at full strength.
- One shared noise scale feeds both energy and forces in Phase 3. Training that pushes it up enough to fix force uncertainty overshoots on energy.

**This is a structural property, not a tuning problem.** A structural fix that moves the noise into the messages, where it couples multiplicatively, is implemented and scored. It is awaiting the Phase 3 owner's review and does not replace the result above until then (see [§9](#9-limitations-and-open-work)).

## 4. How we made the results trustworthy

A comparison table is only as good as the conditions behind it. We built the following checks in rather than assuming them:

- **One shared harness and protocol.** Every model trains through [ml/training/train.py](ml/training/train.py) with a config-driven, fixed budget. The benchmark ([ml/training/benchmark.py](ml/training/benchmark.py)) **refuses** to tabulate runs that differ in molecule, theory level, splits, training settings or MC sample count. It also refuses any run trained from a dirty git tree.
- **No data leakage.** Adjacent MD frames are near-duplicates, so splits are contiguous trajectory blocks, never random. Literature-standard splits are preserved where the source ships one. An RMSD-based [leakage audit](docs/reviews/leakage-audit-2026-10-31.md) checked this directly.
- **Fixed array names and units.** `z`, `R`, `E`, `F` in Å and kcal/mol run through every data layer, with validation that fails loudly ([docs/data_dictionary.md](docs/data_dictionary.md)).
- **Reproducible numbers.** Every checkpoint with a local `.pt` file was independently re-scored and reproduces its stored metrics to within **1e-9 relative**. A fixed-seed golden-run test guards against silent regressions.
- **A ratified UQ protocol (v1.0).** ECE, uncertainty–error Spearman correlation and coverage are defined once in [ml/utils/uq_metrics.py](ml/utils/uq_metrics.py). They are always reported **per quantity (energy and force separately), never pooled**: on the same checkpoint they differ by up to 100×.
- **Findings were tested, not assumed.** Examples:
  - Review caught three real bugs before they were reported: BLIP's batch-coupled noise, an NLL variance-floor collapse, and a 1.645× calibration unit slip.
  - Two independent calibration methods (scalar rescaling and conformalized quantile regression, CQR) were run against each other.
  - A hypothesis about atom-type error was tested and refuted, and the refutation was reported.

## 5. Team and contributions

This section lists each member's role ([glossary.md §1](glossary.md)), what they own in the codebase (agreed ownership in [CLAUDE.md](CLAUDE.md) and [glossary.md §3](glossary.md)), and their notable deliverables. Ownership means responsibility for a component's design and correctness. Commit authorship doesn't map one-to-one onto ownership, because some work was committed on a teammate's behalf ([docs/technical_report.md §9](docs/technical_report.md)).

### Alex Tanui: Scrum Master / Technical Lead / General Project Engineer

- **Owns:** backend API and dashboard (integration); the SchNet baseline; the Phase 2 BLIP model.
- Reframed the repository around the supervisor's MLIP plan, and set up the bronze → silver → gold data staging and the architecture/contract documents.
- Built the FastAPI backend and React dashboard: live predictions for every phase, the three-phase benchmark view, training jobs, data upload, the PDF "build report" endpoint, and Vercel hosting.
- Implemented BLIP's KL term, MC-sample evaluation and the UQ metric suite. Wired post-hoc conformal calibration and CQR into the app.
- Led the Phase 3 calibration investigation. That covered NLL training for force miscalibration (SCRUM-70), the structural root-cause analysis and proposed structural fix, and the protocol-fairness resolution between Phases 2 and 3 (SCRUM-36/40).
- Ran the independent re-score of every checkpoint (SCRUM-71) and the clean retrains under the fixed harness (SCRUM-24).

### Fazin Faizal: Developer / QA & Benchmarking Engineer

- **Owns:** the shared training/evaluation harness ([train.py](ml/training/train.py), [config.py](ml/config.py), model registry, energy scale-and-shift wrapper); metrics and **UQ protocol v1.0**; checkpoint scoring and the three-phase benchmark; the `ml/` and API test suites; the PaiNN model.
- Built the harness's fairness guards: identical-protocol checks, the clean-commit guard, and the molecule/theory consistency check.
- Ran the RMSD-based **leakage audit** of the gold splits (SCRUM-47) and found BLIP's batch-dependence bug.
- Built the golden-run regression guard (SCRUM-38), the MC sample-count stability study and compute profiling (SCRUM-49), and the physics and equivariance tests.

### Ruturaj Yashwant Bhosale: ML/AI Engineer, GNN Lead

- **Owns:** the Phase 1 MPNN backbone that Phases 2 and 3 extend, and the Phase 3 graph-space model's design (glossary: "Ruturaj implements, Dongxiao evaluates").
- Developed a graph-space stochastic **PaiNN** variant, with noise on nodes, edges or both, an adaptive posterior and showcase experiments. It's on the `ruturaj-phase3-graph-stochastic` branch.
- Reviewer for the proposed Phase 3 structural fix.

### Shijin Mathew: Data Engineer / ML Data Pipeline Lead

- **Owns:** the data pipeline ([ml/data/](ml/data/)) covering MD17/MD22 ingestion, validation, `MD17Sample`, contiguous splits and graph construction; data scripts; the PhysNet model.
- Produced the EDA figures behind the dashboard's EDA page and the PhysNet bake-off runs and scoring scripts (SCRUM-50).
- Added a 10-frame purge buffer between split blocks plus split provenance to the gold pipeline (SCRUM-72, on the `shijin-scrum-72` branch). Hardened data upload to reject bad files cleanly.

### Dongxiao Wu: ML Research & Model Evaluation Engineer

- **Owns:** the literature review, evaluation, the literature-informed Phase 3 design, and the DimeNet++ model.
- Built the DimeNet++ ethanol baseline with a full training analysis and frozen evaluation.
- Owned the original UQ metric definitions until 23 Oct 2026. Built a split-conformal calibration baseline and ran the edge-noise experiments that informed Phase 3.
- Wrote the SCRUM-33 literature review (MPNN, SchNet, BLIP mechanisms), the review of the Phase 2/3 comparison (SCRUM-36), and the project retrospective (on the `dongxiao` branch).

**Shared:** sprint planning and reviews, and the decision to move the Phase 3 mechanism into the representations. Each model file's maths stays with its owner; integration changes go in separate commits for that owner to review.

## 6. The dashboard

A React + Vite frontend talks to a FastAPI backend that loads the real trained checkpoints.

| Page | What it shows |
|---|---|
| **Dashboard** | Project overview, research plan and team |
| **EDA** | The exploratory analysis the project is built on: distributions, cutoff-radius check, split integrity |
| **Predict** | Pick a molecule, test frame and phase, and see predicted vs true energy, force magnitude and uncertainty |
| **Benchmarks** | All phases side by side (accuracy and UQ metrics), plus a downloadable PDF report |
| **Train** | Launch and monitor training jobs (local only) |
| **Ethics & Safety** | What our calibration findings mean for trusting a model's confidence before deployment |

The hosted build runs on Vercel. It serves a test-split-only copy of the gold data and refuses writes (`READ_ONLY=1`). To redeploy, run `python scripts/build_serving_gold.py && vercel deploy --prod`.

## 7. Repository layout

```text
ml/                 Python ML package (the research code)
  data/             MD17 loading, validation, bronze→silver→gold pipeline, PyG dataset
  models/           MPNN, BLIP, graph-stochastic, + SchNet/PaiNN/PhysNet/DimeNet++; registry
  training/         train / evaluate / benchmark / calibrate / MC-stability
  utils/            metrics, UQ metrics (protocol v1.0), leakage audit
experiments/
  configs/          one YAML per experiment; config-driven, nothing hardcoded
  runs/             run records + test metrics (checkpoints themselves are gitignored)
backend/            FastAPI app: predictions, molecules, benchmarks, jobs, data upload
frontend/           React + Vite dashboard
data/               bronze (raw, Git LFS) → silver → gold (training-ready)
reports/            generated benchmark tables and figures (do not hand-edit)
docs/               technical report, literature review, model cards, reviews
tests/              ml/ unit, physics and golden-run tests
```

Which phase runs is selected only by `model.phase` in the experiment YAML (`mpnn` | `blip` | `graph_stochastic`, plus the bake-off models). See [glossary.md §6](glossary.md) for the config schema.

## 8. Getting started

### Data

`data/bronze/` (~814 MB) is committed with **Git LFS**. Run `brew install git-lfs` (or `apt install git-lfs`), then `git lfs install`, then clone as normal. Without LFS, `scripts/download_bronze_data.sh` fetches the same files from their public sources. Then build the training-ready data:

```bash
python -m ml.data.preprocessing        # bronze → silver → gold
```

### ML pipeline (from the repo root, with `.venv` active)

```bash
pip install -e ".[dev]"
pytest tests/
python -m ml.training.train experiments/configs/phase1_baseline.yaml       # add --smoke-test for a quick check
python -m ml.training.evaluate --checkpoint experiments/checkpoints/phase1_baseline.pt
python -m ml.training.benchmark                                            # regenerates reports/
```

PyTorch Geometric needs a torch build that matches your platform; if installation fails, see pytorch.org and pytorch-geometric.readthedocs.io.

### Backend and frontend

```bash
cd backend && uvicorn app.main:app --reload     # http://localhost:8000
cd backend && pytest tests/
cd frontend && npm install && npm run dev       # http://localhost:5173
```

## 9. Limitations and open work

We report these openly because they bound what the results can claim:

- **No statistical significance testing yet.** Cross-model comparisons are single-run, single-seed. Treat the differences above as strong first reads, not settled results.
- **Phase 3 structural fix pending review.** The message-level noise variant gets force coverage close to nominal (91.6% at the 90% level). However, its force accuracy is worse (MAE 0.237) and energy stays underconfident, so it isn't a strict improvement. It stays out of the headline table until the Phase 3 owner signs off.
- **`phase2_blip` (the plain-MSE variant) was trained from a dirty tree.** The harness therefore refuses it, and it must be retrained from a clean commit before it can appear in any table.
- **MC sample count.** No stable K was demonstrated up to 100 samples with a 3-seed protocol (SCRUM-49). K = 20 is a pragmatic default, with that caveat.
- **Scope.** All results are on ethanol (DFT). Moving to more molecules, MD22 or real catalyst systems is future work, and the 5 Å cutoff may not transfer to much larger molecules.
- **Work on branches.** Several contributions in §5 live on feature branches (`ruturaj-phase3-graph-stochastic`, `shijin-scrum-72`, `dongxiao`) and are not yet merged into `main`.

## 10. Further reading

| Document | What's in it |
|---|---|
| [glossary.md](glossary.md) | **The spec:** team, architecture, folder ownership, API contracts, data and config schemas |
| [ROADMAP.md](ROADMAP.md) | Phase sequencing, weekly ownership, definition of done |
| [docs/technical_report.md](docs/technical_report.md) | Method, results and what worked and what didn't |
| [docs/literature_review.md](docs/literature_review.md) | Grounding for every architecture and UQ choice, plus project findings |
| [docs/phase1_protocol.md](docs/phase1_protocol.md) | The shared training and evaluation protocol |
| [docs/model_cards/](docs/model_cards/) | One card per trained model |
| [docs/reviews/](docs/reviews/) | Audits and analyses: leakage, calibration, MC stability, SCRUM-70 root cause |
| [reports/all_phases.md](reports/all_phases.md) | Generated table of every scored run, with re-score evidence |
| [docs/sprint_process.md](docs/sprint_process.md) | How the plan maps onto the unit's sprints and Jira |
| [CLAUDE.md](CLAUDE.md) | Condensed conventions and commands for working in the repo |
