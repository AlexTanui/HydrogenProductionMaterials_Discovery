# Glossary — PG-S2-47: Hydrogen Production Materials Discovery Using Deep Neural Networks

Single source of truth for team structure, architecture, folder ownership,
API contracts, and terminology. This repo currently contains **structure
only** in `ml/` — folders and empty placeholder files there; `backend/` and
`frontend/` have a working interface-preview build. This document is the
spec each person builds their part against.

For *when* things happen — phase sequencing, weekly ownership, integration
checkpoints, definition of done, fallback order — see
[ROADMAP.md](ROADMAP.md).

**Client:** Adelaide University — College of Engineering & IT
**Agency supervisor:** Henry Li (<henry.li@adelaide.edu.au>)
**Academic supervisor:** Dhika Pratama (<dhika.pratama@adelaide.edu.au>)

**Relationship between the project title and the technical plan:** the
university project is titled *Hydrogen Production Materials Discovery*, but
the concrete, supervisor-issued technical research plan (*Adaptive
Graph-Space Uncertainty Modelling for Machine Learning Interatomic
Potentials*) is scoped around **MD17** — a standard molecular dynamics
benchmark — not a hydrogen-catalyst dataset. That's intentional, not a
mismatch: the deliverable is a validated modeling *methodology* (an MPNN
with rigorously evaluated uncertainty quantification), developed and proven
on a well-understood benchmark before it would ever be pointed at real
catalyst screening. Everything in this document reflects the MD17/MLIP
scope — that supersedes any earlier framing around `catalytic_activity`,
`stability`, or causal ML, which is no longer part of the technical plan.

---

## 1. Team structure

| Name | Role / Responsibility | Key Skills |
|---|---|---|
| **Alex Tanui** | Scrum Master / Technical Lead / General Project Engineer | Project Coordination, Software Engineering, Python, ML/AI Support, Data Engineering Support, System Integration, GitHub/Jira, Snowflake/AWS, Testing, Documentation |
| **Fazin Faizal** | Developer / QA & Benchmarking Engineer | Python, Deep Learning, Testing, QA, Benchmarking, Model Validation, Debugging, Performance Analysis |
| **Ruturaj Yashwant Bhosale** | ML/AI Engineer – GNN Lead | PyTorch, PyTorch Geometric, Graph Neural Networks, Deep Learning, Atomistic Modelling, Energy/Force Prediction, Model Training |
| **Shijin Mathew** | Data Engineer / ML Data Pipeline Lead | Python, NumPy, Data Preprocessing, MD17/MD22 Handling, AWS S3, Snowflake, Data Validation, Feature/Graph Preparation, Dataset Versioning |
| **Dongxiao Wu** | ML Research & Model Evaluation Engineer | Machine Learning, Literature Review, Baseline Models, Model Evaluation, Statistical Analysis, Error Analysis, Research & Benchmarking |

---

## 2. Architecture

Three independently-runnable layers. Dependency direction only ever flows
one way — never add a dependency in the reverse direction.

```text
┌─────────────────┐      ┌──────────────────────┐      ┌──────────────────┐
│   ml/            │      │   backend/            │      │   frontend/        │
│   (research core) │─────▶│   (FastAPI service)   │─────▶│   (React dashboard) │
└─────────────────┘      └──────────────────────┘      └──────────────────┘
        │                          │
        ▼                          ▼
  experiments/               backend/app.db
  (configs, checkpoints,     (prediction logs)
   results)
```

- **`ml/`** — the actual research deliverable: three MPNN-family models on
  MD17 (deterministic baseline, BLIP weight-space stochasticity, graph-space
  stochasticity), trained and evaluated for both prediction accuracy
  (energy/force MAE) and uncertainty quality (calibration, uncertainty–error
  correlation). This is what the literature review and technical report are
  written about. Has no dependency on `backend/` or `frontend/` — must be
  runnable and testable entirely on its own.
- **`backend/`** — FastAPI service. Imports from `ml/` to load a trained
  checkpoint and serve predictions over HTTP. Does not duplicate modeling
  logic.
- **`frontend/`** — React/Vite dashboard. Talks to `backend/` only through
  HTTP (its API client module) — no knowledge of PyTorch, graphs, or physics.

---

## 3. Folder-by-folder guide (who builds what)

### `ml/` — owner: Ruturaj (models) + Shijin (data) + Fazin (benchmarking/UQ metrics) + Dongxiao (evaluation, literature-informed Phase 3 design)

| Path | What goes here | Suggested owner |
|---|---|---|
| `ml/data/md17.py` | Load raw MD17 trajectory files (`z` atomic numbers, `R` coordinates, `E` energy, `F` forces) and build a neighbor graph per configuration (nodes = atoms, edges = atom pairs within a cutoff radius, edge feature = interatomic distance) | Shijin, with Ruturaj on graph feature design |
| `ml/data/preprocessing.py` | Trajectory-aware splitting (contiguous time blocks, **not** random — see §5) and any unit/coordinate normalization | Shijin |
| `ml/data/datasets.py` | PyTorch Geometric `Dataset` wrapper producing `Data(x, pos, edge_index, edge_attr, y=energy, force=forces)` | Shijin |
| `ml/models/mpnn.py` | **Shared backbone + Phase 1** — a simple custom MPNN (Gilmer-style message passing with RBF/Gaussian-expanded distance edge features, not SchNet/PaiNN/DimeNet++) predicting total energy; forces via autograd (`F = -∂E/∂R`). Phases 2 and 3 both extend this same backbone rather than introducing a different architecture, so any difference in results is attributable to *where* stochasticity lives, not to three unrelated models | Ruturaj |
| `ml/models/blip.py` | **Phase 2** — the Phase 1 backbone with BLIP-style input-dependent Gaussian stochasticity added to the message/update weights, evaluated via multiple stochastic forward passes | Ruturaj, with Dongxiao on calibration analysis |
| `ml/models/graph_stochastic.py` | **Phase 3** — the Phase 1 backbone with stochasticity moved into node and/or edge representations instead of weights (dropout, Gaussian perturbation, or learnable perturbation strength — approach to be settled by the literature review, see `docs/literature_review.md`), compared against Phases 1 and 2 | Whole team; Ruturaj implements, Dongxiao evaluates |
| `ml/training/train.py` | Training loop, config-driven, combined energy+force loss | Ruturaj |
| `ml/training/evaluate.py` | Scores a checkpoint: energy/force MAE always; ECE + uncertainty–error correlation for Phase 2/3 | Fazin |
| `ml/training/benchmark.py` | Runs all three phases and produces the single comparison table (§4 `/benchmarks` shape) for the report | Fazin |
| `ml/utils/metrics.py` | `energy_mae`, `force_mae`, `expected_calibration_error`, `uncertainty_error_correlation` (Spearman) | Fazin |
| `ml/utils/logging.py` | Shared run logging | whoever needs it first |
| `ml/config.py` | The `ExperimentConfig` schema all YAML configs must match (§6) | Ruturaj, agreed with Fazin/Dongxiao |

### `experiments/` — owner: whoever runs the experiment

| Path | What goes here |
|---|---|
| `experiments/configs/*.yaml` | One file per run: which phase/molecule/hyperparameters. Schema in §6. |
| `experiments/checkpoints/` | Trained model weights (gitignored — never commit these) |
| `experiments/results/` | Per-run metrics JSON + combined benchmark CSV, produced by `ml/training/evaluate.py` / `benchmark.py` |

### `backend/` — owner: Alex, with Fazin on test coverage

| Path | What goes here | Owner |
|---|---|---|
| `backend/app/api/` | One route module per resource — must implement the contracts in §4 exactly | Alex |
| `backend/app/models/db.py` | SQLAlchemy tables (prediction logs) | Alex |
| `backend/app/models/schemas.py` | Pydantic request/response models matching §4 | Alex |
| `backend/app/services/inference.py` | The **only** place that loads/calls a model from `ml/` | Alex |
| `backend/app/core/config.py` | Settings (DB URL, checkpoint path, CORS) | Alex |
| `backend/tests/` | API tests | Fazin |

### `frontend/` — owner: Alex (integration), open to whoever wants frontend exposure

| Path | What goes here |
|---|---|
| `frontend/src/pages/Dashboard.tsx` | Landing page: project purpose, the three-phase research plan |
| `frontend/src/pages/Predict.tsx` | Pick an MD17 molecule + frame, calls `POST /predictions`, shows predicted energy/force/uncertainty |
| `frontend/src/pages/Benchmarks.tsx` | Chart/table comparing all three phases on `GET /benchmarks` |
| `frontend/src/api/client.ts` | The **only** file allowed to call `fetch()` — pages must go through here |

### `docs/` — deliverable placeholders (to be filled in as work completes)

| Path | Deliverable it corresponds to | Owner |
|---|---|---|
| `docs/literature_review.md` | Deliverable 1: literature review — Gilmer et al. 2017, Schütt et al. 2017 (SchNet), BLIP, GRAND, DropEdge, DropConn, Edge-Variational GCNs; used to settle the Phase 3 approach | Dongxiao |
| `docs/data_dictionary.md` | MD17 schema + trajectory-splitting rationale (detail behind §5) | Shijin |
| `docs/architecture.md` | Expanded version of §2, kept in sync as the system evolves | Alex |
| `docs/api.md` | Expanded version of §4, kept in sync as endpoints are built | Alex |
| `docs/technical_report.md` | Deliverable 5: final technical report, built around the Three-Phase Technical Summary table | Dongxiao, assembled with input from all |
| `docs/model_cards/` | One card per checkpoint (one per phase, minimum) | Ruturaj / Fazin |

### `tests/` — owner: Fazin

`tests/test_md17.py`, `tests/test_models.py` — unit tests for the `ml/`
package (separate from `backend/tests/`, which covers the API).

### `data/` — owner: Shijin — staged bronze → silver → gold

Nothing under `data/` is committed except `.gitkeep` placeholders — the
payload is ~1GB and, critically, **git never carries it to anyone**.
Cloning this repo gets you the empty folder structure and nothing inside
it. Everyone gets the actual bronze data by running:

```bash
scripts/download_bronze_data.sh
```

which fetches the MD17/MD22 files directly from their public sources
(quantum-machine.org / sgdml.org — both public benchmark hosts, no
credentials needed) into `data/bronze/{md17,md22}/`. It skips files
already present, so it's safe to re-run. This is why bronze data is never
manually copied between machines or committed — it's reproducible from a
public source instead.

Each stage is split into `md17/` and `md22/` subfolders — the two datasets
have different molecule scales (MD22 atoms counts run up to 370 vs. MD17's
~9–24) and shouldn't be loaded/split/benchmarked as if they were one thing.

| Stage | Path | Contents |
|---|---|---|
| **Bronze** | `data/bronze/md17/`, `data/bronze/md22/` | Raw, untouched, as-downloaded. Whatever format the source gives — `.npz`, or `.zip` containing `.xyz`/`.npz`. Never edited in place. |
| **Silver** | `data/silver/md17/`, `data/silver/md22/` | Deduplicated, format-unified, validated. One `.npz` per molecule+theory-level, always in the `z/R/E/F` shape from §5, with shape/NaN/unit checks passed. |
| **Gold** | `data/gold/md17/`, `data/gold/md22/` | Production-ready. Silver data with the trajectory-block train/val/test split (§5) applied and saved alongside it — this is the only stage `ml/data/datasets.py` reads from. |

`ml/data/preprocessing.py` owns bronze → silver → gold; `ml/data/md17.py`
owns silver/gold → PyG graph construction at train time.

**What's actually in bronze right now** (inspected 2026-08-27 — update this
table as the roster changes):

| File | Molecule | Theory | Atoms | Configs | Note |
|---|---|---|---|---|---|
| `md17/md17_ethanol.npz` | ethanol | DFT (aims, PBE+TS, light tier 1) | 9 | 555,092 | |
| `md17/ethanol_ccsd_t.zip` / `ethanol_ccsd_t (1).zip` | ethanol | CCSD(T) | 9 | — | **Same split, two formats** (one has `.npz` inside, the other `.xyz`) — not byte-duplicates (different md5), pick one format at silver, don't load both |
| `md17/benzene2018_dft.npz` | benzene | DFT PBE-TS 500K | 12 | 49,863 | |
| `md17/azobenzene_dft.npz` + `.zip` | azobenzene | DFT | 24 | 99,999 | zip's `.xyz` is redundant with the npz — drop at silver |
| `md17/paracetamol_dft.npz` + `.zip` | paracetamol | DFT PBE-TS 500K | 20 | 106,490 | same redundancy as azobenzene |
| `md17/aspirin_ccsd.zip` | aspirin | CCSD | 21 | — | ships with a **literature-standard train/test split already applied** (`-train.xyz`/`-test.xyz`) — preserve it rather than re-splitting, for comparability with published MD17 results |
| `md17/malonaldehyde_ccsd_t.zip` | malonaldehyde | CCSD(T) | 9 | — | same pre-split situation as aspirin |
| `md17/toluene_ccsd_t.zip` | toluene | CCSD(T) | 15 | — | same pre-split situation as aspirin |
| `md17/md17_uracil.npz` | uracil | DFT | 12 | 133,770 | fetched by `scripts/download_bronze_data.sh`; the originally-uploaded `.zip` (`.xyz` only) is superseded by this — safe to remove |
| `md22/md22_AT-AT-CG-CG.npz` + `.zip` | AT-AT-CG-CG | DFT PBE+MBD 500K | 118 | 10,153 | zip redundant with npz |
| `md22/md22_DHA.npz` | DHA | DFT | 56 | 69,753 | |
| `md22/md22_buckyball-catcher.npz` | buckyball-catcher | DFT (FHI-aims, PBE-MBD) | 148 | 6,102 | |
| `md22/md22_double-walled_nanotube.npz` | double-walled nanotube | DFT | 370 | 5,032 | fetched by `scripts/download_bronze_data.sh`; the originally-uploaded `.zip` (`.xyz` only) is superseded by this — safe to remove |

Two things this inventory makes concrete for the silver step:

1. **Theory level is not cosmetic metadata — track it as part of the
   dataset key.** CCSD(T) is a materially more accurate (and more
   expensive to obtain) reference than DFT. `ethanol` exists at both
   levels here; train/evaluate them as separate datasets
   (`ethanol_dft`, `ethanol_ccsd_t`), never pooled.
2. **Where a literature-standard split already ships with the data
   (aspirin, malonaldehyde, toluene, and the CCSD(T) ethanol pair),
   preserve it** instead of applying the trajectory-block split from
   scratch — it's what published benchmarks compare against, and
   re-splitting throws that comparability away for no benefit.

### `scripts/` — owner: Shijin

`scripts/download_bronze_data.sh` — fetches MD17/MD22 from their public
sources into `data/bronze/`. The only sanctioned way bronze data gets onto
a machine; see the `data/` section above.

### `reports/` — owner: whoever generates the figure

`reports/figures/` — plots/diagrams referenced from `docs/technical_report.md`.

---

## 4. API contracts

Base URL (dev): `http://localhost:8000`. The frontend must reach these only
through `frontend/src/api/client.ts`, proxied via `/api/*`.

### `GET /health`

Liveness check. Response:

```json
{ "status": "ok" }
```

### `POST /predictions`

Predict energy, force magnitude, and (for stochastic phases) uncertainty
for one MD17 configuration. Every call is logged for later analysis.

Request:

```json
{ "molecule": "aspirin", "frame_index": 0 }
```

Response:

```json
{
  "molecule": "aspirin",
  "frame_index": 0,
  "predicted_energy": -406234.5,
  "predicted_force_rms": 12.3,
  "uncertainty": 0.08,
  "phase": "phase1_deterministic"
}
```

`predicted_energy` in kcal/mol, `predicted_force_rms` in kcal/mol/Å (root-
mean-square over all atoms — the per-atom force vector field itself isn't
serialized here, only its summary magnitude). `uncertainty` is `null` for
the Phase 1 deterministic model, which has none by construction.

### `GET /molecules`

The fixed catalog of MD17 molecules available to predict on — a static
reference list, not a live database (MD17's molecule set doesn't change).

```json
[
  { "molecule": "aspirin", "num_atoms": 21 },
  { "molecule": "ethanol", "num_atoms": 9 },
  { "molecule": "benzene", "num_atoms": 12 }
]
```

### `GET /benchmarks`

Every entry from `experiments/results/*.json` — the Three-Phase Technical
Summary, in the shape the report table (§6 of the proposal) needs directly:

```json
[
  {
    "phase": "phase1_deterministic",
    "model": "mpnn",
    "energy_mae": 0.42,
    "force_mae": 1.15,
    "ece": null,
    "uncertainty_correlation": null
  },
  {
    "phase": "phase2_blip",
    "model": "blip_mpnn",
    "energy_mae": 0.39,
    "force_mae": 1.08,
    "ece": 0.05,
    "uncertainty_correlation": 0.61
  }
]
```

`ece` and `uncertainty_correlation` are `null` for the deterministic phase
(no uncertainty to calibrate).

### Auth

None. This is an internal research demo — do not deploy publicly without
adding auth first.

---

## 5. Data schema (MD17 + MD22)

Each sample is one snapshot along a molecular dynamics trajectory — either
one of the standard MD17 small organic molecules, or one of the larger MD22
systems (supramolecules, a nanotube). See §3's inventory table for exactly
which molecules and theory levels are in `data/bronze/` right now.

| Field | Type | Description |
|---|---|---|
| `molecule` | str | Which molecule (e.g. `ethanol`, `aspirin`, `AT-AT-CG-CG`) |
| `theory` | str | Level of quantum-chemistry theory (`DFT`, `CCSD`, `CCSD(T)`) — **part of the dataset's identity, not metadata to discard.** Never pool different theory levels of the same molecule as if they were one dataset |
| `z` | int[num_atoms] | Atomic numbers — defines node identity |
| `R` | float[num_atoms, 3] | 3D atomic coordinates in Å — defines node positions and, via a cutoff radius, the edge set |
| `E` | float | Total potential energy (kcal/mol) — the primary regression target |
| `F` | float[num_atoms, 3] | Per-atom force vectors (kcal/mol/Å) — the secondary regression target, typically weighted higher in the training loss than energy (see §6) |

Confirmed from inspecting the actual files: every source file already uses
Å and kcal/mol (`r_unit`/`e_unit` fields where present), so no unit
conversion is needed across the current roster — but validate this
assumption at silver rather than assuming it holds for anything added later.

**Graph construction:** nodes = atoms, node feature = embedded atomic
number `z`. Edges connect atom pairs within a cutoff radius (default 5 Å),
edge feature = interatomic distance. This replaces SMILES/CIF-based
featurization for this phase of the project — SMILES doesn't carry the 3D
geometry an energy/force model needs. MD22's larger systems (up to 370
atoms) make the cutoff radius matter more than it does for MD17 — a fixed
5 Å radius produces a much sparser relative graph on the nanotube than on
ethanol; revisit per-molecule if Phase 1 underperforms on the larger systems.

**Splitting:** adjacent trajectory frames are nearly identical (the MD
timestep is small). A **random** frame split leaks near-duplicate
structures across train/test and inflates apparent accuracy. Default to
contiguous trajectory blocks instead (e.g. first 80% of timesteps → train,
last 20% → test) — **except** where the bronze data already ships a
literature-standard split (aspirin, malonaldehyde, toluene, CCSD(T)
ethanol — see §3), which should be preserved as-is for comparability with
published results. Both cases are handled by `ml/data/preprocessing.py` at
the silver → gold step.

---

## 6. Experiment config schema

Every training run must be defined by a YAML file in `experiments/configs/`
matching this shape — no hardcoded hyperparameters in training scripts.

```yaml
name: phase1_baseline
data:
  dataset: md17               # md17 | md22 — which top-level folder under data/
  molecule: aspirin           # see §3 inventory for the current bronze roster
  theory: ccsd                 # dft | ccsd | ccsd_t — must match, never pool across levels
  gold_path: data/gold/md17/aspirin_ccsd.npz  # training reads gold only, never bronze/silver directly
  cutoff_radius: 5.0         # Å, for neighbor graph construction
  train_split: 0.8           # contiguous trajectory blocks, not random — see §5
  val_split: 0.1
  test_split: 0.1
  use_literature_split: true # true for aspirin/malonaldehyde/toluene/ccsd_t-ethanol — see §3
  seed: 42
model:
  phase: mpnn               # mpnn | blip | graph_stochastic
  hidden_channels: 128
  num_layers: 4
  num_mc_samples: 20         # stochastic forward passes for UQ (blip / graph_stochastic only)
  perturbation_strength: 0.1 # graph_stochastic only
train:
  epochs: 200
  batch_size: 32
  lr: 0.0005
  energy_loss_weight: 1.0
  force_loss_weight: 100.0   # forces conventionally weighted much higher than energy
  patience: 20
  device: cpu                # cpu | cuda
```

---

## 7. Glossary of terms

| Term | Meaning |
|---|---|
| **MLIP** | Machine Learning Interatomic Potential — a model that approximates DFT, predicting energy/forces from atomic species + coordinates |
| **MPNN** | Message Passing Neural Network — atoms as nodes, interactions as edges; node representations are updated by aggregating neighbor information across edges |
| **SchNet** | A continuous-filter convolutional MPNN variant (Schütt et al., 2017) — required Phase 1 background reading, but **not** the Phase 1 architecture itself; Phase 1 is a simpler Gilmer-style MPNN (see `ml/models/mpnn.py` in §3) |
| **BLIP** | Bayesian Learned Interatomic Potentials — introduces input-dependent Gaussian stochasticity into MPNN message/update *weights*, enabling uncertainty estimation via stochastic forward passes (Phase 2) |
| **Graph-space stochasticity** | This project's own direction: injecting stochasticity into node/edge *representations* instead of weights (Phase 3) |
| **UQ** | Uncertainty Quantification — the model reports not just a prediction, but how much to trust it |
| **ECE** | Expected Calibration Error — measures whether predicted uncertainty actually matches observed error (well-calibrated = low ECE) |
| **Uncertainty–error correlation** | Spearman rank correlation between predicted uncertainty and actual prediction error — should be strongly positive for useful UQ |
| **OOD** | Out-of-distribution — structures unlike the training data, where an MLIP's predictions become unreliable; the whole motivation for UQ |
| **MD17** | Standard benchmark dataset: molecular dynamics trajectories with reference DFT/CCSD/CCSD(T) energy/forces for small organic molecules |
| **MD22** | Companion benchmark to MD17 with much larger systems (supramolecules, a nanotube) — tests whether an MLIP generalizes beyond small-molecule scale |
| **DFT** | Density Functional Theory — the quantum-chemistry simulation method whose output MLIPs approximate |
| **Autograd force prediction** | Computing per-atom forces as `F = -∂E/∂R` via automatic differentiation of a predicted energy, which keeps the potential energy-conserving |
| **Checkpoint** | A saved snapshot of trained model weights (`experiments/checkpoints/*.pt`) |
| **Synthetic fallback / stub** | Placeholder deterministic output used to verify the pipeline runs end-to-end before a real checkpoint exists — never used for reported results |
