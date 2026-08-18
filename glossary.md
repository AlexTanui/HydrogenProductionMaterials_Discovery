# Glossary — PG-S2-47: Hydrogen Production Materials Discovery Using Deep Neural Networks

Single source of truth for team structure, architecture, folder ownership,
API contracts, and terminology. This repo currently contains **structure
only** — folders and empty placeholder files. No implementation exists yet;
this document is the spec each person builds their part against.

**Client:** Adelaide University — College of Engineering & IT
**Agency supervisor:** Henry Li (<henry.li@adelaide.edu.au>)
**Academic supervisor:** Dhika Pratama (<dhika.pratama@adelaide.edu.au>)

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
  (configs, checkpoints,     (SQLite: materials,
   results)                  prediction logs)
```

- **`ml/`** — the actual research deliverable. Data featurization, GNN
  models, causal ML, training/evaluation/benchmarking. This is what the
  literature review and technical report are written about. Has no
  dependency on `backend/` or `frontend/` — must be runnable and testable
  entirely on its own.
- **`backend/`** — FastAPI service. Imports from `ml/` to load a trained
  checkpoint and serve predictions over HTTP. Does not duplicate modeling
  logic.
- **`frontend/`** — React/Vite dashboard. Talks to `backend/` only through
  HTTP (its API client module) — no knowledge of PyTorch, graphs, or
  chemistry.

---

## 3. Folder-by-folder guide (who builds what)

### `ml/` — owner: Ruturaj (models) + Shijin (data) + Dongxiao (causal/eval)

| Path | What goes here | Suggested owner |
|---|---|---|
| `ml/data/preprocessing.py` | Normalize raw dataset exports (Catalysis-Hub, Materials Project, Open Catalyst Project, MD17/MD22) into the canonical schema (§5) | Shijin |
| `ml/data/featurization.py` | Convert a normalized record (SMILES or CIF/structure) into a graph representation (nodes, edges, node/edge features) | Shijin, with Ruturaj on graph feature design |
| `ml/data/datasets.py` | PyTorch Geometric `Dataset`/`InMemoryDataset` wrapper over the featurized data, train/val/test split logic | Shijin |
| `ml/models/gnn.py` | GNN architectures (GCN / GAT / MPNN or equivalent) that regress catalytic activity / stability from a graph | Ruturaj |
| `ml/models/causal.py` | Causal ML (e.g. double machine learning) to estimate which structural features causally affect the target, controlling for confounders (dataset source, synthesis conditions) | Dongxiao |
| `ml/training/train.py` | Training loop, config-driven (reads an experiment YAML, never hardcodes hyperparameters) | Ruturaj |
| `ml/training/evaluate.py` | Scores a trained checkpoint on the held-out test split, writes metrics | Fazin |
| `ml/training/benchmark.py` | Runs every experiment config and produces one comparison table for the report | Fazin |
| `ml/utils/metrics.py` | Shared regression metrics (MAE, RMSE, R²) | Fazin |
| `ml/utils/logging.py` | Shared run logging | whoever needs it first |
| `ml/config.py` | The `ExperimentConfig` schema all YAML configs must match (§6) | Ruturaj, agreed with Fazin/Dongxiao |

### `experiments/` — owner: whoever runs the experiment

| Path | What goes here |
|---|---|
| `experiments/configs/*.yaml` | One file per experiment: architecture choice + hyperparameters. Schema in §6. |
| `experiments/checkpoints/` | Trained model weights (gitignored — never commit these) |
| `experiments/results/` | Per-experiment metrics JSON + combined benchmark CSV, produced by `ml/training/evaluate.py` / `benchmark.py` |

### `backend/` — owner: Alex, with Fazin on test coverage

| Path | What goes here | Owner |
|---|---|---|
| `backend/app/api/` | One route module per resource — must implement the contracts in §4 exactly | Alex |
| `backend/app/models/db.py` | SQLAlchemy tables (materials, prediction logs) | Alex |
| `backend/app/models/schemas.py` | Pydantic request/response models matching §4 | Alex |
| `backend/app/services/inference.py` | The **only** place that loads/calls a model from `ml/` | Alex |
| `backend/app/core/config.py` | Settings (DB URL, checkpoint path, CORS) | Alex |
| `backend/tests/` | API tests | Fazin |

### `frontend/` — owner: Alex (integration), open to whoever wants frontend exposure

| Path | What goes here |
|---|---|
| `frontend/src/pages/Dashboard.tsx` | Landing page / project summary |
| `frontend/src/pages/Predict.tsx` | Form to submit a candidate material, calls `POST /predictions` |
| `frontend/src/pages/Benchmarks.tsx` | Table/chart of `GET /benchmarks` results |
| `frontend/src/api/client.ts` | The **only** file allowed to call `fetch()` — pages must go through here |

### `docs/` — deliverable placeholders (empty, to be filled in as work completes)

| Path | Deliverable it corresponds to | Owner |
|---|---|---|
| `docs/literature_review.md` | Deliverable 1: literature review of AI methods | Dongxiao |
| `docs/data_dictionary.md` | Canonical data schema + source provenance (detail behind §5) | Shijin |
| `docs/architecture.md` | Expanded version of §2, kept in sync as the system evolves | Alex |
| `docs/api.md` | Expanded version of §4, kept in sync as endpoints are built | Alex |
| `docs/technical_report.md` | Deliverable 5: final technical report | Dongxiao, assembled with input from all |
| `docs/model_cards/` | One card per checkpoint reported in the technical report | Ruturaj / Fazin |

### `tests/` — owner: Fazin

`tests/test_featurization.py`, `tests/test_models.py` — unit tests for the
`ml/` package (separate from `backend/tests/`, which covers the API).

### `data/` — owner: Shijin

`data/raw/` (untouched source exports, gitignored), `data/processed/`
(output of `ml/data/preprocessing.py`, gitignored), `data/external/`
(reference data, e.g. lookup tables). Nothing here is committed except the
`.gitkeep` placeholders — data ships separately (S3/Snowflake per Shijin's
pipeline), never into git.

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

Score one candidate material with the currently loaded checkpoint. Every
call must be logged (material queried, prediction, model version) for later
analysis.

Request:

```json
{ "smiles": "O=C=O" }
```

Response:

```json
{
  "smiles": "O=C=O",
  "predicted_activity": 0.4123,
  "model_name": "baseline"
}
```

### `GET /materials?skip=0&limit=50`

List ingested materials, paginated. Response:

```json
[
  {
    "material_id": "mp-1234",
    "smiles": "O=C=O",
    "source": "materials-project",
    "catalytic_activity": 0.55,
    "stability": 0.02
  }
]
```

### `GET /materials/{material_id}`

Single material record, same shape as one list item above. 404 if not found.

### `GET /benchmarks`

Every entry from `experiments/results/*.json`, for the frontend Benchmarks
page and the technical report. Response:

```json
[
  { "experiment": "baseline", "model": "gcn", "mae": 0.12, "mse": 0.02, "rmse": 0.14, "r2": 0.81 }
]
```

### Auth

None. This is an internal research demo — do not deploy publicly without
adding auth first.

---

## 5. Canonical data schema

All raw source exports must be normalized to this schema by
`ml/data/preprocessing.py` before they reach `ml/data/datasets.py`. Do not
special-case source formats downstream of this point.

| Column | Type | Description |
|---|---|---|
| `material_id` | str | Stable unique ID, source-prefixed (e.g. `mp-1234`, `oc20-5678`) |
| `smiles` | str | SMILES for molecular species/adsorbates. Empty for bulk crystals — use `cif_path` instead |
| `catalytic_activity` | float | Primary regression target — normalized activity proxy (e.g. scaled negative adsorption energy for HER/OER) |
| `stability` | float | Formation energy above hull, or equivalent stability proxy (lower = more stable) |
| `source` | str | One of `catalysis-hub`, `materials-project`, `open-catalyst`, `md17`, `md22` |

Candidate sources to evaluate and document in `docs/data_dictionary.md`:
Catalysis-Hub.org (DFT adsorption energies), Materials Project (formation
energy, stability, structures), Open Catalyst Project (OC20/OC22), MD17/MD22
(molecular dynamics trajectories, relevant to Shijin's data handling scope).

Known confounders to control for in causal analysis: DFT functional used,
synthesis method, dataset source (see `ml/models/causal.py`).

---

## 6. Experiment config schema

Every training run must be defined by a YAML file in `experiments/configs/`
matching this shape — no hardcoded hyperparameters in training scripts.

```yaml
name: baseline
data:
  raw_path: data/raw/materials.csv
  target: catalytic_activity
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  seed: 42
model:
  name: gcn        # gcn | gat | mpnn
  hidden_channels: 64
  num_layers: 3
  dropout: 0.1
train:
  epochs: 50
  batch_size: 32
  lr: 0.001
  weight_decay: 0.00001
  patience: 10
  device: cpu       # cpu | cuda
```

---

## 7. Glossary of terms

| Term | Meaning |
|---|---|
| **GNN** | Graph Neural Network — a model architecture operating on graph-structured data (atoms as nodes, bonds/proximity as edges) |
| **GCN / GAT / MPNN** | Specific GNN variants: Graph Convolutional Network, Graph Attention Network, Message-Passing Neural Network |
| **Causal ML / DML** | Causal machine learning; Double Machine Learning — estimates the effect of one feature on an outcome while controlling for confounders, distinct from correlation |
| **Confounder** | A variable that influences both a feature and the outcome, risking a spurious correlation being mistaken for causation |
| **HER / OER** | Hydrogen / Oxygen Evolution Reaction — the two half-reactions in water splitting for hydrogen production; the catalytic activity target relates to these |
| **DFT** | Density Functional Theory — the quantum-chemistry simulation method used to compute most "ground truth" labels (formation energy, adsorption energy) in materials datasets |
| **Checkpoint** | A saved snapshot of trained model weights (`experiments/checkpoints/*.pt`) |
| **SMILES** | A text notation for molecular structure, used as the input format for molecular (as opposed to bulk crystal) featurization |
| **CIF** | Crystallographic Information File — standard format for crystal structure data |
| **ATE** | Average Treatment Effect — the output of a causal estimator, quantifying a feature's causal impact on the target |
| **Synthetic fallback** | Placeholder toy data used only to verify the pipeline runs end-to-end before real datasets are ingested — never used for reported results |
