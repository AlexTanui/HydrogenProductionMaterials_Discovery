# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

PG-S2-47 (Adelaide University, College of Engineering & IT): deep learning
(GNNs) and causal ML to predict hydrogen production catalyst/material
properties (catalytic activity, stability) for high-throughput virtual
screening.

**This repo currently holds structure only** — folders and empty
placeholder files, no implementation yet. Team structure, full
architecture, folder-by-folder ownership, API contracts, data schema, and
experiment config schema all live in one place:
[glossary.md](glossary.md). Read it before implementing anything here — it
is the spec, not just background reading.

## Layout

Three independently-runnable layers (`ml/`, `backend/`, `frontend/`) —
dependency only flows `ml → backend → frontend`, never the reverse. See
glossary.md §2–3 for the full breakdown of what goes in every folder and
who owns it.

## Commands (once each layer is implemented)

```bash
# ML pipeline (run from repo root, with .venv active)
pytest tests/
python -m ml.training.train --config experiments/configs/baseline.yaml
python -m ml.training.evaluate --checkpoint experiments/checkpoints/baseline.pt
python -m ml.training.benchmark

# Backend
cd backend && uvicorn app.main:app --reload      # http://localhost:8000
cd backend && pytest tests/

# Frontend
cd frontend && npm run dev                        # http://localhost:5173
cd frontend && npm run build
```

## Conventions

- **New model architecture** → add a class + registry entry in
  `ml/models/gnn.py`, then a config in `experiments/configs/`. Don't
  hardcode architecture choice elsewhere — everything reads
  `config.model.name` (schema in glossary.md §6).
- **New raw dataset source** → normalize to the canonical schema in
  glossary.md §5 via `ml/data/preprocessing.py` before it touches
  `ml/data/datasets.py`. Don't special-case source formats downstream.
- Config-driven experiments only: no hardcoded hyperparameters in training
  scripts — add/edit a YAML in `experiments/configs/` instead.
- New API endpoint → must match a contract in glossary.md §4. Route
  handlers go in `backend/app/api/`, registered in `backend/app/main.py`.
  Pydantic schemas in `backend/app/models/schemas.py`, SQLAlchemy models in
  `backend/app/models/db.py`. Keep inference logic in
  `backend/app/services/inference.py`, not in route handlers.
- Frontend calls the backend only through `frontend/src/api/client.ts` —
  don't `fetch()` directly from components.
- Every trained checkpoint reported in the technical report needs a model
  card (`docs/model_cards/`).

## Gotchas

- Real experimental/DFT data may not exist yet for local dev — build
  `MaterialsDataset` in `ml/data/datasets.py` with a synthetic fallback path
  so the pipeline stays runnable, but never treat synthetic-fallback
  results as real numbers in `docs/technical_report.md`.
- If the GNN's node feature width changes, both the training path (reads
  `dataset[0].x.shape[1]` automatically) and `backend/app/services/inference.py`
  (which will need the same `in_channels` value to load a checkpoint) must
  stay in sync — this is a common way inference breaks silently after a
  featurization change.
- `backend/app/models/db.py` defaults to SQLite — fine for local dev, not
  for concurrent writers. Confirm before changing.
- RDKit and PyTorch Geometric both have non-trivial install requirements
  (RDKit via conda is more reliable than pip on some platforms; PyG needs a
  matching torch/CUDA build). If install fails on these, check current
  instructions at pytorch.org and pytorch-geometric.readthedocs.io rather
  than pinning blindly.
