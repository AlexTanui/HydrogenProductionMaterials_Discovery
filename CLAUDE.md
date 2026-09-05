# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

PG-S2-47 (Adelaide University, College of Engineering & IT). The university
project is titled *Hydrogen Production Materials Discovery*, but the
supervisor-issued technical plan (*Adaptive Graph-Space Uncertainty
Modelling for Machine Learning Interatomic Potentials*) is scoped around
**MD17**, a standard molecular-dynamics benchmark — building a
message-passing GNN that predicts energy and forces with rigorously
evaluated uncertainty. That's deliberate: the deliverable is a validated
*methodology*, proven on a well-understood benchmark before it would ever
be pointed at real catalyst screening. This supersedes any earlier framing
around `catalytic_activity`, `stability`, or causal ML — those are no
longer part of the technical plan.

**Current state:** `ml/data/` is implemented and verified; `ml/models/`,
`ml/training/`, and `ml/utils/` are still empty placeholders; `backend/`
and `frontend/` are at interface-preview stage (real code, stub
predictions).

Team structure, full architecture, folder-by-folder ownership, API
contracts, data schema, and experiment config schema all live in one place:
[glossary.md](glossary.md). Read it before implementing anything here — it
is the spec, not just background reading. For *when* things happen — phase
sequencing, weekly ownership, integration checkpoints, definition of done —
see [ROADMAP.md](ROADMAP.md).

## The three-phase plan

The research deliverable is three MPNN-family models on MD17, evaluated for
both accuracy (energy/force MAE) and uncertainty quality. Phases are
sequential — each extends the previous phase's backbone rather than
introducing a new architecture, so any difference in results is
attributable to *where* stochasticity lives, not to three unrelated models.

| Phase | What | Lives in |
|---|---|---|
| 1 | Deterministic Gilmer-style MPNN; total energy, forces via autograd (`F = -∂E/∂R`) | `ml/models/mpnn.py` |
| 2 | Phase 1 backbone + BLIP input-dependent Gaussian stochasticity in message/update **weights**; UQ via multiple stochastic forward passes | `ml/models/blip.py` |
| 3 | Phase 1 backbone + stochasticity moved into node/edge **representations** instead of weights (exact mechanism is a team decision from the literature review) | `ml/models/graph_stochastic.py` |

Which phase runs is selected by `model.phase` in the experiment YAML
(`mpnn` | `blip` | `graph_stochastic`) — see glossary.md §6 for the full
config schema. Never hardcode a phase choice in training or evaluation
code. Timeline, per-week ownership, and definition-of-done per phase are in
[ROADMAP.md](ROADMAP.md).

## Data layer conventions

`ml/data/` is implemented and verified (9/9 MD17 bronze sources process
clean). Full detail in [docs/data_dictionary.md](docs/data_dictionary.md);
the contract that matters everywhere downstream:

**Array names are fixed: `z`, `R`, `E`, `F`.** They carry through bronze,
silver, gold, and `MD17Sample` unchanged. Don't rename them to
`atomic_numbers`/`positions`/`energy`/`forces` at any layer — every stage
and the data dictionary assume these four names.

**Units are fixed and already consistent across the whole current roster:**

| Array | Meaning | Unit |
|---|---|---|
| `z` | atomic numbers | — |
| `R` | 3D coordinates | Å (`r_unit="Ang"`) |
| `E` | total potential energy | kcal/mol (`e_unit="kcal/mol"`) |
| `F` | per-atom forces | kcal/mol/Å |

No conversion is needed anywhere today: `validate_sample` in
`ml/data/preprocessing.py` raises and fails the silver step if a source
*declares* anything else, so metrics can assume kcal/mol without
converting. Note the limit of that check — a source file that carries no
`r_unit`/`e_unit` fields is defaulted to `Ang`/`kcal/mol` by `MD17Sample`
and passes, so it validates declared units, not actual ones. Report the
unit in metric output anyway; a future MD22 or external source could arrive
in eV, and MAE in the wrong unit is wrong by a silent constant factor
rather than visibly broken.

**`MD17Sample` (`ml/data/md17.py`)** is the validated in-memory form of one
molecule+theory. `load_npz()` returns one; `load_split_zip()` returns
`{"train": MD17Sample, "test": MD17Sample}`. Its `__post_init__` normalizes
and validates, so downstream code can rely on:

- `z` — `int64`, shape `[n_atoms]`
- `R` — `float32`, shape `[n_configs, n_atoms, 3]` (a single 2D frame is
  promoted to 3D, so `R` is *always* batched)
- `E` — `float32`, shape `[n_configs]`
- `F` — `float32`, same shape as `R`
- `molecule`, `theory`, `r_unit`, `e_unit` — identity/units
- `n_atoms`, `n_configs` — properties

Shape disagreement between `z`/`R`/`E`/`F` raises at construction, not
downstream.

**Theory level is part of dataset identity.** `ethanol` exists at both DFT
and CCSD(T). Never pool theory levels as one dataset; every silver/gold
file is named `{molecule}_{theory}.npz` so it's structurally hard to do by
accident. Metrics and benchmark rows must be keyed on molecule *and*
theory.

**Training reads gold only** — never bronze/silver directly. `MD17Dataset`
(`ml/data/datasets.py`) yields PyG `Data(x, pos, edge_index, edge_attr,
y=energy, force=forces)`, building the neighbor graph per frame at train
time via `build_graph` (5 Å cutoff, 16-bin RBF-expanded distance edges —
not raw scalar distance).

**Splits are contiguous trajectory blocks, never random** — adjacent MD
frames are near-duplicates, so a random split leaks near-identical
structures and inflates apparent accuracy. Where a literature-standard
split ships with the source (aspirin, malonaldehyde, toluene, CCSD(T)
ethanol) it is preserved exactly, for comparability with published numbers.
Any evaluation that re-splits gold data is a bug.

**Adding an element:** add it to `_PERIODIC_TABLE` in `ml/data/md17.py`.
Unknown symbols raise deliberately — they used to fall back to atomic
number 0, which silently corrupts the graph.

## Working as Fazin (QA & Benchmarking Engineer)

Fazin owns **evaluation and verification**, not model implementation:

| Owns | Path |
|---|---|
| Metric implementations | `ml/utils/metrics.py` |
| Checkpoint scoring | `ml/training/evaluate.py` |
| Three-phase comparison harness | `ml/training/benchmark.py` |
| `ml/` unit tests | `tests/` |
| API test coverage | `backend/tests/` |

Does **not** own: the models (`ml/models/*` — Ruturaj), the data pipeline
(`ml/data/*` — Shijin), or backend/frontend implementation (Alex). When
work touches those, propose the change to the owner rather than editing
directly.

**Dongxiao owns UQ metric *definitions*** — ECE, uncertainty–error Spearman
correlation, and the calibration analysis built on them. Fazin implements
against those definitions and wires them into the benchmark harness, but
doesn't unilaterally decide binning strategy, correlation choice, or what
counts as calibrated. Confirm the definition with Dongxiao before
implementing, and cite it in the docstring.

The harness must enforce a **fair comparison**: same molecule, same theory,
same gold splits, same MC sample count across phases. That fairness check
is Fazin's responsibility — a benchmark table comparing phases scored under
different conditions is worse than no table.

## Layout

Three independently-runnable layers (`ml/`, `backend/`, `frontend/`) —
dependency only flows `ml → backend → frontend`, never the reverse. See
glossary.md §2–3 for the full breakdown of what goes in every folder and
who owns it.

## Commands (once each layer is implemented)

```bash
# ML pipeline (run from repo root, with .venv active)
pytest tests/
python -m ml.data.preprocessing                   # bronze → silver → gold (implemented)
python -m ml.training.train --config experiments/configs/phase1_baseline.yaml
python -m ml.training.evaluate --checkpoint experiments/checkpoints/phase1_baseline.pt
python -m ml.training.benchmark

# Backend
cd backend && uvicorn app.main:app --reload      # http://localhost:8000
cd backend && pytest tests/

# Frontend
cd frontend && npm run dev                        # http://localhost:5173
cd frontend && npm run build
```

## Conventions

- **New model variant** → it extends the Phase 1 backbone in
  `ml/models/mpnn.py` rather than replacing it; add it to the phase file it
  belongs to (`mpnn.py` / `blip.py` / `graph_stochastic.py`), then a config
  in `experiments/configs/`. Don't hardcode the phase choice elsewhere —
  everything reads `config.model.phase` (schema in glossary.md §6).
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

- Real MD17 data ships with the repo via Git LFS (`data/bronze/`, ~814MB) —
  no synthetic fallback exists or is needed. If `data/gold/` is empty, run
  `python -m ml.data.preprocessing` rather than inventing stand-in data.
  `scripts/download_bronze_data.sh` fetches the same files from their public
  sources if `git-lfs` isn't available.
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
