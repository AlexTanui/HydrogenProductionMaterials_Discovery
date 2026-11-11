# API — implementation notes

Expanded version of [glossary.md §4](../glossary.md#4-api-contracts),
kept in sync as endpoints are built. The contract itself (request/response
shapes) lives there; this file is the "how it actually works" detail
behind it. Base URL (dev): `http://localhost:8000`, proxied by the
frontend dev server as `/api/*` (`frontend/vite.config.ts`).

## Status

All three endpoints are implemented against real data and real trained
checkpoints — not stubs. `POST /predictions` only has real inference for
`ethanol` (DFT); every other catalog entry has real MD17 data but no
trained model, and is refused rather than guessed (see below).

## Running it

```bash
cd backend && uvicorn app.main:app --reload   # http://localhost:8000
cd frontend && npm run dev                     # http://localhost:5173
```

No extra setup needed to import `ml.*` from inside `backend/` — see
`app/core/config.py`'s `sys.path` note. Checkpoints are gitignored, not
committed: train them first if `experiments/checkpoints/` is empty —
`python -m ml.training.train experiments/configs/<config>.yaml` for each
of `phase1_baseline`, `phase2_blip_nll`, `phase3_graph_stochastic`. The
server starts fine with some or all checkpoints missing — it warm-loads
each one at startup and prints a clear per-phase line (loaded, or why
not) rather than crashing; the corresponding phase just 503s until it
exists.

## `POST /predictions`

Implementation: `app/services/inference.py` (the model/data logic),
`app/api/predictions.py` (the route, which also logs every call to
`PredictionLog` — `app/models/db.py`).

- **Checkpoint selection**: `app.core.config.PHASE_CHECKPOINTS` maps each
  of the three canonical phase names to one checkpoint file. Loaded once
  per phase (`functools.lru_cache`) and reused across requests — a 20-MC-sample
  stochastic prediction on a 9-atom molecule takes tens of
  milliseconds, not a reload per call.
- **Data**: `frame_index` indexes `MD17Dataset(gold_path, split="test", ...)`
  — the held-out test split, not the raw trajectory — so a live
  prediction is always on a frame the model never trained on. The
  response's `true_energy`/`true_force_rms` come from that same dataset
  object, not a separate lookup, so they can never disagree with what was
  actually scored.
- **Deterministic vs. stochastic**: `ScaledEnergyModel.supports_distribution`
  (set by whether the wrapped model defines `predict_distribution`)
  decides the branch — Phase 1 (MPNN) calls `predict_energy_and_forces`
  once; Phase 2/3 (BLIP, GraphStochastic) call `predict_distribution`
  (20 MC samples) and report `uncertainty` as the predicted energy
  standard deviation.
- **Untrained molecules refused, not guessed**: `TRAINED_MOLECULE = "ethanol"`
  in `app/core/config.py`. Running an ethanol-trained model on, say,
  aspirin would still execute (the architecture doesn't care how many
  atoms there are) and produce a plausible-looking number — exactly the
  kind of silently-wrong result this project's whole Phase 2 calibration
  work has been about avoiding. A 503 names the molecule and why.

| Failure | Status | When |
|---|---|---|
| Unknown molecule | 404 | not in `GET /molecules`'s catalog at all |
| Molecule has data but no trained checkpoint | 503 | any molecule other than `ethanol` right now |
| `frame_index` out of range | 422 | outside `[0, n_test_frames - 1]` |
| Unknown `phase` | 422 | not one of the three canonical names or three aliases |
| Checkpoint file missing | 503 | trained config never run, or run on another machine |

## `GET /molecules`

Implementation: `app/services/inference.py::_scan_gold_catalog`, called
once at import time. Reads every `data/gold/md17/*.npz`'s own `molecule`/
`theory`/`z` fields — never a hardcoded list — so it can't drift from
what's actually on disk. When `ethanol` has both `dft` and `ccsd_t` gold
files, `dft` wins (the theory every trained checkpoint uses); this
endpoint has no theory field to disambiguate further.

## `GET /benchmarks`

Implementation: `app/api/benchmarks.py`. Reads exactly the three files
`PHASE_CHECKPOINTS` names (`experiments/results/<checkpoint-stem>.json`),
one per project phase, and returns each one's `benchmark_row` unchanged —
the same row `ml/training/evaluate.py` computes and the ML harness's own
benchmark table uses, not a second implementation of the same metrics.
A phase whose checkpoint has never been trained/scored is simply absent
from the response.

Why one row per *project phase* rather than one per `experiments/runs/`
entry or per Phase 1 bake-off architecture: a results file's own `phase`
field is the model *registry* key (`ml/models/registry.py`'s `"blip"`,
`"mpnn"`, etc.), which is the same for two different runs of the same
model (`phase2_blip` and `phase2_blip_nll` are both `"blip"`) — reading
that field directly would produce duplicate, ambiguous rows for a reader
or a chart. `PHASE_CHECKPOINTS`'s keys are the three unambiguous labels
this endpoint (and `POST /predictions`) actually use.

## Known gaps

- Phase 2's served checkpoint (`phase2_blip_nll.pt`) is numerically stable
  but not well-calibrated — see `docs/model_cards/blip.md` (once written)
  and `docs/phase2_remaining_work.md`. The live demo shows this honestly:
  Phase 2's reported `uncertainty` is real, just not yet trustworthy.
- No auth (by design for this stage — glossary.md §4).
- `backend/app.db` (SQLite) is fine for local dev, not concurrent writers
  — CLAUDE.md's own note on `app/models/db.py`.
