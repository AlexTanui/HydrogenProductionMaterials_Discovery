"""API tests against the real ethanol/DFT checkpoints (Phase 1-3).

Writes to the shared dev SQLite DB (app/models/db.py) - fine for local
dev per CLAUDE.md's own note on that file; these are write-only
PredictionLog rows, nothing reads them back.

Skips outright if no checkpoint has been trained yet (fresh clone,
`experiments/checkpoints/` empty - it's gitignored) rather than failing:
these are integration tests against real trained artifacts, not something
a fresh checkout can satisfy on its own.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import PHASE_CHECKPOINTS
from app.main import app

ANY_CHECKPOINT_TRAINED = any(path.exists() for path in PHASE_CHECKPOINTS.values())
pytestmark = pytest.mark.skipif(
    not ANY_CHECKPOINT_TRAINED,
    reason="no trained checkpoint in experiments/checkpoints/ - train one first",
)

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_molecules_catalog_is_read_from_the_real_gold_files():
    response = client.get("/molecules")
    assert response.status_code == 200
    by_name = {m["molecule"]: m["num_atoms"] for m in response.json()}

    # ethanol is the only molecule with a trained checkpoint - it must be here.
    assert by_name["ethanol"] == 9
    # These two have real gold files but the old stub never listed them.
    assert "azobenzene" in by_name
    assert "paracetamol" in by_name
    # These two were in the old stub's hardcoded list but have no gold file.
    assert "naphthalene" not in by_name
    assert "salicylic_acid" not in by_name


@pytest.mark.parametrize(
    "phase,expect_uncertainty",
    [
        (None, False),  # default
        ("phase1_deterministic", False),
        ("phase1", False),
        ("phase2_blip_k1", True),
        ("phase2_blip", True),
        ("phase2", True),
        ("phase3_graph_stochastic", True),
        ("phase3", True),
    ],
)
def test_predict_ethanol_against_every_trained_phase(phase, expect_uncertainty):
    checkpoint = PHASE_CHECKPOINTS[
        {"phase1": "phase1_deterministic", "phase2": "phase2_blip", "phase3": "phase3_graph_stochastic"}.get(
            phase, phase or "phase1_deterministic"
        )
    ]
    if not checkpoint.exists():
        pytest.skip(f"{checkpoint} not trained")

    body = {"molecule": "ethanol", "frame_index": 0}
    if phase is not None:
        body["phase"] = phase
    response = client.post("/predictions", json=body)

    assert response.status_code == 200
    data = response.json()
    assert data["molecule"] == "ethanol"
    assert data["frame_index"] == 0
    assert (data["uncertainty"] is not None) == expect_uncertainty

    # Real DFT ground truth, not predicted - and the prediction should be
    # within a sane range of it (loose bound: this is a correctness check
    # on the plumbing, not a re-assertion of each model's own MAE).
    assert data["true_energy"] is not None
    assert data["true_force_rms"] is not None
    assert abs(data["predicted_energy"] - data["true_energy"]) < 50.0
    assert abs(data["predicted_force_rms"] - data["true_force_rms"]) < 50.0


def test_predict_rejects_a_molecule_with_no_trained_checkpoint():
    response = client.post("/predictions", json={"molecule": "aspirin", "frame_index": 0})
    assert response.status_code == 503
    assert "aspirin" in response.json()["detail"]
    assert "ethanol" in response.json()["detail"]


def test_predict_rejects_an_unknown_molecule():
    response = client.post("/predictions", json={"molecule": "not_a_real_molecule", "frame_index": 0})
    assert response.status_code == 404


def test_predict_rejects_an_out_of_range_frame_index():
    response = client.post("/predictions", json={"molecule": "ethanol", "frame_index": 10_000_000})
    assert response.status_code == 422
    assert "frame_index" in response.json()["detail"]


def test_predict_rejects_an_unknown_phase():
    response = client.post("/predictions", json={"molecule": "ethanol", "frame_index": 0, "phase": "phase99"})
    assert response.status_code == 422
    assert "phase99" in response.json()["detail"]


def test_benchmarks_returns_one_row_per_project_phase_not_per_architecture():
    response = client.get("/benchmarks")
    assert response.status_code == 200
    rows = response.json()

    phases_present = {row["phase"] for row in rows}
    # Each phase appears at most once - the bug this replaced could emit
    # five "phase1_deterministic"-ish rows (one per bake-off architecture)
    # or crash outright reading the wrong JSON keys.
    assert len(phases_present) == len(rows)
    assert phases_present <= set(PHASE_CHECKPOINTS)

    for row in rows:
        assert row["energy_mae"] > 0
        assert row["force_mae"] > 0
        assert row["energy_rmse"] >= row["energy_mae"]  # RMSE >= MAE always
        assert row["force_rmse"] >= row["force_mae"]
        if row["phase"] == "phase1_deterministic":
            assert row["ece"] is None
            assert row["uncertainty_correlation"] is None
            assert row["force_ece"] is None
            assert row["energy_sharpness"] is None
        else:
            # Phase 2/3 have real (if poorly calibrated) UQ metrics once scored -
            # for both energy and force, not just the energy summary pair.
            assert row["ece"] is not None and row["ece"] >= 0
            assert row["force_ece"] is not None and row["force_ece"] >= 0
            assert row["energy_coverage_90"] is not None
            assert row["energy_sharpness"] is not None and row["energy_sharpness"] > 0
