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

import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import GOLD_DIR, PHASE_CHECKPOINTS
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


def test_benchmarks_returns_one_row_per_checkpoint_with_a_unique_phase_key():
    response = client.get("/benchmarks")
    assert response.status_code == 200
    rows = response.json()

    phases_present = {row["phase"] for row in rows}
    # Each phase key appears at most once, even though five of these rows
    # share the registry phase "mpnn"/"schnet"/etc. is not what's checked -
    # PHASE_CHECKPOINTS's keys are what must stay unique. The bug this
    # replaced could emit duplicate, ambiguous rows for two runs of the
    # same registry phase (phase2_blip vs phase2_blip_nll, both "blip"),
    # or crash outright reading the wrong JSON keys.
    assert len(phases_present) == len(rows)
    assert phases_present <= set(PHASE_CHECKPOINTS)
    # The full Phase 1 bake-off (5 architectures) plus both BLIP runs plus
    # Phase 3 - every trained checkpoint, not a collapsed subset.
    assert {"phase1_deterministic", "phase1_schnet", "phase1_painn", "phase1_physnet", "phase1_dimenet"} <= phases_present

    for row in rows:
        assert row["energy_mae"] > 0
        assert row["force_mae"] > 0
        assert row["energy_rmse"] >= row["energy_mae"]  # RMSE >= MAE always
        assert row["force_rmse"] >= row["force_mae"]
        if row["phase"].startswith("phase1_"):
            # Every Phase 1 architecture is deterministic - none has a
            # predict_distribution, so none has UQ metrics, not just MPNN.
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


# --------------------------------------------------------------------------
# Training jobs and data upload (train the rest of the molecules; add new
# data through the real bronze -> silver -> gold pipeline)
# --------------------------------------------------------------------------

def test_jobs_phases_lists_every_registered_architecture():
    response = client.get("/jobs/phases")
    assert response.status_code == 200
    assert set(response.json()) == {"mpnn", "schnet", "painn", "physnet", "dimenet", "blip", "graph_stochastic"}


def test_jobs_train_rejects_an_unknown_molecule():
    response = client.post("/jobs/train", json={"molecule": "not_a_real_molecule", "phase": "mpnn"})
    assert response.status_code == 404


def test_jobs_train_rejects_an_unknown_phase():
    response = client.post("/jobs/train", json={"molecule": "aspirin", "phase": "not_a_real_phase"})
    assert response.status_code == 422


def test_jobs_train_rejects_a_mismatched_theory():
    response = client.post("/jobs/train", json={"molecule": "aspirin", "phase": "mpnn", "theory": "dft"})
    assert response.status_code == 422
    assert "ccsd" in response.json()["detail"]


@pytest.mark.skipif(not (GOLD_DIR / "aspirin_ccsd.npz").exists(), reason="aspirin gold data not present")
def test_launching_and_predicting_against_a_real_training_job():
    """End to end: launch a real (tiny, 2-epoch) subprocess training job on
    a molecule that has no fixed checkpoint, poll it to completion, then
    predict against its own checkpoint via phase="job:<id>" - the actual
    mechanism this feature is for, not a mock of it."""
    response = client.post("/jobs/train", json={"molecule": "aspirin", "phase": "mpnn", "epochs": 2})
    assert response.status_code == 200
    job = response.json()
    assert job["status"] == "running"
    job_id = job["id"]
    config_name = job["config_name"]

    try:
        deadline = time.time() + 90
        status = job
        while time.time() < deadline:
            status = client.get(f"/jobs/{job_id}").json()
            if status["status"] != "running":
                break
            time.sleep(2)
        assert status["status"] == "complete", f"job did not complete in time: {status}"
        assert status["checkpoint_ready"] is True
        assert status["epochs_run"] == 2

        predict_response = client.post(
            "/predictions", json={"molecule": "aspirin", "frame_index": 0, "phase": f"job:{job_id}"}
        )
        assert predict_response.status_code == 200
        body = predict_response.json()
        assert body["phase"] == f"job:{job_id}"
        assert body["true_energy"] is not None

        # A job trained on aspirin must refuse to predict a different molecule.
        mismatch = client.post(
            "/predictions", json={"molecule": "ethanol", "frame_index": 0, "phase": f"job:{job_id}"}
        )
        assert mismatch.status_code == 422
    finally:
        # Real training artifacts this test produced - clean up so repeat
        # runs don't accumulate fake jobs that GET /jobs (and a server
        # restart's recover_jobs) would then show during a real demo.
        from app.core.config import CHECKPOINTS_DIR, JOB_CONFIGS_DIR, JOB_LOGS_DIR, RESULTS_DIR, RUNS_DIR

        import shutil

        (JOB_CONFIGS_DIR / f"{config_name}.yaml").unlink(missing_ok=True)
        (CHECKPOINTS_DIR / f"{config_name}.pt").unlink(missing_ok=True)
        (JOB_LOGS_DIR / f"{config_name}.log").unlink(missing_ok=True)
        (RESULTS_DIR / f"{config_name}.json").unlink(missing_ok=True)
        shutil.rmtree(RUNS_DIR / config_name, ignore_errors=True)


def test_predict_rejects_an_unknown_job_id():
    response = client.post(
        "/predictions", json={"molecule": "ethanol", "frame_index": 0, "phase": "job:does_not_exist"}
    )
    assert response.status_code == 404


def _synthetic_md17_npz(molecule: str, theory: str, n_atoms: int = 3, n_configs: int = 20) -> bytes:
    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    np.savez(
        buf,
        z=np.array([1, 6, 8][:n_atoms], dtype=np.int64),
        R=rng.normal(size=(n_configs, n_atoms, 3)).astype(np.float32),
        E=rng.normal(loc=-1000.0, size=n_configs).astype(np.float32),
        F=rng.normal(size=(n_configs, n_atoms, 3)).astype(np.float32),
        name=molecule,
    )
    buf.seek(0)
    return buf.read()


def test_data_upload_runs_the_real_bronze_to_gold_pipeline_and_appears_in_molecules():
    molecule, theory = "zzz_test_molecule", "dft"
    npz_bytes = _synthetic_md17_npz(molecule, theory)

    response = client.post(
        "/data/upload",
        files={"file": (f"{molecule}.npz", npz_bytes, "application/octet-stream")},
        data={"molecule": molecule, "theory": theory},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["molecule"] == molecule
    assert body["theory"] == theory
    assert body["n_train"] + body["n_val"] + body["n_test"] == 20
    assert body["used_literature_split"] is False

    try:
        catalog = {m["molecule"]: m["num_atoms"] for m in client.get("/molecules").json()}
        assert catalog.get(molecule) == 3
    finally:
        # Test-only data; clean up rather than leaving a permanent fake
        # molecule in the shared gold/bronze/silver directories.
        from app.core.config import BRONZE_DIR, GOLD_DIR, SILVER_DIR

        (GOLD_DIR / f"{molecule}_{theory}.npz").unlink(missing_ok=True)
        (SILVER_DIR / f"{molecule}_{theory}.npz").unlink(missing_ok=True)
        for leftover in BRONZE_DIR.glob(f"*{molecule}*"):
            leftover.unlink(missing_ok=True)

        from app.services.inference import refresh_catalog

        refresh_catalog()


def test_data_upload_rejects_an_unsupported_file_type():
    response = client.post(
        "/data/upload",
        files={"file": ("data.txt", b"not a real dataset", "text/plain")},
    )
    assert response.status_code == 422
