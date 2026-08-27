"""Deterministic stub predictor.

Per ROADMAP.md weeks 1-4: backend/frontend are scaffolded against the real
API contracts before a trained Phase 1 checkpoint exists. This stub is
swapped for a real ml/ checkpoint load once Phase 1 training lands - see
glossary.md SS3 for the ml/models/mpnn.py contract it will call into, and
SS4 for the exact response shape this must keep matching.
"""
from __future__ import annotations

import hashlib

# Static MD17 molecule catalog - fixed by the dataset, not a live database.
MD17_MOLECULES: dict[str, int] = {
    "aspirin": 21,
    "benzene": 12,
    "ethanol": 9,
    "malonaldehyde": 9,
    "naphthalene": 18,
    "salicylic_acid": 16,
    "toluene": 15,
    "uracil": 12,
}


class InferenceService:
    phase = "phase1_deterministic"

    def predict(self, molecule: str, frame_index: int) -> tuple[float, float, float | None]:
        num_atoms = MD17_MOLECULES.get(molecule, 10)
        digest = hashlib.sha256(f"{molecule}:{frame_index}".encode()).hexdigest()
        seed = int(digest[:8], 16)

        # Deterministic, plausible-looking placeholders - not a trained model's output.
        predicted_energy = -num_atoms * 2000.0 - (seed % 5000) / 10.0
        predicted_force_rms = 5.0 + (seed % 2000) / 100.0
        uncertainty = None  # Phase 1 is deterministic - no uncertainty by construction.

        return predicted_energy, predicted_force_rms, uncertainty


inference_service = InferenceService()
