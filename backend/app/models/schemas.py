from __future__ import annotations

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    molecule: str
    frame_index: int = 0


class PredictionResponse(BaseModel):
    molecule: str
    frame_index: int
    predicted_energy: float
    predicted_force_rms: float
    uncertainty: float | None
    phase: str


class MoleculeOut(BaseModel):
    molecule: str
    num_atoms: int


class BenchmarkResult(BaseModel):
    phase: str
    model: str
    energy_mae: float
    force_mae: float
    ece: float | None = None
    uncertainty_correlation: float | None = None
