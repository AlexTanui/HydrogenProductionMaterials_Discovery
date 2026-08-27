from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.models.db import PredictionLog, get_db
from app.models.schemas import PredictionRequest, PredictionResponse
from app.services.inference import inference_service

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("", response_model=PredictionResponse)
def predict(request: PredictionRequest, db: Session = Depends(get_db)) -> PredictionResponse:
    energy, force_rms, uncertainty = inference_service.predict(request.molecule, request.frame_index)

    db.add(
        PredictionLog(
            molecule=request.molecule,
            frame_index=request.frame_index,
            phase=inference_service.phase,
            predicted_energy=energy,
            predicted_force_rms=force_rms,
            uncertainty=uncertainty,
        )
    )
    db.commit()

    return PredictionResponse(
        molecule=request.molecule,
        frame_index=request.frame_index,
        predicted_energy=energy,
        predicted_force_rms=force_rms,
        uncertainty=uncertainty,
        phase=inference_service.phase,
    )
