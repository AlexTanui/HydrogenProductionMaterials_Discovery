from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.models.db import PredictionLog, get_db
from app.models.schemas import PredictionRequest, PredictionResponse
from app.services.inference import inference_service

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("", response_model=PredictionResponse)
def predict(request: PredictionRequest, db: Session = Depends(get_db)) -> PredictionResponse:
    result = inference_service.predict(request.molecule, request.frame_index, request.phase)

    db.add(
        PredictionLog(
            molecule=request.molecule,
            frame_index=request.frame_index,
            phase=result["phase"],
            predicted_energy=result["predicted_energy"],
            predicted_force_rms=result["predicted_force_rms"],
            uncertainty=result["uncertainty"],
        )
    )
    db.commit()

    return PredictionResponse(molecule=request.molecule, frame_index=request.frame_index, **result)
