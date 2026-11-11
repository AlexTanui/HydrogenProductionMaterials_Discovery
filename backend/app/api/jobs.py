from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import TrainJobRequest, TrainJobStatus
from app.services import jobs as jobs_service

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/phases")
def phases() -> list[str]:
    """The registry phases (ml/models/registry.py keys) a job can train -
    not the same list as /predictions' three canonical phase names, since
    a job trains one specific architecture, not a project-level phase."""
    return jobs_service.list_phases()


@router.post("/train", response_model=TrainJobStatus)
def start_training(request: TrainJobRequest) -> TrainJobStatus:
    job = jobs_service.start_job(request.molecule, request.phase, request.theory, request.epochs)
    return TrainJobStatus(**jobs_service.job_status(job))


@router.get("", response_model=list[TrainJobStatus])
def list_jobs() -> list[TrainJobStatus]:
    return [TrainJobStatus(**s) for s in jobs_service.list_jobs()]


@router.get("/{job_id}", response_model=TrainJobStatus)
def get_job(job_id: str) -> TrainJobStatus:
    job = jobs_service.get_job(job_id)
    return TrainJobStatus(**jobs_service.job_status(job))
