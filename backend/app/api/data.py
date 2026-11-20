from __future__ import annotations

from fastapi import APIRouter, Depends, Form, UploadFile

from app.core.read_only import require_writable

from app.models.schemas import DataUploadResponse
from app.services import data_pipeline

router = APIRouter(prefix="/data", tags=["data"])


@router.post("/upload", response_model=DataUploadResponse, dependencies=[Depends(require_writable)])
async def upload(
    file: UploadFile,
    molecule: str | None = Form(default=None),
    theory: str | None = Form(default=None),
) -> DataUploadResponse:
    """Runs a new raw MD17-format file through the real bronze -> silver
    -> gold pipeline (ml/data/preprocessing.py) - same validation, same
    split logic, as the nine built-in sources. `molecule`/`theory`
    override what's read from the file itself; required for a .npz that
    doesn't carry them.
    """
    result = await data_pipeline.ingest_upload(file, molecule, theory)
    return DataUploadResponse(**result)
