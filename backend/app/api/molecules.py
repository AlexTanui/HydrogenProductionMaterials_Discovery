from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import MoleculeOut
from app.services.inference import MD17_MOLECULES

router = APIRouter(prefix="/molecules", tags=["molecules"])


@router.get("", response_model=list[MoleculeOut])
def list_molecules() -> list[MoleculeOut]:
    return [MoleculeOut(molecule=name, num_atoms=n) for name, n in MD17_MOLECULES.items()]
