"""Upload a new raw MD17-format source file and run it through the real
bronze -> silver -> gold pipeline (ml/data/preprocessing.py) - no new
validation logic here, just the file-handling around calling
bronze_to_silver/silver_to_gold directly, the same two functions
ml.data.preprocessing.run_all calls for the nine built-in sources.

Fast and synchronous: this is array validation and re-splitting, seconds
not hours, unlike training - no background job needed for this part.
"""
from __future__ import annotations

import uuid

import numpy as np
from fastapi import HTTPException, UploadFile

from app.core.config import BRONZE_DIR, GOLD_DIR, SILVER_DIR
from app.services import inference as inference_module

from ml.data.preprocessing import bronze_to_silver, silver_to_gold

_ALLOWED_SUFFIXES = (".npz", ".zip")
# Generous but bounded: real MD17 .npz/.zip sources run tens to hundreds
# of MB (data/bronze/md17 is ~814MB for all nine combined); this stops an
# accidental multi-GB upload from filling the disk, not a determined
# attacker - there is no auth on this API at all (glossary.md SS4).
_MAX_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GiB


def _safe_filename(name: str) -> str:
    """Rejects a path, keeps just the final component - an uploaded
    filename is attacker-controlled input; '../../x' must not escape
    data/bronze/md17/."""
    cleaned = name.replace("\\", "/").rsplit("/", 1)[-1]
    if not cleaned or cleaned in (".", ".."):
        raise HTTPException(422, f"invalid filename {name!r}")
    return cleaned


async def ingest_upload(file: UploadFile, molecule: str | None, theory: str | None) -> dict:
    filename = _safe_filename(file.filename or "")
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            422,
            f"unsupported file type {suffix!r} for {filename!r}; expected one of {_ALLOWED_SUFFIXES} "
            "- the same two raw MD17 source shapes ml/data/preprocessing.py already knows how to read "
            "(.npz for a single split, .zip for a shipped literature train/test split)",
        )

    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    # A unique bronze filename per upload, so re-uploading under the same
    # original name (e.g. retrying after a fixed export) never clobbers a
    # previous bronze file silently - the molecule_theory.npz name is what
    # matters downstream (silver/gold), not this one.
    bronze_path = BRONZE_DIR / f"{uuid.uuid4().hex[:10]}_{filename}"

    size = 0
    with bronze_path.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > _MAX_UPLOAD_BYTES:
                out.close()
                bronze_path.unlink(missing_ok=True)
                raise HTTPException(413, f"upload exceeds {_MAX_UPLOAD_BYTES // (1024*1024)} MiB limit")
            out.write(chunk)

    try:
        silver_path = bronze_to_silver(bronze_path, SILVER_DIR, molecule=molecule, theory=theory)
    except Exception as exc:
        # The real integrity gate (NaN/shape/unit checks, ml/data/preprocessing.py
        # ::validate_sample) rejecting bad data - surface its message
        # directly rather than a generic 500; this is meant to be seen.
        bronze_path.unlink(missing_ok=True)
        raise HTTPException(422, f"{filename}: failed validation - {exc}") from exc

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    gold_molecule = molecule
    gold_theory = theory
    if gold_molecule is None or gold_theory is None:
        with np.load(silver_path, allow_pickle=True) as data:
            gold_molecule = gold_molecule or str(data["molecule"])
            gold_theory = gold_theory or str(data["theory"])
    gold_path = silver_to_gold(silver_path, GOLD_DIR / f"{gold_molecule}_{gold_theory}.npz")

    with np.load(gold_path, allow_pickle=True) as data:
        n_train, n_val, n_test = len(data["train_idx"]), len(data["val_idx"]), len(data["test_idx"])
        used_literature_split = bool(data["used_literature_split"]) if "used_literature_split" in data.files else False

    # So GET /molecules (and job launches) see this molecule immediately,
    # no server restart - see that function's own docstring for why this
    # has to mutate the catalog in place rather than just re-running the
    # scan locally.
    inference_module.refresh_catalog()

    return {
        "molecule": gold_molecule,
        "theory": gold_theory,
        "gold_path": str(gold_path),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "used_literature_split": used_literature_split,
    }
