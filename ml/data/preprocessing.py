"""Bronze -> silver -> gold promotion for MD17 (see glossary.md sections 3 and 5).

Silver: dedupe, unify format, validate (shape/NaN/unit checks).
Gold: apply the trajectory-block split, or preserve a shipped
literature-standard split where one exists (aspirin, malonaldehyde,
toluene, ccsd_t-ethanol) - never both, never random.

Migrated from rajcleaning.ipynb (Raj's original data-cleaning pass);
see git history for what changed and why. Run directly:

    python -m ml.data.preprocessing
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ml.data.md17 import MD17Sample, load_npz, load_split_zip

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BRONZE_DIR = ROOT_DIR / "data" / "bronze" / "md17"
SILVER_DIR = ROOT_DIR / "data" / "silver" / "md17"
GOLD_DIR = ROOT_DIR / "data" / "gold" / "md17"

_EXPECTED_R_UNIT = "Ang"
_EXPECTED_E_UNIT = "kcal/mol"

# One canonical source file per molecule+theory level - matches
# scripts/download_bronze_data.sh and glossary.md's inventory. The bool
# flags whether this source ships a literature-standard train/test split
# that should be preserved rather than re-split at the gold step.
MD17_SOURCES: list[tuple[str, str, str, bool]] = [
    ("md17_ethanol.npz", "ethanol", "dft", False),
    ("ethanol_ccsd_t.zip", "ethanol", "ccsd_t", True),
    ("benzene2018_dft.npz", "benzene", "dft", False),
    ("azobenzene_dft.npz", "azobenzene", "dft", False),
    ("paracetamol_dft.npz", "paracetamol", "dft", False),
    ("aspirin_ccsd.zip", "aspirin", "ccsd", True),
    ("malonaldehyde_ccsd_t.zip", "malonaldehyde", "ccsd_t", True),
    ("toluene_ccsd_t.zip", "toluene", "ccsd_t", True),
    ("md17_uracil.npz", "uracil", "dft", False),
]


def validate_sample(sample: MD17Sample) -> list[str]:
    problems = []
    if np.isnan(sample.R).any():
        problems.append("NaN values in R")
    if np.isnan(sample.E).any():
        problems.append("NaN values in E")
    if np.isnan(sample.F).any():
        problems.append("NaN values in F")
    if sample.n_atoms == 0:
        problems.append("zero atoms")
    if sample.n_configs == 0:
        problems.append("zero configurations")
    if sample.r_unit != _EXPECTED_R_UNIT:
        problems.append(f"unexpected r_unit {sample.r_unit!r}, expected {_EXPECTED_R_UNIT!r}")
    if sample.e_unit != _EXPECTED_E_UNIT:
        problems.append(f"unexpected e_unit {sample.e_unit!r}, expected {_EXPECTED_E_UNIT!r}")
    return problems


def bronze_to_silver(bronze_path: str | Path, silver_dir: str | Path,
                      molecule: Optional[str] = None, theory: Optional[str] = None) -> Path:
    bronze_path = Path(bronze_path)
    silver_dir = Path(silver_dir)
    silver_dir.mkdir(parents=True, exist_ok=True)

    if bronze_path.suffix == ".npz":
        sample = load_npz(bronze_path, molecule=molecule, theory=theory)
        problems = validate_sample(sample)
        if problems:
            raise ValueError(f"{bronze_path.name}: failed validation: {problems}")
        literature_test_mask = np.zeros(sample.n_configs, dtype=bool)
        has_literature_split = False

    elif bronze_path.suffix == ".zip":
        splits = load_split_zip(bronze_path, molecule=molecule, theory=theory)
        train, test = splits["train"], splits["test"]
        for name, s in (("train", train), ("test", test)):
            problems = validate_sample(s)
            if problems:
                raise ValueError(f"{bronze_path.name} [{name}]: failed validation: {problems}")
        sample = MD17Sample(
            molecule=train.molecule, theory=train.theory, z=train.z,
            R=np.concatenate([train.R, test.R], axis=0),
            E=np.concatenate([train.E, test.E], axis=0),
            F=np.concatenate([train.F, test.F], axis=0),
        )
        literature_test_mask = np.concatenate([np.zeros(train.n_configs, dtype=bool), np.ones(test.n_configs, dtype=bool)])
        has_literature_split = True
    else:
        raise ValueError(f"Unsupported bronze file type: {bronze_path.suffix}")

    out_path = silver_dir / f"{sample.molecule}_{sample.theory}.npz"
    np.savez_compressed(
        out_path, z=sample.z, R=sample.R, E=sample.E, F=sample.F,
        molecule=sample.molecule, theory=sample.theory,
        has_literature_split=has_literature_split, literature_test_mask=literature_test_mask,
    )
    return out_path


def trajectory_block_split(n_configs: int, train: float = 0.8, val: float = 0.1, test: float = 0.1):
    assert abs(train + val + test - 1.0) < 1e-6, "splits must sum to 1.0"
    n_train = int(round(n_configs * train))
    n_val = int(round(n_configs * val))
    idx = np.arange(n_configs)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def silver_to_gold(silver_path: str | Path, gold_path: str | Path,
                    train: float = 0.8, val: float = 0.1, test: float = 0.1,
                    use_literature_split: Optional[bool] = None) -> Path:
    silver_path = Path(silver_path)
    gold_path = Path(gold_path)
    gold_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(silver_path, allow_pickle=True)
    z, R, E, F = data["z"], data["R"], data["E"], data["F"]
    n_configs = R.shape[0]

    has_lit_split = bool(data["has_literature_split"]) if "has_literature_split" in data.files else False
    if use_literature_split is None:
        use_literature_split = has_lit_split

    if use_literature_split and has_lit_split:
        test_mask = data["literature_test_mask"]
        test_idx = np.where(test_mask)[0]
        remaining = np.where(~test_mask)[0]
        n_train_of_remaining = int(round(len(remaining) * (train / (train + val))))
        train_idx = remaining[:n_train_of_remaining]
        val_idx = remaining[n_train_of_remaining:]
    else:
        train_idx, val_idx, test_idx = trajectory_block_split(n_configs, train, val, test)

    assert set(train_idx).isdisjoint(test_idx), "train/test leakage detected"
    assert set(val_idx).isdisjoint(test_idx), "val/test leakage detected"
    assert set(train_idx).isdisjoint(val_idx), "train/val leakage detected"

    np.savez_compressed(
        gold_path, z=z, R=R, E=E, F=F,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        molecule=str(data["molecule"]), theory=str(data["theory"]),
        used_literature_split=bool(use_literature_split and has_lit_split),
    )
    return gold_path


def run_all(bronze_dir: Path = BRONZE_DIR, silver_dir: Path = SILVER_DIR, gold_dir: Path = GOLD_DIR) -> list[tuple[str, str, Path]]:
    """Runs bronze -> silver -> gold for every known MD17 source. Returns
    (molecule, theory, gold_path) for each one that succeeded."""
    if not bronze_dir.exists():
        print(f"No such directory: {bronze_dir}")
        return []

    silver_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBronze dir: {bronze_dir}")
    print(f"{'Source':<28} {'Molecule':<14} {'Theory':<8} {'Status'}")
    print("-" * 78)

    results = []
    for filename, molecule, theory, use_lit_split in MD17_SOURCES:
        bronze_path = bronze_dir / filename
        if not bronze_path.exists():
            print(f"{filename:<28} {molecule:<14} {theory:<8} MISSING (skip)")
            continue
        try:
            silver_path = bronze_to_silver(bronze_path, silver_dir, molecule=molecule, theory=theory)
            gold_path = silver_to_gold(silver_path, gold_dir / f"{molecule}_{theory}.npz", use_literature_split=use_lit_split)
            n_configs = int(np.load(gold_path, allow_pickle=True)["train_idx"].shape[0])
            print(f"{filename:<28} {molecule:<14} {theory:<8} OK -> {gold_path.name} ({n_configs} train configs)")
            results.append((molecule, theory, gold_path))
        except Exception as e:
            print(f"{filename:<28} {molecule:<14} {theory:<8} FAILED: {type(e).__name__}: {e}")

    print("-" * 78)
    print(f"{len(results)}/{len(MD17_SOURCES)} MD17 datasets processed into {gold_dir}")
    return results


if __name__ == "__main__":
    run_all()
