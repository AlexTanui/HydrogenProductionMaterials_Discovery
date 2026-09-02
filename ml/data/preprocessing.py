"""Bronze -> silver -> gold promotion for MD17 and MD22 (see glossary.md sections 3 and 5).

Silver: dedupe, unify format, validate (shape/NaN/unit checks).
Gold: apply the trajectory-block split, or preserve a shipped
literature-standard split where one exists (aspirin, malonaldehyde,
toluene, ccsd_t-ethanol) - never both, never random.

MD17 and MD22 run through the same code path but stay in separate
data/{silver,gold}/{md17,md22}/ trees - the two differ by more than an
order of magnitude in both system size (9-24 atoms vs 56-370) and total
energy magnitude, and must never be pooled, split, or benchmarked as one
dataset. No MD22 source ships a literature-standard split, so every MD22
molecule takes the trajectory-block path.

Migrated from rajcleaning.ipynb (Raj's original data-cleaning pass);
see git history for what changed and why. Run directly:

    python -m ml.data.preprocessing                 # both datasets
    python -m ml.data.preprocessing --dataset md22  # just one
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ml.data.md17 import MD17Sample, load_npz, load_split_zip

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"

# Kept as module-level names for backwards compatibility - these are the
# MD17 paths, which is what they always pointed at.
BRONZE_DIR = DATA_DIR / "bronze" / "md17"
SILVER_DIR = DATA_DIR / "silver" / "md17"
GOLD_DIR = DATA_DIR / "gold" / "md17"

_EXPECTED_R_UNIT = "Ang"
_EXPECTED_E_UNIT = "kcal/mol"

# Smallest plausible real data file. Cloning without the git-lfs client
# populates these exact paths with ~130-byte pointer text files instead of
# real arrays (see README "Getting the data"); catching that here gives a
# usable message instead of a confusing zip/pickle error deep in numpy.
_LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec"
_MIN_REAL_FILE_BYTES = 1024

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

# MD22's four sources, all plain .npz, all DFT, none shipping a
# literature-standard split - MD22 papers quote a train-set *size* per
# molecule rather than distributing a fixed index split, so there is
# nothing to preserve and every one of these takes the trajectory-block
# path. The .zip duplicates of AT-AT-CG-CG and the nanotube are
# documented-redundant and deliberately not tracked (see .gitignore).
MD22_SOURCES: list[tuple[str, str, str, bool]] = [
    ("md22_AT-AT-CG-CG.npz", "AT-AT-CG-CG", "dft", False),
    ("md22_DHA.npz", "DHA", "dft", False),
    ("md22_buckyball-catcher.npz", "buckyball-catcher", "dft", False),
    ("md22_double-walled_nanotube.npz", "double-walled-nanotube", "dft", False),
]


@dataclass(frozen=True)
class DatasetSpec:
    """Where one dataset's files live and what sources belong to it."""

    name: str
    sources: list[tuple[str, str, str, bool]]

    @property
    def bronze_dir(self) -> Path:
        return DATA_DIR / "bronze" / self.name

    @property
    def silver_dir(self) -> Path:
        return DATA_DIR / "silver" / self.name

    @property
    def gold_dir(self) -> Path:
        return DATA_DIR / "gold" / self.name


DATASETS: dict[str, DatasetSpec] = {
    "md17": DatasetSpec("md17", MD17_SOURCES),
    "md22": DatasetSpec("md22", MD22_SOURCES),
}


def check_bronze_file(path: Path) -> None:
    """Raises if `path` isn't real data - missing, or a git-lfs pointer."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    if path.stat().st_size < _MIN_REAL_FILE_BYTES:
        with open(path, "rb") as f:
            head = f.read(len(_LFS_POINTER_MAGIC))
        if head == _LFS_POINTER_MAGIC:
            raise ValueError(
                f"{path.name} is a git-lfs pointer, not real data - this clone was made "
                "without the git-lfs client. Install it (`git lfs install`) and `git lfs pull`, "
                "or run scripts/download_bronze_data.sh to fetch from the public sources."
            )
        raise ValueError(f"{path.name} is only {path.stat().st_size} bytes - truncated download?")


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
    check_bronze_file(bronze_path)
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


def run_all(bronze_dir: Optional[Path] = None, silver_dir: Optional[Path] = None,
            gold_dir: Optional[Path] = None,
            sources: Optional[list[tuple[str, str, str, bool]]] = None,
            label: str = "MD17") -> list[tuple[str, str, Path]]:
    """Runs bronze -> silver -> gold for every source in `sources`. Returns
    (molecule, theory, gold_path) for each one that succeeded.

    Defaults are the MD17 paths and roster, so existing callers and the
    documented `run_all()` behaviour are unchanged. `run_dataset("md22")`
    is the way to reach MD22."""
    bronze_dir = BRONZE_DIR if bronze_dir is None else Path(bronze_dir)
    silver_dir = SILVER_DIR if silver_dir is None else Path(silver_dir)
    gold_dir = GOLD_DIR if gold_dir is None else Path(gold_dir)
    sources = MD17_SOURCES if sources is None else sources

    if not bronze_dir.exists():
        print(f"No such directory: {bronze_dir}")
        return []

    silver_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBronze dir: {bronze_dir}")
    print(f"{'Source':<32} {'Molecule':<24} {'Theory':<8} {'Status'}")
    print("-" * 100)

    results = []
    for filename, molecule, theory, use_lit_split in sources:
        bronze_path = bronze_dir / filename
        prefix = f"{filename:<32} {molecule:<24} {theory:<8}"
        if not bronze_path.exists():
            print(f"{prefix} MISSING (skip)")
            continue
        try:
            silver_path = bronze_to_silver(bronze_path, silver_dir, molecule=molecule, theory=theory)
            gold_path = silver_to_gold(silver_path, gold_dir / f"{molecule}_{theory}.npz", use_literature_split=use_lit_split)
            gold = np.load(gold_path, allow_pickle=True)
            n_train = int(gold["train_idx"].shape[0])
            n_atoms = int(gold["z"].shape[0])
            split_kind = "literature" if bool(gold["used_literature_split"]) else "block"
            print(f"{prefix} OK -> {gold_path.name} ({n_atoms} atoms, {n_train} train configs, {split_kind} split)")
            results.append((molecule, theory, gold_path))
        except Exception as e:
            print(f"{prefix} FAILED: {type(e).__name__}: {e}")

    print("-" * 100)
    print(f"{len(results)}/{len(sources)} {label} datasets processed into {gold_dir}")
    return results


def run_dataset(dataset: str) -> list[tuple[str, str, Path]]:
    """Runs the full pipeline for one named dataset ('md17' or 'md22')."""
    try:
        spec = DATASETS[dataset]
    except KeyError:
        raise ValueError(f"unknown dataset {dataset!r}, expected one of {sorted(DATASETS)}") from None
    return run_all(spec.bronze_dir, spec.silver_dir, spec.gold_dir,
                   sources=spec.sources, label=spec.name.upper())


def summarize(results_by_dataset: dict[str, list[tuple[str, str, Path]]]) -> None:
    """Prints per-molecule scale across every dataset that was processed.

    Worth reading before training rather than after: MD22's systems are
    an order of magnitude larger than MD17's in atom count, and their
    total energies are larger by *several* orders (a 370-atom nanotube
    sits near -1e6 kcal/mol where ethanol sits near -1e5). A model, loss
    weighting, or normalisation choice tuned on MD17 will not transfer
    unscathed, and a fixed 5 A cutoff produces a far sparser relative
    graph on the nanotube than on ethanol (glossary.md section 5). None
    of that blocks cleaning - graphs are built later, in
    ml/data/datasets.py - but it should be a deliberate decision, not a
    default inherited silently.
    """
    print(f"\n{'Dataset':<8} {'Molecule':<24} {'Theory':<8} {'Atoms':>6} {'Configs':>9} "
          f"{'E mean':>14} {'E std':>12}")
    print("-" * 100)
    for dataset, results in results_by_dataset.items():
        for molecule, theory, gold_path in results:
            gold = np.load(gold_path, allow_pickle=True)
            z, E = gold["z"], gold["E"]
            print(f"{dataset:<8} {molecule:<24} {theory:<8} {z.shape[0]:>6} {E.shape[0]:>9} "
                  f"{E.mean():>14.2f} {E.std():>12.2f}")
    print("-" * 100)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Clean bronze MD17/MD22 data into silver and gold stages.")
    parser.add_argument("--dataset", choices=[*sorted(DATASETS), "all"], default="all",
                        help="which dataset to process (default: all)")
    parser.add_argument("--no-summary", action="store_true",
                        help="skip the per-molecule scale summary printed at the end")
    args = parser.parse_args(argv)

    names = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    results_by_dataset = {name: run_dataset(name) for name in names}

    if not args.no_summary and any(results_by_dataset.values()):
        summarize(results_by_dataset)

    total = sum(len(r) for r in results_by_dataset.values())
    expected = sum(len(DATASETS[n].sources) for n in names)
    print(f"\n{total}/{expected} sources processed across {', '.join(names)}")
    return 0 if total == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
