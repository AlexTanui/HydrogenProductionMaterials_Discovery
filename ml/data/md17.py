"""Load raw MD17/MD22 trajectory files and build neighbor graphs.

Handles both source formats found in data/bronze/ (see glossary.md SS3):
  - a direct .npz with z/R/E/F arrays
  - a .zip containing pre-split train/test members, either as nested .npz
    files or as extended-XYZ .xyz text

Migrated from rajcleaning.ipynb (Raj's original data-cleaning pass), with
the __MACOSX zip-junk bug and the silent-unknown-element fallback fixed -
see the git history for what changed and why.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REQUIRED_KEYS = ("z", "R", "E", "F")

# Elements that actually appear across the current MD17/MD22 roster
# (small organic molecules, DNA base pairs, a nanotube). Extend as new
# molecules are added - an unrecognized symbol raises rather than
# silently mapping to atomic number 0, which isn't a real element and
# would corrupt the graph without any visible error.
_PERIODIC_TABLE = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9,
    "Na": 11, "Mg": 12, "P": 15, "S": 16, "Cl": 17,
}


def _symbol_to_z(symbol: str) -> int:
    try:
        return _PERIODIC_TABLE[symbol]
    except KeyError:
        raise ValueError(
            f"Unrecognized element symbol {symbol!r} - add it to _PERIODIC_TABLE "
            "in ml/data/md17.py rather than silently treating it as atomic number 0"
        ) from None


@dataclass
class MD17Sample:
    """One molecule+theory's worth of trajectory data, already validated."""

    molecule: str
    theory: str
    z: np.ndarray
    R: np.ndarray
    E: np.ndarray
    F: np.ndarray
    r_unit: str = "Ang"
    e_unit: str = "kcal/mol"

    def __post_init__(self) -> None:
        self.z = np.asarray(self.z, dtype=np.int64).reshape(-1)
        self.R = np.asarray(self.R, dtype=np.float32)
        self.F = np.asarray(self.F, dtype=np.float32)
        self.E = np.asarray(self.E, dtype=np.float32).reshape(-1)

        if self.R.ndim == 2:
            self.R = self.R[None, ...]
        if self.F.ndim == 2:
            self.F = self.F[None, ...]

        n_atoms = self.z.shape[0]
        if self.R.shape[1:] != (n_atoms, 3):
            raise ValueError(f"{self.molecule}: R shape {self.R.shape} inconsistent with {n_atoms} atoms")
        if self.F.shape != self.R.shape:
            raise ValueError(f"{self.molecule}: F shape {self.F.shape} != R shape {self.R.shape}")
        if self.E.shape[0] != self.R.shape[0]:
            raise ValueError(f"{self.molecule}: E has {self.E.shape[0]} entries, R has {self.R.shape[0]} configs")

    @property
    def n_atoms(self) -> int:
        return int(self.z.shape[0])

    @property
    def n_configs(self) -> int:
        return int(self.R.shape[0])


def _infer_theory(filename: str) -> str:
    name = filename.lower()
    if "ccsd_t" in name or "ccsd-t" in name:
        return "ccsd_t"
    if "ccsd" in name:
        return "ccsd"
    return "dft"


def _infer_molecule(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"^md1[78]_", "", stem)
    stem = re.sub(r"[_-]?(dft|ccsd_t|ccsd-t|ccsd|train|test)$", "", stem, flags=re.IGNORECASE)
    return stem


def load_npz(path: str | Path, molecule: Optional[str] = None, theory: Optional[str] = None) -> MD17Sample:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"{path.name}: missing required keys {missing}, has {list(data.keys())}")
    inferred_molecule = str(data["name"]) if "name" in data.files else _infer_molecule(path.name)
    return MD17Sample(
        molecule=molecule or inferred_molecule,
        theory=theory or _infer_theory(path.name),
        z=data["z"], R=data["R"], E=data["E"], F=data["F"],
        r_unit=str(data["r_unit"]) if "r_unit" in data.files else "Ang",
        e_unit=str(data["e_unit"]) if "e_unit" in data.files else "kcal/mol",
    )


def _split_xyz_frames(text: str) -> list[str]:
    lines = text.strip().splitlines()
    frames = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n_atoms = int(lines[i].strip())
        frames.append("\n".join(lines[i : i + 2 + n_atoms]))
        i += 2 + n_atoms
    return frames


def _parse_xyz_frame(text: str) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    lines = [l for l in text.strip().splitlines() if l.strip()]
    n_atoms = int(lines[0].strip())
    comment = lines[1]
    energy_match = re.search(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", comment)
    energy = float(energy_match.group()) if energy_match else float("nan")
    symbols, coords, forces = [], [], []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
        forces.append([float(x) for x in parts[4:7]])
    z = np.array([_symbol_to_z(s) for s in symbols], dtype=np.int64)
    return z, energy, np.array(coords, dtype=np.float32), np.array(forces, dtype=np.float32)


def _split_name(member: str) -> Optional[str]:
    lower = member.lower()
    if "train" in lower:
        return "train"
    if "test" in lower:
        return "test"
    return None


def _is_real_member(name: str) -> bool:
    """Excludes macOS's __MACOSX/._* AppleDouble resource-fork junk.

    These mirror the real filename (so they still end in .npz/.xyz and
    still match "train"/"test" substring checks) but contain resource-fork
    binary data, not the actual archived content - loading one raises an
    UnpicklingError. Every zip zipped on a Mac carries these.
    """
    return "__MACOSX" not in name and not Path(name).name.startswith("._")


def load_split_zip(path: str | Path, molecule: Optional[str] = None, theory: Optional[str] = None) -> dict[str, MD17Sample]:
    import io
    import zipfile

    path = Path(path)
    molecule = molecule or _infer_molecule(path.name)
    theory = theory or _infer_theory(path.name)

    result: dict[str, MD17Sample] = {}
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        npz_members = [n for n in names if n.lower().endswith(".npz") and _is_real_member(n)]
        xyz_members = [n for n in names if n.lower().endswith(".xyz") and _is_real_member(n)]

        if npz_members:
            for member in npz_members:
                split = _split_name(member)
                if split is None:
                    continue
                with zf.open(member) as f:
                    buf = io.BytesIO(f.read())
                    data = np.load(buf, allow_pickle=True)
                    result[split] = MD17Sample(
                        molecule=molecule, theory=theory,
                        z=data["z"], R=data["R"], E=data["E"], F=data["F"],
                        r_unit=str(data["r_unit"]) if "r_unit" in data.files else "Ang",
                        e_unit=str(data["e_unit"]) if "e_unit" in data.files else "kcal/mol",
                    )
        elif xyz_members:
            for member in xyz_members:
                split = _split_name(member)
                if split is None:
                    continue
                with zf.open(member) as f:
                    text = f.read().decode("utf-8")
                z = None
                Rs, Es, Fs = [], [], []
                for frame_text in _split_xyz_frames(text):
                    fz, e, r, force = _parse_xyz_frame(frame_text)
                    z = fz if z is None else z
                    Rs.append(r); Es.append(e); Fs.append(force)
                result[split] = MD17Sample(molecule=molecule, theory=theory, z=z, R=np.stack(Rs), E=np.array(Es), F=np.stack(Fs))
        else:
            raise ValueError(f"{path.name}: no .npz or .xyz members found ({names})")

    if "train" not in result or "test" not in result:
        raise ValueError(f"{path.name}: expected both train and test splits, found {list(result.keys())}")
    return result


def _gaussian_rbf(dist: torch.Tensor, num_rbf: int = 16, cutoff: float = 5.0) -> torch.Tensor:
    """Expands scalar distances into a Gaussian radial basis.

    Gilmer et al. (2017) - the Phase 1 reference - expand interatomic
    distance this way before feeding it to the edge network; a raw scalar
    distance is a poor edge feature because energy/forces are sharply
    nonlinear in distance near equilibrium bond lengths, and a plain
    MLP over one raw number struggles to learn that without excess
    capacity. This is foundational to the baseline, not an upgrade.
    """
    centers = torch.linspace(0.0, cutoff, num_rbf, device=dist.device)
    width = centers[1] - centers[0]
    return torch.exp(-((dist.unsqueeze(-1) - centers) ** 2) / (2 * width**2))


def build_graph(R_frame: np.ndarray, cutoff: float = 5.0, num_rbf: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds a neighbor graph from one frame's 3D coordinates.

    Returns (edge_index, edge_attr) where edge_attr is RBF-expanded
    distance, shape [num_edges, num_rbf] - not raw scalar distance.
    """
    pos = torch.as_tensor(R_frame, dtype=torch.float32)
    dist = torch.cdist(pos, pos)
    n_atoms = pos.shape[0]
    mask = (dist <= cutoff) & ~torch.eye(n_atoms, dtype=torch.bool)
    row, col = mask.nonzero(as_tuple=True)
    edge_index = torch.stack([row, col], dim=0)
    edge_attr = _gaussian_rbf(dist[row, col], num_rbf=num_rbf, cutoff=cutoff)
    return edge_index, edge_attr
