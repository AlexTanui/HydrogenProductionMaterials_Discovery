"""Typed batch interface between the MD17 data layer and the metric harness.

Everything the harness measures arrives through these two dataclasses, so
this is the single place that knows about shapes, units, and dataset
identity. `ml/utils/metrics.py` reasons about arithmetic only.

Three rules this file exists to enforce structurally rather than by
convention:

1. **A batch is one molecule at one level of theory.** `molecule` and
   `theory` are scalar fields, not per-frame arrays, so a mixed batch has
   no representation here at all. `data/bronze/` carries ethanol at both
   DFT (`md17_ethanol.npz`) and CCSD(T) (`ethanol_ccsd_t.zip`); their
   absolute energies sit on different scales and pooling them produces a
   number that looks fine and means nothing. The dataclasses are frozen,
   so identity cannot be rewritten after the fact to force a merge, and
   the only combining helper (`require_comparable`) raises.

2. **Energy is always total, never per-atom.** Per-atom normalisation is
   a reporting convention and belongs in one place, in the metric layer.
   A contract that accepted either would make every downstream number
   ambiguous.

3. **Forces have no declared unit anywhere upstream.** `MD17Sample` in
   `ml/data/md17.py` declares `r_unit` and `e_unit` and stops there, so
   the force unit is *derived* as energy/length. That inference lives in
   exactly one visible place: `Units.force`.

Validation runs on construction and raises. A shape or unit bug that
reaches a metric still prints a float, and a plausible wrong number in a
benchmark table costs far more than a crash does.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch

if TYPE_CHECKING:  # avoids a runtime ml.utils -> ml.data dependency
    from ml.data.md17 import MD17Sample

PROJECT_ENERGY_UNIT = "kcal/mol"
PROJECT_LENGTH_UNIT = "Ang"

# Linear scale factors only - no additive offsets, so a variance scales by
# the square of the factor that scales its mean (see `to_project_units`).
_ENERGY_TO_KCAL_PER_MOL = {
    "kcal/mol": 1.0,
    "eV": 23.060547830618307,
    "kJ/mol": 0.2390057361376673,   # 1 / 4.184
    "Hartree": 627.5094740631,
}
_LENGTH_TO_ANG = {
    "Ang": 1.0,
    "Angstrom": 1.0,
    "nm": 10.0,
    "Bohr": 0.529177210903,
}


class ContractViolation(ValueError):
    """A batch does not satisfy this interface. Never caught internally."""


class IncomparableData(ContractViolation):
    """Two batches were combined that must not be (identity or units)."""


@dataclass(frozen=True)
class Units:
    """Energy and length units for one batch. Force is derived, not declared."""

    energy: str = PROJECT_ENERGY_UNIT
    length: str = PROJECT_LENGTH_UNIT

    def __post_init__(self) -> None:
        if self.energy not in _ENERGY_TO_KCAL_PER_MOL:
            raise ContractViolation(
                f"unknown energy unit {self.energy!r}; known: "
                f"{sorted(_ENERGY_TO_KCAL_PER_MOL)}. Add it to "
                "_ENERGY_TO_KCAL_PER_MOL in ml/utils/contract.py rather than "
                "converting at the call site."
            )
        if self.length not in _LENGTH_TO_ANG:
            raise ContractViolation(
                f"unknown length unit {self.length!r}; known: "
                f"{sorted(_LENGTH_TO_ANG)}. Add it to _LENGTH_TO_ANG in "
                "ml/utils/contract.py rather than converting at the call site."
            )

    @property
    def force(self) -> str:
        """THE force-unit assumption for this project.

        `MD17Sample` declares no force unit, so it is inferred as
        energy/length. If a source ever ships forces in a unit that is not
        its own energy unit over its own length unit, this is the one line
        that has to change - and the one line to check when force MAE looks
        wrong by a suspiciously round factor.
        """
        return f"{self.energy}/{self.length}"

    @property
    def is_project_units(self) -> bool:
        return self.energy == PROJECT_ENERGY_UNIT and self.length == PROJECT_LENGTH_UNIT

    @classmethod
    def from_sample(cls, sample: "MD17Sample") -> "Units":
        return cls(energy=str(sample.e_unit), length=str(sample.r_unit))

    def as_dict(self) -> dict[str, str]:
        """JSON-serialisable, for the run record (SCRUM-48)."""
        return {"energy": self.energy, "length": self.length, "force": self.force}


@dataclass(frozen=True, order=True)
class DatasetKey:
    """What makes two batches comparable. Hashable, so it keys result dicts."""

    molecule: str
    theory: str

    def __str__(self) -> str:
        return f"{self.molecule}/{self.theory}"

    def as_dict(self) -> dict[str, str]:
        return {"molecule": self.molecule, "theory": self.theory}


def _check_tensor(t: object, name: str) -> None:
    if not isinstance(t, torch.Tensor):
        raise ContractViolation(f"{name} must be a torch.Tensor, got {type(t).__name__}")


def _check_finite(t: torch.Tensor, name: str, produced_by: str) -> None:
    """Reject NaN/Inf, naming the field and saying who produced the values.

    The wording matters. A traceback out of the metric layer reads as "the
    harness is broken" unless it says otherwise, and the harness then gets
    blamed for the bug it just caught.
    """
    if bool(torch.isfinite(t).all()):
        return
    n_bad = int((~torch.isfinite(t)).sum())
    raise ContractViolation(
        f"{name} contains {n_bad} non-finite value(s) (NaN or Inf) out of "
        f"{t.numel()}. {produced_by} The metric harness has not failed here - it "
        "is refusing to average non-finite values into a benchmark number, "
        "because one NaN turns every mean downstream of it into NaN."
    )


_FROM_MODEL = (
    "These values were produced by the model, not by the metric: the "
    "checkpoint emitted them, which usually means training diverged."
)
_FROM_DATA = (
    "These values came from the dataset, not from the model or the metric: "
    "ml/data/preprocessing.py NaN-checks at the silver stage, so gold data "
    "arriving here with non-finite entries points at the data pipeline."
)


def _check_energy(t: torch.Tensor, name: str, produced_by: str) -> None:
    _check_tensor(t, name)
    if t.dim() != 1:
        raise ContractViolation(
            f"{name} must have shape (B,), got {tuple(t.shape)}. A (B, 1) tensor "
            "is rejected deliberately: subtracting it from a (B,) reference "
            "broadcasts to (B, B) and yields a plausible wrong number instead "
            "of an error. Squeeze it at the model head."
        )
    if not t.is_floating_point():
        raise ContractViolation(f"{name} must be a floating-point tensor, got {t.dtype}")
    _check_finite(t, name, produced_by)


def _check_forces(t: torch.Tensor, name: str, produced_by: str) -> None:
    _check_tensor(t, name)
    if t.dim() != 2 or t.shape[1] != 3:
        raise ContractViolation(
            f"{name} must have shape (N, 3) - PyG convention, atoms of all "
            f"frames concatenated, not padded (B, n_atoms, 3) - got {tuple(t.shape)}"
        )
    if not t.is_floating_point():
        raise ContractViolation(f"{name} must be a floating-point tensor, got {t.dtype}")
    _check_finite(t, name, produced_by)


def _check_variance(v: Optional[torch.Tensor], mean: torch.Tensor, name: str) -> None:
    if v is None:
        return
    _check_tensor(v, name)
    if v.shape != mean.shape:
        raise ContractViolation(
            f"{name} has shape {tuple(v.shape)}, must match its mean {tuple(mean.shape)}"
        )
    if not v.is_floating_point():
        raise ContractViolation(f"{name} must be a floating-point tensor, got {v.dtype}")
    _check_finite(v, name, _FROM_MODEL)
    if bool((v < 0).any()):
        raise ContractViolation(
            f"{name} has negative entries; a variance cannot be negative. This is a "
            "model output, not a metric failure - a model emitting log-variance "
            "must exponentiate before building ModelOutput."
        )


@dataclass(frozen=True, eq=False)
class ModelOutput:
    """One batch of predictions.

    `E_var`/`F_var` are the seam for Phase 2/3 uncertainty: Phase 1 leaves
    them None, and adding ECE or uncertainty-error correlation later reads
    these fields without changing this file.

    `eq=False` because a generated __eq__ would compare tensor fields with
    `==`, which returns a tensor rather than a bool and breaks any truth
    test. Value equality is only needed on Units and DatasetKey, which
    have it.
    """

    E: torch.Tensor                       # (B,) total energy per frame
    F: torch.Tensor                       # (N, 3) per-atom forces
    E_var: Optional[torch.Tensor] = None  # (B,)
    F_var: Optional[torch.Tensor] = None  # (N, 3)
    units: Units = Units()

    def __post_init__(self) -> None:
        _check_energy(self.E, "ModelOutput.E", _FROM_MODEL)
        _check_forces(self.F, "ModelOutput.F", _FROM_MODEL)
        _check_variance(self.E_var, self.E, "ModelOutput.E_var")
        _check_variance(self.F_var, self.F, "ModelOutput.F_var")

    @property
    def n_frames(self) -> int:
        return int(self.E.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.F.shape[0])

    @property
    def is_stochastic(self) -> bool:
        return self.E_var is not None or self.F_var is not None

    def to_project_units(self) -> "ModelOutput":
        a, b = _conversion_factors(self.units)
        if a == 1.0 and b == 1.0:
            return self
        f = a / b
        return ModelOutput(
            E=self.E * a,
            F=self.F * f,
            # A variance scales by the square of its mean's linear factor.
            E_var=None if self.E_var is None else self.E_var * (a * a),
            F_var=None if self.F_var is None else self.F_var * (f * f),
            units=Units(),
        )


@dataclass(frozen=True, eq=False)
class ReferenceData:
    """One batch of ground truth, for exactly one molecule at one theory."""

    E: torch.Tensor            # (B,) total energy per frame
    F: torch.Tensor            # (N, 3) per-atom forces
    batch: torch.Tensor        # (N,) int64, non-decreasing, values 0..B-1
    molecule: str
    theory: str
    units: Units = Units()
    z: Optional[torch.Tensor] = None   # (N,) int64, carried through from the loader

    def __post_init__(self) -> None:
        _check_energy(self.E, "ReferenceData.E", _FROM_DATA)
        _check_forces(self.F, "ReferenceData.F", _FROM_DATA)
        if not self.molecule or not self.theory:
            raise ContractViolation(
                "molecule and theory must both be non-empty; they are the "
                "dataset's identity, not optional metadata"
            )
        _check_batch(self.batch, n_atoms=self.n_atoms, n_frames=self.n_frames)
        if self.z is not None:
            _check_tensor(self.z, "ReferenceData.z")
            if self.z.dim() != 1 or self.z.shape[0] != self.n_atoms:
                raise ContractViolation(
                    f"ReferenceData.z must have shape (N,) = ({self.n_atoms},), "
                    f"got {tuple(self.z.shape)}"
                )

    @property
    def n_frames(self) -> int:
        return int(self.E.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.F.shape[0])

    @property
    def atom_counts(self) -> torch.Tensor:
        """(B,) int64 - atoms per frame, the only correct per-atom denominator."""
        return torch.bincount(self.batch, minlength=self.n_frames)

    @property
    def key(self) -> DatasetKey:
        return DatasetKey(molecule=self.molecule, theory=self.theory)

    @classmethod
    def from_sample(
        cls,
        sample: "MD17Sample",
        frame_indices: Union[Sequence[int], torch.Tensor],
    ) -> "ReferenceData":
        """Build a batch from Shijin's loader. The only seam between the two.

        A change to `MD17Sample`'s field names or layout breaks this one
        function rather than every call site in the harness.
        """
        idx = torch.as_tensor(frame_indices, dtype=torch.long).reshape(-1)
        if idx.numel() == 0:
            raise ContractViolation("frame_indices is empty; a batch needs at least one frame")
        n_configs, n_atoms = int(sample.n_configs), int(sample.n_atoms)
        if bool(((idx < 0) | (idx >= n_configs)).any()):
            raise ContractViolation(
                f"frame_indices out of range for {sample.molecule}/{sample.theory}: "
                f"values must lie in [0, {n_configs})"
            )
        n_frames = int(idx.numel())
        return cls(
            E=torch.as_tensor(sample.E)[idx],
            # (B, n_atoms, 3) -> (B * n_atoms, 3), frame-major so that atoms of
            # one frame stay contiguous and line up with `batch` below.
            F=torch.as_tensor(sample.F)[idx].reshape(n_frames * n_atoms, 3),
            batch=torch.repeat_interleave(torch.arange(n_frames, dtype=torch.int64), n_atoms),
            molecule=str(sample.molecule),
            theory=str(sample.theory),
            units=Units.from_sample(sample),
            z=torch.as_tensor(sample.z, dtype=torch.int64).repeat(n_frames),
        )

    def to_project_units(self) -> "ReferenceData":
        a, b = _conversion_factors(self.units)
        if a == 1.0 and b == 1.0:
            return self
        return replace(self, E=self.E * a, F=self.F * (a / b), units=Units())


def _check_batch(batch: torch.Tensor, n_atoms: int, n_frames: int) -> None:
    name = "ReferenceData.batch"
    _check_tensor(batch, name)
    if batch.dim() != 1:
        raise ContractViolation(f"{name} must be 1-D, got {tuple(batch.shape)}")
    if batch.dtype != torch.int64:
        raise ContractViolation(f"{name} must be int64 (PyG convention), got {batch.dtype}")
    if batch.shape[0] != n_atoms:
        raise ContractViolation(
            f"{name} has {batch.shape[0]} entries but F has {n_atoms} rows; "
            "there must be exactly one batch entry per atom"
        )
    if n_atoms == 0:
        raise ContractViolation("batch contains no atoms")
    if bool((batch[1:] < batch[:-1]).any()):
        raise ContractViolation(
            f"{name} must be non-decreasing - PyG concatenates frames in order, "
            "and per-frame grouping in the metric layer relies on it"
        )
    if int(batch[0]) != 0 or int(batch[-1]) != n_frames - 1:
        raise ContractViolation(
            f"{name} spans frames {int(batch[0])}..{int(batch[-1])} but E has "
            f"{n_frames} frames; they must agree"
        )
    if bool((torch.bincount(batch, minlength=n_frames) == 0).any()):
        raise ContractViolation(f"{name} leaves at least one frame with no atoms")


def _conversion_factors(units: Units) -> tuple[float, float]:
    return _ENERGY_TO_KCAL_PER_MOL[units.energy], _LENGTH_TO_ANG[units.length]


def require_compatible(output: ModelOutput, reference: ReferenceData) -> None:
    """Check a prediction against the ground truth it is scored on."""
    if output.n_frames != reference.n_frames:
        raise ContractViolation(
            f"prediction has {output.n_frames} frames, reference has {reference.n_frames}"
        )
    if output.n_atoms != reference.n_atoms:
        raise ContractViolation(
            f"prediction has {output.n_atoms} atoms, reference has {reference.n_atoms}"
        )
    if output.units != reference.units:
        raise ContractViolation(
            f"unit mismatch: prediction in {output.units.as_dict()}, reference in "
            f"{reference.units.as_dict()}. Comparing these rescales every metric "
            "by a constant factor while still printing a believable number. "
            "Call .to_project_units() on both."
        )


def require_comparable(a: ReferenceData, b: ReferenceData) -> None:
    """Check that two batches may be accumulated together."""
    if a.key != b.key:
        raise IncomparableData(
            f"refusing to combine {a.key} with {b.key}. Absolute energies are not "
            "comparable across molecules or levels of theory - ethanol exists at "
            "both DFT and CCSD(T) in data/bronze/ - so these must be reported as "
            "separate keys, never pooled."
        )
    if a.units != b.units:
        raise IncomparableData(
            f"refusing to combine {a.key} batches in {a.units.as_dict()} and "
            f"{b.units.as_dict()}; convert with .to_project_units() first"
        )
