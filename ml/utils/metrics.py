"""Streaming energy/force accuracy metrics for the benchmark harness (SCRUM-37).

Every phase of this project is measured by this module, so it is built for
a test set of thousands of frames arriving in mini-batches rather than for
one array in memory.

**The rule this module exists to enforce:** accumulate sums of absolute
errors, sums of squared errors, and counts; divide exactly once, in
`compute()`. Never average per-batch metric values. Averaging batch RMSEs
is wrong even for equal batch sizes because the square root does not
commute with the mean, and averaging batch MAEs is wrong whenever batch
sizes differ - which the last batch of a dataset almost always does.
`tests/test_metrics.py` pins this with a deliberately unequal 4+2+1 split.

Two conventions exist for each quantity, both always computed and both
always labelled in the output, because a bare number in a benchmark table
is not interpretable without them:

- energy `total` (mean of |dE|) vs `per_atom` (each frame's error divided
  by its own atom count first)
- force `component` (mean over all N*3 scalars - the MD17/SchNet/PaiNN
  convention) vs `atom_norm` (mean of the L2 norm of each atom's error
  vector)

The two force MAE modes are not related by a fixed factor; the ratio moves
with the error distribution. The two force RMSE modes *are*: they share a
sum of squares and differ only in their denominator, so `atom_norm` is
exactly sqrt(3) times `component`.

Results are keyed by (molecule, theory), never pooled into one number by
default - `data/bronze/` holds ethanol at both DFT and CCSD(T), whose
absolute energies are not comparable. `compute()` returns plain
JSON-serialisable values and records which convention and unit produced
each one, ready for the run record in SCRUM-48 (not written here).

Uncertainty metrics (ECE, uncertainty-error correlation) are deliberately
absent: they are Dongxiao's to define. A stochastic `ModelOutput` scores
identically to a deterministic one here - `E_var`/`F_var` are noted in the
output and otherwise ignored.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ml.utils.contract import (
    DatasetKey,
    IncomparableData,
    ModelOutput,
    ReferenceData,
    Units,
    require_compatible,
)

SCHEMA_VERSION = 1

ENERGY_NORMALISATIONS = ("total", "per_atom")
FORCE_MODES = ("component", "atom_norm")

ENERGY_FIELDS = (
    "energy_mae_total",
    "energy_rmse_total",
    "energy_mae_per_atom",
    "energy_rmse_per_atom",
)
FORCE_FIELDS = (
    "force_mae_component",
    "force_rmse_component",
    "force_mae_atom_norm",
    "force_rmse_atom_norm",
)


@dataclass
class _KeyState:
    """Running sums for one (molecule, theory). Never a mean until compute()."""

    key: DatasetKey
    units: Units
    n_frames: int = 0
    n_atoms: int = 0          # summed over frames, not atoms-per-frame
    n_stochastic_frames: int = 0

    e_abs_total: float = 0.0
    e_sq_total: float = 0.0
    e_abs_per_atom: float = 0.0
    e_sq_per_atom: float = 0.0

    f_abs_component: float = 0.0
    # Also the sum of squared atom-norms: sum_a ||dF_a||^2 == sum_ac dF_ac^2.
    # The two force RMSE modes differ only in denominator.
    f_sq_component: float = 0.0
    f_abs_atom_norm: float = 0.0

    def add(self, other: "_KeyState") -> None:
        self.n_frames += other.n_frames
        self.n_atoms += other.n_atoms
        self.n_stochastic_frames += other.n_stochastic_frames
        self.e_abs_total += other.e_abs_total
        self.e_sq_total += other.e_sq_total
        self.e_abs_per_atom += other.e_abs_per_atom
        self.e_sq_per_atom += other.e_sq_per_atom
        self.f_abs_component += other.f_abs_component
        self.f_sq_component += other.f_sq_component
        self.f_abs_atom_norm += other.f_abs_atom_norm


def _describe_fields(units: Units) -> dict[str, dict[str, str]]:
    """What convention and unit produced each metric field."""
    per_atom = f"{units.energy}/atom"
    force = units.force
    return {
        "energy_mae_total": {
            "quantity": "energy", "statistic": "mae",
            "normalisation": "total", "unit": units.energy,
        },
        "energy_rmse_total": {
            "quantity": "energy", "statistic": "rmse",
            "normalisation": "total", "unit": units.energy,
        },
        "energy_mae_per_atom": {
            "quantity": "energy", "statistic": "mae",
            "normalisation": "per_atom", "unit": per_atom,
        },
        "energy_rmse_per_atom": {
            "quantity": "energy", "statistic": "rmse",
            "normalisation": "per_atom", "unit": per_atom,
        },
        "force_mae_component": {
            "quantity": "force", "statistic": "mae",
            "mode": "component", "unit": force,
        },
        "force_rmse_component": {
            "quantity": "force", "statistic": "rmse",
            "mode": "component", "unit": force,
        },
        "force_mae_atom_norm": {
            "quantity": "force", "statistic": "mae",
            "mode": "atom_norm", "unit": force,
        },
        "force_rmse_atom_norm": {
            "quantity": "force", "statistic": "rmse",
            "mode": "atom_norm", "unit": force,
        },
    }


def _metrics_from_sums(state: _KeyState) -> dict[str, float]:
    """The single division. Everything above this line is a running sum."""
    n_frames = state.n_frames
    n_atoms = state.n_atoms
    n_components = 3 * n_atoms
    return {
        "energy_mae_total": state.e_abs_total / n_frames,
        "energy_rmse_total": math.sqrt(state.e_sq_total / n_frames),
        "energy_mae_per_atom": state.e_abs_per_atom / n_frames,
        "energy_rmse_per_atom": math.sqrt(state.e_sq_per_atom / n_frames),
        "force_mae_component": state.f_abs_component / n_components,
        "force_rmse_component": math.sqrt(state.f_sq_component / n_components),
        "force_mae_atom_norm": state.f_abs_atom_norm / n_atoms,
        # Same sum of squares as the component RMSE over a denominator 3x
        # smaller, so this is exactly sqrt(3) x force_rmse_component. The MAE
        # modes have no such fixed relation.
        "force_rmse_atom_norm": math.sqrt(state.f_sq_component / n_atoms),
    }


class MetricAccumulator:
    """Accumulates energy/force errors across mini-batches, keyed by dataset.

        acc = MetricAccumulator()
        for output, reference in eval_loader:
            acc.update(output, reference)
        results = acc.compute()

    Batches for different (molecule, theory) keys may be interleaved freely;
    each key accumulates separately and is reported separately.
    """

    def __init__(self) -> None:
        self._states: dict[DatasetKey, _KeyState] = {}

    def update(self, pred: ModelOutput, ref: ReferenceData) -> None:
        """Add one batch. Shapes, units and finiteness are checked first."""
        require_compatible(pred, ref)

        state = self._states.get(ref.key)
        if state is None:
            state = _KeyState(key=ref.key, units=ref.units)
            self._states[ref.key] = state
        elif state.units != ref.units:
            raise IncomparableData(
                f"{ref.key} was accumulated in {state.units.as_dict()} and this "
                f"batch is in {ref.units.as_dict()}. Mixing them would rescale "
                "part of the sum by a constant factor and still print a "
                "believable number. Call .to_project_units() on every batch."
            )

        # detach: forces come from autograd, and holding the graph in a metric
        # object leaks the whole eval loop's memory.
        # float64: float32 running sums drift measurably over thousands of
        # frames, which is exactly the error this module exists to avoid.
        d_e = (pred.E - ref.E).detach().double()
        counts = ref.atom_counts.double()
        state.e_abs_total += float(d_e.abs().sum())
        state.e_sq_total += float((d_e * d_e).sum())

        d_e_per_atom = d_e / counts
        state.e_abs_per_atom += float(d_e_per_atom.abs().sum())
        state.e_sq_per_atom += float((d_e_per_atom * d_e_per_atom).sum())

        d_f = (pred.F - ref.F).detach().double()
        sq = d_f * d_f
        state.f_abs_component += float(d_f.abs().sum())
        state.f_sq_component += float(sq.sum())
        state.f_abs_atom_norm += float(sq.sum(dim=1).sqrt().sum())

        state.n_frames += ref.n_frames
        state.n_atoms += ref.n_atoms
        if pred.is_stochastic:
            state.n_stochastic_frames += ref.n_frames

    def compute(self) -> dict:
        """Final results: per-key records plus an overall block where valid."""
        if not self._states:
            raise ValueError(
                "MetricAccumulator.compute() called before any update(). Returning "
                "metrics for zero frames would put a meaningless number in a "
                "benchmark table; check that the eval split is non-empty."
            )
        ordered = sorted(self._states.values(), key=lambda s: s.key)
        return {
            "schema_version": SCHEMA_VERSION,
            "conventions": {
                "energy_normalisation": list(ENERGY_NORMALISATIONS),
                "force_mode": list(FORCE_MODES),
            },
            "per_key": [self._record(s) for s in ordered],
            "overall": self._overall(ordered),
        }

    def reset(self) -> None:
        self._states.clear()

    def keys(self) -> tuple[DatasetKey, ...]:
        return tuple(sorted(self._states))

    def __len__(self) -> int:
        return len(self._states)

    @staticmethod
    def _record(state: _KeyState) -> dict:
        record = {
            **state.key.as_dict(),
            "n_frames": state.n_frames,
            "n_atoms": state.n_atoms,
            "stochastic": state.n_stochastic_frames > 0,
            "units": state.units.as_dict(),
        }
        record.update(_metrics_from_sums(state))
        record["fields"] = _describe_fields(state.units)
        return record

    @staticmethod
    def _overall(ordered: list[_KeyState]) -> dict:
        """Pool only where pooling means something.

        Forces are local per-atom quantities in identical units, so they pool
        across keys. Absolute energies do not: different levels of theory sit
        on different scales, so energy is pooled only within a single theory
        and the reason is recorded when it is left out.
        """
        unit_set = {s.units for s in ordered}
        theories = sorted({s.key.theory for s in ordered})
        molecules = sorted({s.key.molecule for s in ordered})

        overall: dict = {
            "n_keys": len(ordered),
            "n_frames": sum(s.n_frames for s in ordered),
            "n_atoms": sum(s.n_atoms for s in ordered),
            "keys": [s.key.as_dict() for s in ordered],
            "units": None,
            "excluded": {},
        }

        if len(unit_set) > 1:
            reason = (
                "keys were accumulated in more than one unit system "
                f"({sorted(u.as_dict()['energy'] for u in unit_set)}); nothing can "
                "be pooled until every batch is converted with .to_project_units()"
            )
            overall["excluded"] = {"energy": reason, "force": reason}
            return overall

        units = next(iter(unit_set))
        overall["units"] = units.as_dict()

        pooled = _KeyState(key=DatasetKey("*", "*"), units=units)
        for state in ordered:
            pooled.add(state)
        values = _metrics_from_sums(pooled)

        for name in FORCE_FIELDS:
            overall[name] = values[name]

        if len(theories) > 1:
            overall["excluded"]["energy"] = (
                f"energy is not pooled across levels of theory {theories}: absolute "
                "energies sit on different scales, so a pooled energy error is not "
                "a meaningful quantity. Read the per-key values instead."
            )
        else:
            for name in ENERGY_FIELDS:
                overall[name] = values[name]
            if len(molecules) > 1:
                overall["energy_pooling_note"] = (
                    f"pooled across {len(molecules)} molecules at theory "
                    f"{theories[0]}; totals are size-weighted, so the per-key "
                    "table is authoritative"
                )

        fields = _describe_fields(units)
        overall["fields"] = {k: v for k, v in fields.items() if k in overall}
        overall["stochastic"] = any(s.n_stochastic_frames > 0 for s in ordered)
        return overall


def _single_batch(
    pred: ModelOutput,
    ref: ReferenceData,
    quantity: str,
    statistic: str,
    normalisation: Optional[str] = None,
    mode: Optional[str] = None,
) -> float:
    """One code path: the convenience functions run the accumulator too."""
    if quantity == "energy":
        if normalisation not in ENERGY_NORMALISATIONS:
            raise ValueError(
                f"normalisation must be one of {ENERGY_NORMALISATIONS}, "
                f"got {normalisation!r}"
            )
        field = f"energy_{statistic}_{normalisation}"
    else:
        if mode not in FORCE_MODES:
            raise ValueError(f"mode must be one of {FORCE_MODES}, got {mode!r}")
        field = f"force_{statistic}_{mode}"

    accumulator = MetricAccumulator()
    accumulator.update(pred, ref)
    return accumulator.compute()["per_key"][0][field]


def energy_mae(pred: ModelOutput, ref: ReferenceData, normalisation: str = "total") -> float:
    """Mean absolute energy error for one batch, in the reference's unit."""
    return _single_batch(pred, ref, "energy", "mae", normalisation=normalisation)


def energy_rmse(pred: ModelOutput, ref: ReferenceData, normalisation: str = "total") -> float:
    """Root-mean-square energy error for one batch, in the reference's unit."""
    return _single_batch(pred, ref, "energy", "rmse", normalisation=normalisation)


def force_mae(pred: ModelOutput, ref: ReferenceData, mode: str = "component") -> float:
    """Mean absolute force error for one batch, in the reference's force unit."""
    return _single_batch(pred, ref, "force", "mae", mode=mode)


def force_rmse(pred: ModelOutput, ref: ReferenceData, mode: str = "component") -> float:
    """Root-mean-square force error for one batch, in the reference's force unit."""
    return _single_batch(pred, ref, "force", "rmse", mode=mode)
