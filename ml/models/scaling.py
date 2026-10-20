"""One energy scale-and-shift, applied identically to every model in the harness.

MD17 total energies sit on a large constant with a small physically
meaningful spread on top: ethanol/DFT's training split has a mean of about
-97,196 kcal/mol and a standard deviation of about 4.2. A freshly
initialised network outputs O(1), so without help the optimiser spends its
first epochs travelling five orders of magnitude instead of learning
chemistry. Every model is therefore wrapped in `ScaledEnergyModel`:

    E = scale * E_model + n_atoms * shift
    F = scale * F_model

with per-atom statistics computed **once, from the training subset only**
(never val or test):

    shift = mean(E_train) / n_atoms
    scale = std(E_train)  / n_atoms

Why one wrapper rather than per-model conventions (docs/reviews/
scrum-52-painn-review.md, H3): two conventions were in use - a mean
subtracted outside the model in train.py, a per-atom shift inside PaiNN
and PhysNet - and evaluate.py added nothing back, so any model trained
with the outside shift scored about 97,196 kcal/mol off. Here the shift
lives in the model's own buffers, so the checkpoint carries it and
evaluation, the backend and any later consumer get absolute energies in
kcal/mol with no side channel and no per-model special case. The loss is
on absolute energies.

Forces stay exactly ``-dE/dR`` for the wrapped energy: the shift term does
not depend on positions, and `scale` multiplies energy and forces alike.

Nothing is applied twice as long as each model's own normalisation stays
at identity: PaiNN's internal `energy_shift`/`energy_scale` buffers keep
their defaults (0, 1), and PhysNet's `init_shift_from_energies` is never
called by the harness (its learnable per-element `atom_shift`/`atom_scale`
start at 0 and 1).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

__all__ = ["ScaledEnergyModel"]


class ScaledEnergyModel(nn.Module):
    """Wraps a registered model: ``E = scale * E_model + n_atoms * shift``, ``F = scale * F_model``.

    `shift` and `scale` are per-atom, in kcal/mol, and stored as buffers so
    they travel in the checkpoint's state_dict (see ml/models/registry.py).
    The defaults (0, 1) are the identity, used when a checkpoint is rebuilt
    before its state_dict is loaded.
    """

    def __init__(self, model: nn.Module, shift: float = 0.0, scale: float = 1.0) -> None:
        super().__init__()
        if not scale > 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.model = model
        self.register_buffer("shift", torch.tensor(float(shift)))
        self.register_buffer("scale", torch.tensor(float(scale)))

    @staticmethod
    def statistics_from_energies(energies: Tensor, n_atoms: int) -> tuple[float, float]:
        """Per-atom (shift, scale) from total energies: (mean / n_atoms, std / n_atoms).

        Pass the *training subset's* energies only. Computed in float64: the
        mean of ~-97,000 kcal/mol over thousands of frames loses digits in
        float32. `std` is the unbiased (n - 1) estimate, torch's default.
        """
        energies = torch.as_tensor(energies).detach().double().reshape(-1)
        if energies.numel() < 2:
            raise ValueError("need at least two training energies to estimate a spread")
        if n_atoms < 1:
            raise ValueError(f"n_atoms must be positive, got {n_atoms}")
        return float(energies.mean()) / n_atoms, float(energies.std()) / n_atoms

    def energy_scaling(self) -> dict:
        """The two statistics as plain numbers, for checkpoints and run records."""
        return {
            "shift_per_atom": float(self.shift),
            "scale_per_atom": float(self.scale),
            "unit": "kcal/mol",
            "formula": "E = scale * E_model + n_atoms * shift; F = scale * F_model",
        }

    @property
    def supports_distribution(self) -> bool:
        """Whether the wrapped model exposes `predict_distribution` (BLIP;
        Phase 3 once it lands). Generic capability probe, same spirit as
        `cutoff_radius` below - evaluate.py checks this before deciding
        whether to score one stochastic sample or a full MC distribution,
        without a per-model branch."""
        return hasattr(self.model, "predict_distribution")

    def predict_distribution(
        self,
        z: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        batch: Optional[Tensor] = None,
        num_mc_samples: Optional[int] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Rescaled (E_mean, E_var, F_mean, F_var) in absolute kcal/mol / kcal/mol/A.

        Only valid when `supports_distribution` is True. Variance scales by
        `scale**2` - same rule as `ModelOutput.to_project_units()` in
        ml/utils/contract.py (Var(aX) = a^2 Var(X); the shift is a constant
        and does not touch variance at all).
        """
        if not self.supports_distribution:
            raise AttributeError(
                f"{type(self.model).__name__} has no predict_distribution; "
                "check supports_distribution before calling this"
            )
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        e_mean, e_var, f_mean, f_var = self.model.predict_distribution(
            z, pos, edge_index, edge_attr, batch=batch, num_mc_samples=num_mc_samples
        )
        n_atoms = torch.bincount(batch, minlength=e_mean.shape[0]).to(e_mean.dtype)
        scale_sq = self.scale * self.scale
        return (
            self.scale * e_mean + n_atoms * self.shift,
            scale_sq * e_var,
            self.scale * f_mean,
            scale_sq * f_var,
        )

    @property
    def supports_training_distribution(self) -> bool:
        """Whether the wrapped model exposes `predict_training_distribution`
        (BLIP with `train_num_mc_samples` > 1). See that method and
        ml/training/train.py's `batch_loss`."""
        return hasattr(self.model, "predict_training_distribution")

    def predict_training_distribution(
        self,
        z: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        batch: Optional[Tensor] = None,
        num_mc_samples: Optional[int] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Rescaled (E_mean, E_var, F_mean, F_var), gradient-carrying - the
        training-time counterpart to `predict_distribution`. No
        `torch.no_grad()` anywhere in this path; calling it outside a
        training context just wastes compute, it does not break anything.
        """
        if not self.supports_training_distribution:
            raise AttributeError(
                f"{type(self.model).__name__} has no predict_training_distribution; "
                "check supports_training_distribution before calling this"
            )
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        e_mean, e_var, f_mean, f_var = self.model.predict_training_distribution(
            z, pos, edge_index, edge_attr, batch=batch, num_mc_samples=num_mc_samples
        )
        n_atoms = torch.bincount(batch, minlength=e_mean.shape[0]).to(e_mean.dtype)
        scale_sq = self.scale * self.scale
        return (
            self.scale * e_mean + n_atoms * self.shift,
            scale_sq * e_var,
            self.scale * f_mean,
            scale_sq * f_var,
        )

    @property
    def cutoff_radius(self) -> Optional[float]:
        """The wrapped model's own cutoff, whatever its constructor calls it.

        Read from `config()`, so `evaluate.py`'s cutoff check and
        `train.py`'s config check work for every model without a per-model
        branch. Constructors name it `cutoff` or (PaiNN) `cutoff_radius`.
        """
        config = self.model.config()
        value = config.get("cutoff_radius", config.get("cutoff"))
        return None if value is None else float(value)

    def predict_energy_and_forces(
        self,
        z: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        batch: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Absolute energy (B,) in kcal/mol and forces (N, 3) in kcal/mol/A."""
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        energy, forces = self.model.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)
        n_atoms = torch.bincount(batch, minlength=energy.shape[0]).to(energy.dtype)
        return self.scale * energy + n_atoms * self.shift, self.scale * forces

    def forward(self, z, pos, edge_index, edge_attr=None, batch=None) -> tuple[Tensor, Tensor]:
        return self.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)
