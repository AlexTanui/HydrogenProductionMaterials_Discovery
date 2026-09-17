"""PhysNet (Unke & Meuwly, JCTC 2019) for the Phase 1 bake-off.

Energy is predicted as a sum of atom-wise contributions,
``E = sum_i (scale[z_i] * E_i + shift[z_i])``, with forces from autograd
(``F = -dE/dR``). This is PhysNet's **short-range term only**: the
electrostatic and dispersion corrections of the full model are omitted,
as the bake-off ticket specifies. MD17 is a single fixed-composition
neutral organic molecule sampled near equilibrium, so there is no
composition variation for a charge model to explain and no long-range
regime for a dispersion tail to matter in; both terms would be fitting
constants. This is recorded in ``docs/model_cards/physnet.md`` as a
deliberate simplification, not an oversight.

Interface, matching the bake-off contract so the shared training loop can
call any competing model interchangeably::

    predict_energy(z, edge_index, edge_attr, batch) -> [n_graphs]
    predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)
        -> ([n_graphs], [n_atoms, 3])

Deliberately free of torch-geometric: aggregation is ``index_add_``, so
this module imports only ``torch``. PyG is needed to *load* data
(``ml/data/datasets.py``), not to run the model, and keeping the
dependency out of the model file means the tests run anywhere.

READ THIS BEFORE COPYING THE PATTERN INTO ANOTHER BAKE-OFF MODEL
---------------------------------------------------------------
``edge_attr`` arriving from ``ml/data/datasets.py::MD17Dataset`` is
computed in numpy, at dataset-construction time, by
``ml/data/md17.py::build_graph``. It therefore carries **no autograd
history back to positions**. A model that predicts energy from
``edge_attr`` alone and then asks autograd for ``-dE/dpos`` gets a tensor
of exact zeros - silently, with no error. With ``force_loss_weight: 100``
the force term would then be a constant the optimiser cannot move, and
the run would look like it trained while learning nothing about forces.

``predict_energy_and_forces`` therefore **recomputes** the radial basis
from ``pos`` (see ``rbf_from_positions``), making the whole energy a
differentiable function of coordinates. The passed ``edge_attr`` is
accepted for interface compatibility and ignored on that path.
``tests/test_physnet.py`` pins this down with a finite-difference check
against autograd, which is the validation ``ROADMAP.md`` week 3-4 calls
for.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

# Must stay identical to ml/data/md17.py::_gaussian_rbf - the graph the
# model differentiates has to be the graph the dataset built, or training
# and evaluation see different edge features. tests/test_physnet.py
# asserts the two agree numerically.
DEFAULT_NUM_RBF = 16
DEFAULT_CUTOFF = 5.0


def shifted_softplus(x: torch.Tensor) -> torch.Tensor:
    """PhysNet's activation: softplus(x) - ln(2), so ssp(0) = 0.

    Smooth and non-zero everywhere, unlike ReLU. That matters here beyond
    taste: forces are a *derivative* of the network output, so a kink in
    the activation becomes a discontinuity in the predicted force.
    """
    return nn.functional.softplus(x) - 0.6931471805599453


def rbf_from_positions(pos: torch.Tensor, edge_index: torch.Tensor,
                       num_rbf: int = DEFAULT_NUM_RBF,
                       cutoff: float = DEFAULT_CUTOFF) -> tuple[torch.Tensor, torch.Tensor]:
    """Gaussian-expands edge distances, differentiably w.r.t. ``pos``.

    Mirrors ml/data/md17.py::_gaussian_rbf exactly, but computes the
    distance from live coordinates instead of consuming a precomputed
    array, so gradients flow back to positions.

    Returns ``(rbf [n_edges, num_rbf], dist [n_edges])``.
    """
    row, col = edge_index[0], edge_index[1]
    vec = pos[row] - pos[col]
    # The +1e-12 keeps the gradient of sqrt finite if two atoms coincide.
    # Without it a coincident pair yields NaN forces for the whole batch.
    dist = torch.sqrt((vec * vec).sum(dim=-1) + 1e-12)

    centers = torch.linspace(0.0, cutoff, num_rbf, device=pos.device, dtype=pos.dtype)
    width = centers[1] - centers[0]
    rbf = torch.exp(-((dist.unsqueeze(-1) - centers) ** 2) / (2 * width ** 2))
    return rbf, dist


def cosine_cutoff(dist: torch.Tensor, cutoff: float = DEFAULT_CUTOFF) -> torch.Tensor:
    """Smooth 1 -> 0 envelope over [0, cutoff], zero beyond it.

    ``build_graph`` uses a hard distance cutoff to decide which pairs get
    an edge. Energy is then discontinuous at the boundary: an atom
    drifting across 5 A adds its full contribution instantly, so the force
    has a spike there. Multiplying every message by this envelope drives
    a message smoothly to zero as its edge reaches the cutoff, which is
    what makes the predicted forces continuous. PhysNet does the same.
    """
    return torch.where(
        dist < cutoff,
        0.5 * (torch.cos(math.pi * dist / cutoff) + 1.0),
        torch.zeros_like(dist),
    )


class ResidualBlock(nn.Module):
    """Pre-activation residual MLP - PhysNet's basic building unit."""

    def __init__(self, n_features: int):
        super().__init__()
        self.lin1 = nn.Linear(n_features, n_features)
        self.lin2 = nn.Linear(n_features, n_features)
        nn.init.zeros_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin1(shifted_softplus(x))
        h = self.lin2(shifted_softplus(h))
        return x + h


class InteractionBlock(nn.Module):
    """One round of message passing, PhysNet-style.

    Neighbour features are gated by a learned function of the radial
    basis and by the smooth cutoff, summed into the receiving atom, then
    mixed with a *gated* residual update: ``x <- u * x + f(v)`` with ``u``
    a learnable per-feature gate. The gate lets each feature channel
    choose how much of its previous state to keep, which is what allows
    several interaction blocks to stack without washing out the
    embedding.
    """

    def __init__(self, n_features: int, num_rbf: int, n_residual: int = 2):
        super().__init__()
        self.lin_self = nn.Linear(n_features, n_features)
        self.lin_neighbour = nn.Linear(n_features, n_features)
        # No bias: at the cutoff the RBF is ~0 and the message must vanish
        # with it. A bias would leak a distance-independent message.
        #
        # Deliberately NOT zero-initialised. Zeroing this weight makes the
        # initial atom features independent of position, so autograd
        # returns exactly zero forces from an untrained model - which is
        # indistinguishable from the stale-edge_attr bug the module
        # docstring describes, and would make
        # tests/test_physnet.py::test_forces_are_not_identically_zero pass
        # or fail for the wrong reason.
        self.lin_rbf = nn.Linear(num_rbf, n_features, bias=False)

        self.residuals = nn.ModuleList(ResidualBlock(n_features) for _ in range(n_residual))
        self.lin_out = nn.Linear(n_features, n_features)
        self.gate = nn.Parameter(torch.ones(n_features))

    def forward(self, x: torch.Tensor, rbf: torch.Tensor, cut: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index[0], edge_index[1]
        x_act = shifted_softplus(x)

        x_self = shifted_softplus(self.lin_self(x_act))
        x_neigh = shifted_softplus(self.lin_neighbour(x_act))

        # message_ij = neighbour feature * learned radial filter * envelope
        filt = self.lin_rbf(rbf) * cut.unsqueeze(-1)
        messages = x_neigh[col] * filt

        aggregated = torch.zeros_like(x_self)
        aggregated.index_add_(0, row, messages)

        v = x_self + aggregated
        for block in self.residuals:
            v = block(v)
        return self.gate * x + self.lin_out(shifted_softplus(v))


class OutputBlock(nn.Module):
    """Maps atom features to that module's atomic energy contribution."""

    def __init__(self, n_features: int, n_residual: int = 1):
        super().__init__()
        self.residuals = nn.ModuleList(ResidualBlock(n_features) for _ in range(n_residual))
        self.lin = nn.Linear(n_features, 1, bias=False)
        # Small but non-zero: the model starts close to the per-element
        # shift (so it isn't five orders of magnitude from the data)
        # while keeping the energy genuinely position-dependent from step
        # one. An exact zero here would make an untrained model report
        # zero forces, which is the bug this file is built to avoid.
        nn.init.normal_(self.lin.weight, std=1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.residuals:
            x = block(x)
        return self.lin(shifted_softplus(x)).squeeze(-1)


class PhysNet(nn.Module):
    """PhysNet short-range model: E = sum_i (scale[z_i] * E_i + shift[z_i]).

    Args:
        hidden_channels: feature width F.
        num_modules: stacked interaction+output modules. Each contributes
            additively to every atomic energy, so a shallow path to the
            output exists from the first module - PhysNet's answer to
            vanishing gradients in deep message passing.
        num_layers: interaction blocks per module.
        num_rbf, cutoff: must match ml/data/md17.py::build_graph.
        max_z: largest atomic number the embedding covers.
        atom_energy_shift: per-element constant added to each atomic
            contribution. **Set this from your training data.** MD17
            energies sit near -1e5 kcal/mol; a network initialised in the
            usual way outputs O(1), so without a shift the optimiser
            spends its early epochs travelling five orders of magnitude
            instead of learning chemistry. Pass a per-element mean, or
            use ``init_shift_from_energies``.
    """

    def __init__(self, hidden_channels: int = 128, num_modules: int = 3,
                 num_layers: int = 2, num_rbf: int = DEFAULT_NUM_RBF,
                 cutoff: float = DEFAULT_CUTOFF, max_z: int = 20,
                 num_residual_output: int = 1,
                 atom_energy_shift: Optional[torch.Tensor] = None):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.max_z = max_z

        # ml/training/evaluate.py probes `cutoff_radius` with getattr to
        # confirm a checkpoint is being scored at the radius it was
        # trained at. Expose that name as well as the short one, or the
        # check silently finds nothing and passes vacuously.
        self.cutoff_radius = cutoff
        self.num_modules = num_modules
        self.num_layers = num_layers
        self.num_residual_output = num_residual_output

        self.embedding = nn.Embedding(max_z + 1, hidden_channels)

        self.modules_interaction = nn.ModuleList(
            nn.ModuleList(InteractionBlock(hidden_channels, num_rbf) for _ in range(num_layers))
            for _ in range(num_modules)
        )
        self.modules_output = nn.ModuleList(
            OutputBlock(hidden_channels, num_residual_output) for _ in range(num_modules)
        )

        self.atom_scale = nn.Parameter(torch.ones(max_z + 1))
        shift = torch.zeros(max_z + 1) if atom_energy_shift is None else atom_energy_shift.clone()
        self.atom_shift = nn.Parameter(shift)

    def config(self) -> dict:
        """Constructor kwargs, for ml/models/registry.py::save_checkpoint.

        Part of the registry's REQUIRED_INTERFACE. A checkpoint carries
        this so evaluation rebuilds the exact model that wrote it instead
        of guessing hyperparameters from the config file that happens to
        be lying around at scoring time.
        """
        return {
            "hidden_channels": self.hidden_channels,
            "num_modules": self.num_modules,
            "num_layers": self.num_layers,
            "num_rbf": self.num_rbf,
            "cutoff": self.cutoff,
            "max_z": self.max_z,
            "num_residual_output": self.num_residual_output,
        }

    @torch.no_grad()
    def init_shift_from_energies(self, z: torch.Tensor, energies: torch.Tensor) -> None:
        """Sets the per-element shift so the untrained model already
        predicts roughly the right magnitude.

        Splits the mean total energy evenly across the atoms of one
        molecule. Crude - it gives every element in the molecule the same
        constant rather than a true atomic reference - but it removes the
        five-orders-of-magnitude offset, which is the part that actually
        stalls training. The shift stays learnable from there.
        """
        z = z.reshape(-1).long()
        per_atom = float(energies.reshape(-1).mean()) / max(z.numel(), 1)
        self.atom_shift.data.zero_()
        self.atom_shift.data[torch.unique(z)] = per_atom

    def _atomic_energies(self, z: torch.Tensor, rbf: torch.Tensor, cut: torch.Tensor,
                         edge_index: torch.Tensor) -> torch.Tensor:
        x = self.embedding(z)
        atomic = torch.zeros(z.shape[0], device=z.device, dtype=rbf.dtype)
        for interactions, output in zip(self.modules_interaction, self.modules_output):
            for block in interactions:
                x = block(x, rbf, cut, edge_index)
            atomic = atomic + output(x)
        return self.atom_scale[z] * atomic + self.atom_shift[z]

    @staticmethod
    def _sum_per_graph(atomic: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
        if batch is None:
            return atomic.sum().reshape(1)
        n_graphs = int(batch.max().item()) + 1
        out = torch.zeros(n_graphs, device=atomic.device, dtype=atomic.dtype)
        out.index_add_(0, batch, atomic)
        return out

    def predict_energy(self, z: torch.Tensor, *args) -> torch.Tensor:
        """Energy only, from the precomputed edge features.

        Accepts **both** call shapes currently in circulation, because
        the two specs disagree and this model has to satisfy whichever
        the caller uses:

            predict_energy(z, edge_index, edge_attr, batch)        # SCRUM-50 ticket
            predict_energy(z, pos, edge_index, edge_attr, batch)   # registry docstring

        They are told apart by the second argument: ``pos`` is floating
        point of shape [N, 3], ``edge_index`` is integer of shape [2, E].
        This is a compatibility shim, not a design - **the team should
        settle on one signature** and it should be deleted. Raised in the
        model card's known-risks section.

        Force extraction is not supported from this path: ``edge_attr``
        has no gradient path to positions, so ``-dE/dpos`` taken from
        this output is exactly zero. Use ``predict_energy_and_forces``
        whenever forces matter.
        """
        if args and torch.is_floating_point(args[0]) and args[0].dim() == 2 and args[0].shape[1] == 3:
            _pos, edge_index, edge_attr, *rest = args      # registry form
        else:
            edge_index, edge_attr, *rest = args            # ticket form
        batch = rest[0] if rest else None

        z = z.reshape(-1).long()
        cut = torch.ones(edge_index.shape[1], device=edge_attr.device, dtype=edge_attr.dtype)
        atomic = self._atomic_energies(z, edge_attr, cut, edge_index)
        return self._sum_per_graph(atomic, batch)

    def predict_energy_and_forces(self, z: torch.Tensor, pos: torch.Tensor,
                                  edge_index: torch.Tensor,
                                  edge_attr: Optional[torch.Tensor] = None,
                                  batch: Optional[torch.Tensor] = None
                                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """Energy and forces, with forces as ``-dE/dpos`` via autograd.

        ``edge_attr`` is accepted for interface compatibility and
        ignored: the radial basis is recomputed from ``pos`` so the
        energy is genuinely a function of coordinates. See the module
        docstring for why using the passed value here yields zero forces.

        Works under ``torch.no_grad()`` for evaluation - grad mode is
        re-enabled internally, since forces are part of the prediction,
        not part of the training bookkeeping.
        """
        z = z.reshape(-1).long()
        with torch.enable_grad():
            if not pos.requires_grad:
                # detach() first: under torch.no_grad() at eval time pos may
                # be a non-leaf, and requires_grad_() on a non-leaf raises.
                pos = pos.detach().requires_grad_(True)
            rbf, dist = rbf_from_positions(pos, edge_index, self.num_rbf, self.cutoff)
            cut = cosine_cutoff(dist, self.cutoff)
            atomic = self._atomic_energies(z, rbf, cut, edge_index)
            energy = self._sum_per_graph(atomic, batch)

            # create_graph=True only while training, so the force term is
            # itself differentiable for the backward pass. At eval that
            # would retain a graph nobody consumes - a slow memory leak
            # across a 55,509-config test set.
            grad = torch.autograd.grad(
                energy.sum(), pos, create_graph=self.training,
            )[0]
        forces = -grad
        return (energy, forces) if self.training else (energy.detach(), forces.detach())

    def forward(self, z: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (f"PhysNet(hidden_channels={self.hidden_channels}, "
                f"num_modules={len(self.modules_output)}, "
                f"num_rbf={self.num_rbf}, cutoff={self.cutoff}, "
                f"params={n_params:,})")


def build_model(config) -> PhysNet:
    """Builds a PhysNet from an experiment config (glossary.md section 6).

    Accepts either an object with attributes or a plain dict, so it works
    against ml/config.py once that lands without needing it now.
    """
    def get(section, key, default):
        node = config.get(section, {}) if isinstance(config, dict) else getattr(config, section, None)
        if node is None:
            return default
        if isinstance(node, dict):
            return node.get(key, default)
        return getattr(node, key, default)

    return PhysNet(
        hidden_channels=get("model", "hidden_channels", 128),
        num_modules=get("model", "num_modules", 3),
        num_layers=get("model", "num_layers", 2),
        num_rbf=get("model", "num_rbf", DEFAULT_NUM_RBF),
        cutoff=get("data", "cutoff_radius", DEFAULT_CUTOFF),
    )
