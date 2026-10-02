"""PaiNN (Polarizable Atom Interaction Neural Network) for the Phase 1 bake-off (SCRUM-52).

Schutt, Unke & Gastegger (2021), *Equivariant message passing for the
prediction of tensorial properties and molecular spectra*.

Each atom carries a **pair** of features: a rotation-invariant scalar
`s_i` (F channels) and a rotation-**equivariant** vector `v_i` (3 x F
channels). Message passing mixes the two, which lets the model represent
directional information that a distance-only MPNN (Phase 1's Gilmer
backbone) cannot, while keeping the predicted energy exactly invariant to
rotation, translation and atom reordering.

Why the energy is invariant and the forces equivariant *by construction*
rather than by training - the four rules every line below obeys:

1. **Vectors reach scalars only through rotation-invariant contractions**
   - the norm ``||v||`` and the inner product ``<u, v>``. Both are
   unchanged by a rotation, so nothing a vector contributes to `s` can
   depend on the orientation of the molecule.
2. **Vectors are only ever produced by scaling a vector by a scalar** -
   either an existing `v_j`, or the unit direction ``r_hat_ij``. Since
   ``R (a * v) == a * (R v)``, equivariance survives every such step.
3. **No bias term on any linear map applied to a vector channel**
   (`_UpdateBlock.U` / `.V` are ``bias=False``). A bias adds a *fixed*
   vector, and ``R(Wv + b) != R(Wv) + b`` unless ``b == 0``. This is the
   classic silent equivariance break: the model still trains and still
   produces plausible MAEs. `tests/ml/test_painn.py` is what catches it.
4. **Only relative displacements** ``r_ij = pos_j - pos_i`` are ever read,
   never an absolute coordinate. Translation invariance is therefore
   structural, not learned.

Sum-aggregation over neighbours gives permutation equivariance, and a
sum-pooled per-atom energy gives a size-extensive total energy.

**This module deliberately ignores the `edge_attr` supplied by
`MD17Dataset`** - see `predict_energy` for the full reason. It recomputes
its own radial basis from `pos` so that ``F = -dE/dR`` reaches atomic
positions through *every* path in the energy expression.

Depends on `torch` only - not on `torch_geometric` - so the equivariance
tests exercise the model with no graph-library machinery in between.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

__all__ = ["PaiNN"]

# Large enough for every element in `_PERIODIC_TABLE` (ml/data/md17.py tops
# out at Cl = 17) with headroom, and small enough that the unused rows cost
# nothing that matters. Indexing is by atomic number directly, so row 0 is
# never used - `_symbol_to_z` raises rather than emitting 0.
DEFAULT_MAX_ATOMIC_NUMBER = 100


def _safe_norm(x: Tensor, dim: int, eps: float) -> Tensor:
    """``||x||`` with a gradient that is finite *and continuous* at the origin.

    ``d||x||/dx = x / ||x||`` is undefined at ``x = 0``, and both obvious
    spellings misbehave there (measured on torch 2.13):

    - ``torch.sqrt(torch.sum(x * x))`` returns a **NaN** gradient at
      exactly zero, which silently poisons every downstream parameter.
    - ``torch.linalg.norm`` special-cases the origin and returns a zero
      gradient, so it produces no NaN - but one ulp away the gradient is
      already 1.0. That discontinuity is worse than it looks: forces are
      this gradient, so the force field acquires a jump exactly where the
      vector features vanish.

    Both cases are reachable. ``v_i`` is exactly zero for an atom with no
    neighbours inside the cutoff, and - more commonly - wherever the
    neighbours' ``r_hat`` contributions cancel, which is every
    centrosymmetric environment. `tests/ml/test_painn.py` pins the
    symmetric case: the central atom of a symmetric pair must feel exactly
    zero force, and `torch.linalg.norm` gets that wrong.

    ``sqrt(sum(x^2) + eps)`` is smooth everywhere. It reads ``sqrt(eps)``
    at the origin (1e-4 for the default eps) and is biased by roughly
    ``eps / (2||x||)`` elsewhere, both negligible against the O(1) norms
    this sees. It does **not** affect equivariance at any `eps`:
    ``sum(x^2)`` is rotation-invariant, so the perturbed norm is too, and
    ``x / safe_norm(x)`` stays exactly equivariant.
    """
    return torch.sqrt(torch.sum(x * x, dim=dim) + eps)


class _BesselBasis(nn.Module):
    """Sinc/Bessel radial basis with a cosine cutoff envelope (PaiNN appendix A).

    ``g_n(d) = sqrt(2/r_c) * sin(n*pi*d / r_c) / d`` for n = 1..num_rbf.

    This is PaiNN's *own* basis, not the 16-bin Gaussian expansion that
    `ml/data/md17.py::_gaussian_rbf` puts in `edge_attr`. The radial basis
    is part of the architecture under test, so the bake-off compares it as
    published; the two are configured independently (`model.num_rbf` here
    vs `data.num_rbf` for the dataset). See `predict_energy` for why the
    dataset's expansion cannot be reused regardless.
    """

    def __init__(self, num_rbf: int, cutoff: float, eps: float) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.eps = float(eps)
        # A float buffer, so `.double()` / `.to(device)` carry it along with
        # the parameters and it lands in the checkpoint's state_dict.
        self.register_buffer(
            "freqs", torch.arange(1, num_rbf + 1, dtype=torch.float32) * math.pi
        )

    def forward(self, dist: Tensor) -> tuple[Tensor, Tensor]:
        """(E,) distances -> ((E, num_rbf) basis, (E, 1) cutoff envelope)."""
        d = dist.unsqueeze(-1)                                    # (E, 1)
        basis = math.sqrt(2.0 / self.cutoff) * torch.sin(self.freqs * d / self.cutoff) / d
        # Smoothly drives every message to zero at the cutoff, so an atom
        # entering or leaving a neighbour list does not step-change the
        # energy - a discontinuity there would make the force field
        # non-conservative exactly where MD needs it most.
        envelope = 0.5 * (torch.cos(math.pi * d / self.cutoff) + 1.0)
        # Defensive: `build_graph` already applies the same cutoff, but a
        # graph built at a wider radius must not contribute past it.
        return basis, envelope * (d < self.cutoff)


class _MessageBlock(nn.Module):
    """Interatomic (equivariant) message passing: neighbours update (s_i, v_i).

        ds_i = sum_j  m_ss
        dv_i = sum_j [ v_j * m_vv  +  m_vs * r_hat_ij ]

    where ``(m_ss, m_vv, m_vs) = split(phi(s_j) * W(r_ij))``. The scalar
    path is an ordinary continuous-filter convolution; the vector path adds
    the two - and only the two - equivariant terms available: rescale the
    neighbour's existing vector, and emit a new vector along the bond
    direction.
    """

    def __init__(self, hidden: int, num_rbf: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.scalar_net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3 * hidden),
        )
        # Maps the radial basis to per-channel filter weights. Biasless,
        # following the reference implementation, so the filter is a pure
        # function of the radial basis and carries no distance-independent
        # floor. Note this is *not* what makes the potential continuous at
        # the cutoff - `envelope` multiplies this product afterwards and
        # would zero a bias too. Continuity is the envelope's job alone.
        self.filter_net = nn.Linear(num_rbf, 3 * hidden, bias=False)

    def forward(
        self,
        s: Tensor,            # (N, F)
        v: Tensor,            # (N, 3, F)
        edge_index: Tensor,   # (2, E)
        r_hat: Tensor,        # (E, 3)
        basis: Tensor,        # (E, num_rbf)
        envelope: Tensor,     # (E, 1)
    ) -> tuple[Tensor, Tensor]:
        # `build_graph` emits a symmetric edge set, so both (i, j) and
        # (j, i) are present. Fixing one convention still matters: mixing
        # them would pair a neighbour's features with the wrong direction
        # vector. Here column 0 is the neighbour j, column 1 the centre i
        # receiving the message (PyG's src -> dst convention).
        j, i = edge_index[0], edge_index[1]

        weights = self.filter_net(basis) * envelope               # (E, 3F)
        messages = self.scalar_net(s)[j] * weights                # (E, 3F)
        m_ss, m_vv, m_vs = torch.split(messages, self.hidden, dim=-1)

        # (E, 1, F) * (E, 3, F) rescales the neighbour's vector channel-wise;
        # (E, 3, 1) * (E, 1, F) broadcasts the bond direction into F channels.
        dv = v[j] * m_vv.unsqueeze(1) + r_hat.unsqueeze(-1) * m_vs.unsqueeze(1)

        ds_sum = torch.zeros_like(s).index_add(0, i, m_ss)
        dv_sum = torch.zeros_like(v).index_add(0, i, dv)
        return ds_sum, dv_sum


class _UpdateBlock(nn.Module):
    """Per-atom (no neighbours) mixing of the scalar and vector channels.

        dv_i = a_vv * U v_i
        ds_i = a_ss + a_sv * <U v_i, V v_i>

    with ``(a_vv, a_sv, a_ss) = MLP([s_i, ||V v_i||])``.

    `U` and `V` are linear maps over the **channel** axis of `v` only -
    they never mix the three spatial components - so ``U(Rv) == R(Uv)``.
    Both are `bias=False`; see rule 3 in the module docstring.
    """

    def __init__(self, hidden: int, eps: float) -> None:
        super().__init__()
        self.hidden = hidden
        self.eps = eps
        self.U = nn.Linear(hidden, hidden, bias=False)
        self.V = nn.Linear(hidden, hidden, bias=False)
        self.scalar_net = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3 * hidden),
        )

    def forward(self, s: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        # nn.Linear acts on the last axis, so (N, 3, F) -> (N, 3, F) with the
        # spatial axis untouched. That is the whole reason this is equivariant.
        Uv = self.U(v)
        Vv = self.V(v)

        # The only two ways a vector is allowed to influence a scalar.
        Vv_norm = _safe_norm(Vv, dim=1, eps=self.eps)             # (N, F)
        a = self.scalar_net(torch.cat([s, Vv_norm], dim=-1))      # (N, 3F)
        a_vv, a_sv, a_ss = torch.split(a, self.hidden, dim=-1)

        dv = a_vv.unsqueeze(1) * Uv                               # (N, 3, F)
        ds = a_ss + a_sv * torch.sum(Uv * Vv, dim=1)              # (N, F)
        return ds, dv


class PaiNN(nn.Module):
    """Equivariant MPNN predicting total energy, with forces from autograd.

    Deliberately has **no force head**. Forces are ``F = -dE/dR`` taken
    through `pos`, which makes the model a conservative potential: the
    force field is the exact gradient of a scalar energy, so energy is
    conserved along an MD trajectory. A separate force output would be
    cheaper but would not integrate stably, and would break the very
    property this bake-off is measuring.

    Args:
        hidden_channels: width F of both the scalar and vector channels.
        num_layers: message+update blocks. PaiNN uses 3 for MD17.
        num_rbf: Bessel basis functions (this model's own radial
            expansion - independent of the dataset's `data.num_rbf`).
        cutoff_radius: must match the radius the neighbour graph was built
            with (`data.cutoff_radius`), or messages are silently dropped
            by the cutoff envelope.
        max_atomic_number: embedding table size; indexed by `z` directly.
        eps: floor inside every vector norm, see `_safe_norm`.

    `energy_shift` / `energy_scale` are buffers, not parameters: MD17
    total energies sit around -97,000 kcal/mol for ethanol, which no
    freshly-initialised readout can reach. The training loop is expected
    to set them from the *training* split's statistics before the first
    step. They are buffers so they travel in the checkpoint's state_dict
    and evaluation reproduces training's scale exactly.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 3,
        num_rbf: int = 20,
        cutoff_radius: float = 5.0,
        max_atomic_number: int = DEFAULT_MAX_ATOMIC_NUMBER,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if hidden_channels < 2:
            raise ValueError(f"hidden_channels must be >= 2, got {hidden_channels}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff_radius = float(cutoff_radius)
        self.max_atomic_number = max_atomic_number
        self.eps = float(eps)

        self.embedding = nn.Embedding(max_atomic_number + 1, hidden_channels)
        self.radial_basis = _BesselBasis(num_rbf, cutoff_radius, eps)
        self.message_blocks = nn.ModuleList(
            _MessageBlock(hidden_channels, num_rbf) for _ in range(num_layers)
        )
        self.update_blocks = nn.ModuleList(
            _UpdateBlock(hidden_channels, eps) for _ in range(num_layers)
        )
        # Reads the scalar channel only: a per-atom energy must be a scalar,
        # and routing the vector channel here would break rotation invariance.
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )
        self.register_buffer("energy_shift", torch.zeros(1))
        self.register_buffer("energy_scale", torch.ones(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier on every weight, zero on every bias (PaiNN's initialisation)."""
        nn.init.xavier_uniform_(self.embedding.weight)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def config(self) -> dict:
        """Constructor kwargs, for embedding in a checkpoint. See ml/models/registry.py."""
        return {
            "hidden_channels": self.hidden_channels,
            "num_layers": self.num_layers,
            "num_rbf": self.num_rbf,
            "cutoff_radius": self.cutoff_radius,
            "max_atomic_number": self.max_atomic_number,
            "eps": self.eps,
        }

    def set_energy_statistics(self, shift: float, scale: float) -> None:
        """Set the per-atom energy shift/scale from *training*-split statistics.

        Called by the training loop before the first step; never from
        evaluation, which must reproduce the checkpoint's values exactly.
        """
        if scale <= 0:
            raise ValueError(f"energy_scale must be positive, got {scale}")
        with torch.no_grad():
            self.energy_shift.fill_(float(shift))
            self.energy_scale.fill_(float(scale))

    # ------------------------------------------------------------------
    # Public interface - shared by every model in the Phase 1 bake-off
    # ------------------------------------------------------------------

    def predict_energy(
        self,
        z: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        batch: Tensor,
    ) -> Tensor:
        """Total potential energy per molecule. Returns (B,), in the data's energy unit.

        **Why `pos` is a required argument** (it is absent from the
        original SCRUM-52 signature): interatomic distances *must* be
        recomputed from `pos` inside the forward pass, so that autograd can
        reach atomic positions. `MD17Dataset.get()` builds `edge_attr`
        through `pos.numpy()`, which severs the graph - the tensor arrives
        with ``requires_grad=False`` and ``grad_fn=None``. A model that
        takes its radial features from `edge_attr` therefore computes

            -dE/dpos   instead of   -(dE/dpos + dE/drbf * drbf/dpos)

        and ``F = -dE/dR`` is silently wrong: the model still trains, still
        produces believable MAEs, and is no longer a conservative
        potential for the energy it reports. A conservative-field test does
        not catch it either, because both sides of that comparison inherit
        the same detached tensor. This applies to every conservative model
        in the bake-off, not only PaiNN.

        PaiNN additionally *cannot* run without `pos` at all: its vector
        channels are constructed from bond directions ``r_hat_ij``, which
        the rotation-invariant `edge_attr` does not carry.

        Args:
            z: (N,) or (N, 1) int64 atomic numbers. `MD17Dataset` supplies
                these as `Data.x` with shape (N, 1); both are accepted.
            pos: (N, 3) float atomic coordinates, in Angstrom.
            edge_index: (2, E) int64, row 0 = neighbour j, row 1 = centre i.
            edge_attr: accepted for interface compatibility with the rest
                of the bake-off and **deliberately unused** - only its
                shape is checked. See above.
            batch: (N,) int64, non-decreasing, values 0..B-1.
        """
        z = self._normalise_z(z, pos)
        self._check_shapes(z, pos, edge_index, edge_attr, batch)

        j, i = edge_index[0], edge_index[1]
        # Relative displacement only - never an absolute coordinate. This
        # single line is the whole of the model's translation invariance.
        r_vec = pos[j] - pos[i]                                   # (E, 3)
        dist = _safe_norm(r_vec, dim=-1, eps=self.eps)            # (E,)
        r_hat = r_vec / dist.unsqueeze(-1)                        # (E, 3)
        basis, envelope = self.radial_basis(dist)

        s = self.embedding(z)                                     # (N, F)
        # Vectors start at exactly zero: before any message passing an atom
        # has no directional information, and zero is the only value that is
        # itself rotation-invariant. `_safe_norm` is what keeps this finite.
        v = torch.zeros(s.shape[0], 3, self.hidden_channels, dtype=s.dtype, device=s.device)

        for message, update in zip(self.message_blocks, self.update_blocks):
            ds, dv = message(s, v, edge_index, r_hat, basis, envelope)
            s = s + ds
            v = v + dv
            ds, dv = update(s, v)
            s = s + ds
            v = v + dv

        atomic_energy = self.readout(s).squeeze(-1)               # (N,)
        atomic_energy = atomic_energy * self.energy_scale + self.energy_shift

        num_graphs = int(batch.max()) + 1
        energy = torch.zeros(num_graphs, dtype=atomic_energy.dtype, device=atomic_energy.device)
        # Sum-pooling, not mean: total energy is size-extensive. `contract.py`
        # requires shape (B,) - a (B, 1) result would broadcast to (B, B)
        # against the reference and print a plausible wrong number.
        return energy.index_add(0, batch, atomic_energy)

    def predict_energy_and_forces(
        self,
        z: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        batch: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Energy (B,) and forces (N, 3) via ``F = -dE/dR``.

        `pos` is required for the reason given on `predict_energy`: the
        gradient has to reach atomic positions through every path in the
        energy expression, which only holds if distances are recomputed
        here rather than read from a detached `edge_attr`.

        If the caller's `pos` already has ``requires_grad=True`` the
        gradient flows back to *that* tensor, so a training loop can
        differentiate further through it. Otherwise a grad-enabled copy is
        made, which lets evaluation call this under `torch.no_grad()`.

        ``create_graph`` follows `self.training`: training needs the
        double-backward path so the force loss can be optimised, while
        evaluation must not retain it or the eval loop leaks memory.
        """
        # enable_grad overrides an enclosing no_grad, so evaluation code can
        # keep its usual `with torch.no_grad():` wrapper around the loop.
        with torch.enable_grad():
            if not pos.requires_grad:
                pos = pos.detach().requires_grad_(True)
            energy = self.predict_energy(z, pos, edge_index, edge_attr, batch)
            # energy.sum() is safe across a batch: an atom belongs to exactly
            # one molecule, so d(sum_b E_b)/d pos_a == d E_{mol(a)} / d pos_a.
            # No per-molecule loop, and no cross-molecule contamination.
            (grad,) = torch.autograd.grad(
                energy.sum(), pos, create_graph=self.training,
            )
        return energy, -grad

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Alias for `predict_energy_and_forces` - the model's primary output."""
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        return self.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)

    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_z(z: Tensor, pos: Tensor) -> Tensor:
        """Accept `MD17Dataset`'s (N, 1) `Data.x` as well as a flat (N,)."""
        if z.dim() == 2 and z.shape[1] == 1:
            z = z.squeeze(-1)
        if z.dim() != 1:
            raise ValueError(
                f"z must have shape (N,) or (N, 1), got {tuple(z.shape)}. "
                "MD17Dataset supplies atomic numbers as Data.x with shape (N, 1)."
            )
        if z.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"z must be an integer tensor of atomic numbers, got {z.dtype}. "
                "It indexes an embedding table directly."
            )
        if z.shape[0] != pos.shape[0]:
            raise ValueError(f"z has {z.shape[0]} atoms but pos has {pos.shape[0]}")
        return z.long()

    def _check_shapes(
        self,
        z: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        batch: Tensor,
    ) -> None:
        if pos.dim() != 2 or pos.shape[1] != 3:
            raise ValueError(f"pos must have shape (N, 3), got {tuple(pos.shape)}")
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}")
        if batch.dim() != 1 or batch.shape[0] != pos.shape[0]:
            raise ValueError(
                f"batch must have shape (N,) = ({pos.shape[0]},), got {tuple(batch.shape)}"
            )
        if edge_attr is not None and edge_attr.shape[0] != edge_index.shape[1]:
            # edge_attr is unused, but a mismatch means the caller has paired
            # a graph with the wrong features - worth failing on even here.
            raise ValueError(
                f"edge_attr has {edge_attr.shape[0]} rows but edge_index has "
                f"{edge_index.shape[1]} edges"
            )
        if int(self.max_atomic_number) < int(z.max()):
            raise ValueError(
                f"atomic number {int(z.max())} exceeds max_atomic_number "
                f"{self.max_atomic_number}; raise it in the model config"
            )
