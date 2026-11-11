"""Correctness tests for ml/models/painn.py (SCRUM-52).

PaiNN's entire reason for being in the Phase 1 bake-off is equivariance,
and a subtly broken equivariant model still trains, still converges, and
still produces plausible-looking MAEs. Accuracy numbers cannot detect
that; only these symmetry assertions can. Each test states the symmetry it
pins and what breaking it would look like in a results table.

**Everything runs in float64.** At float32 a rotation test needs a
tolerance around 1e-5, which is loose enough to pass a model whose
equivariance is broken at the 1e-4 level - precisely the "trains fine,
looks plausible" failure these tests exist to catch. float64 puts the
roundoff floor around 1e-12, so `TOL = 1e-10` is a real assertion rather
than a formality. `_safe_norm`'s `eps` does not interfere: `sum(x^2)` is
rotation-invariant for any `eps`, so the perturbed norm is too.

Two structural choices worth stating, because they isolate what is under
test:

- **The same `edge_index` is reused** across a rotation/translation rather
  than rebuilding the graph. Rotation and translation preserve every
  interatomic distance, so the neighbour list is unchanged by
  construction; rebuilding it would add graph-construction roundoff at the
  cutoff boundary to a result that is meant to measure the model alone.
- **Weights are randomised, not trained.** Equivariance is a structural
  property of the architecture. If it only holds after training, it does
  not hold.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ml.data.md17 import build_graph
from ml.models.painn import PaiNN

# float64 roundoff through 2 message+update blocks lands around 1e-12;
# 1e-10 leaves two orders of headroom without being a rubber stamp.
TOL = 1e-10

# Small but not degenerate: 3 elements so the embedding table is actually
# exercised, 9 atoms to match ethanol, and a spread that puts some pairs
# inside the 5 A cutoff and some outside.
_Z = [6, 6, 8, 1, 1, 1, 1, 1, 1]
CUTOFF = 5.0


def make_model(seed: int = 0, **kwargs) -> PaiNN:
    """A small randomly-initialised PaiNN in float64, in eval mode."""
    torch.manual_seed(seed)
    params = dict(hidden_channels=32, num_layers=2, num_rbf=8, cutoff_radius=CUTOFF)
    params.update(kwargs)
    model = PaiNN(**params).double()
    # A non-trivial shift/scale, so any test that passes only because the
    # readout output is near zero fails instead.
    model.set_energy_statistics(shift=-7.5, scale=2.25)
    model.eval()
    return model


def make_molecule(seed: int = 1, n_atoms: int = len(_Z)):
    """One random molecule as (z, pos, edge_index, edge_attr, batch), float64.

    The graph comes from the production `build_graph`, so the edge
    convention and cutoff here are the ones `MD17Dataset` actually uses.
    """
    generator = torch.Generator().manual_seed(seed)
    pos = torch.randn(n_atoms, 3, generator=generator, dtype=torch.float64) * 1.4
    edge_index, edge_attr = build_graph(pos.numpy(), cutoff=CUTOFF, num_rbf=16)
    z = torch.tensor(_Z[:n_atoms], dtype=torch.int64).view(-1, 1)
    batch = torch.zeros(n_atoms, dtype=torch.int64)
    return z, pos, edge_index, edge_attr.double(), batch


def random_rotation(seed: int) -> torch.Tensor:
    """A uniformly random proper rotation in SO(3) (det = +1, not a reflection)."""
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    # QR's sign convention is not unique; pin it, then force det = +1 so this
    # is a rotation rather than a rotoreflection.
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


# --------------------------------------------------------------------------
# 1. Rotation: E invariant, F equivariant
# --------------------------------------------------------------------------

def test_rotation_leaves_energy_unchanged_and_rotates_forces():
    # `pos` holds row vectors, so rotating every atom by R is `pos @ R.T`
    # (equivalently, each row r -> R r in column convention). Forces are
    # vectors attached to atoms, so they must transform the same way.
    #
    # Breaking this is the classic `bias=True` on a vector-channel linear
    # map: the model still trains, but its forces point in orientation-
    # dependent directions and an MD trajectory driven by it drifts.
    model = make_model()
    z, pos, edge_index, edge_attr, batch = make_molecule()
    rotation = random_rotation(seed=7)

    energy, forces = model.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)
    energy_rot, forces_rot = model.predict_energy_and_forces(
        z, pos @ rotation.T, edge_index, edge_attr, batch
    )

    assert torch.allclose(energy_rot, energy, atol=TOL, rtol=0)
    assert torch.allclose(forces_rot, forces @ rotation.T, atol=TOL, rtol=0)

    # Guard against a vacuous pass: a model that ignored geometry entirely
    # would satisfy the assertions above with zero forces.
    assert forces.abs().max() > 1e-6


def test_rotation_holds_for_many_rotations():
    """One lucky rotation is not evidence; sweep several."""
    model = make_model()
    z, pos, edge_index, edge_attr, batch = make_molecule()
    energy, forces = model.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)

    for seed in range(5):
        rotation = random_rotation(seed=100 + seed)
        energy_rot, forces_rot = model.predict_energy_and_forces(
            z, pos @ rotation.T, edge_index, edge_attr, batch
        )
        assert torch.allclose(energy_rot, energy, atol=TOL, rtol=0), f"seed {seed}"
        assert torch.allclose(forces_rot, forces @ rotation.T, atol=TOL, rtol=0), f"seed {seed}"


# --------------------------------------------------------------------------
# 2. Translation: E and F both unchanged
# --------------------------------------------------------------------------

def test_translation_leaves_energy_and_forces_unchanged():
    # Only relative displacements `pos_j - pos_i` are read, so a rigid
    # shift must cancel exactly. A model that used an absolute coordinate
    # anywhere would make the energy depend on where the molecule sits in
    # the simulation box, which is physically meaningless.
    #
    # The shift is large relative to the 5 A cutoff: a small one could be
    # absorbed by float64 slack and pass a model that is only approximately
    # translation-invariant.
    model = make_model()
    z, pos, edge_index, edge_attr, batch = make_molecule()
    shift = torch.tensor([13.5, -7.25, 91.0], dtype=torch.float64)

    energy, forces = model.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)
    energy_shifted, forces_shifted = model.predict_energy_and_forces(
        z, pos + shift, edge_index, edge_attr, batch
    )

    assert torch.allclose(energy_shifted, energy, atol=TOL, rtol=0)
    assert torch.allclose(forces_shifted, forces, atol=TOL, rtol=0)


# --------------------------------------------------------------------------
# 3. Permutation: E invariant, F permutes with the atoms
# --------------------------------------------------------------------------

def test_permuting_atoms_leaves_energy_unchanged_and_permutes_forces():
    # Atom ordering is an artefact of the file format, not physics. Sum
    # aggregation over neighbours is what makes this hold; a mean or a
    # max-with-ties, or an aggregation that indexed the wrong edge column,
    # would break it.
    model = make_model()
    z, pos, edge_index, edge_attr, batch = make_molecule()
    n_atoms = pos.shape[0]

    generator = torch.Generator().manual_seed(3)
    perm = torch.randperm(n_atoms, generator=generator)
    assert not torch.equal(perm, torch.arange(n_atoms)), "permutation must not be the identity"

    # `perm` says new atom k is old atom perm[k]. Old index a therefore sits
    # at new index inverse[a], and every edge endpoint is remapped through it.
    inverse = torch.argsort(perm)
    edge_index_perm = inverse[edge_index]

    energy, forces = model.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)
    energy_perm, forces_perm = model.predict_energy_and_forces(
        z[perm], pos[perm], edge_index_perm, edge_attr, batch
    )

    assert torch.allclose(energy_perm, energy, atol=TOL, rtol=0)
    assert torch.allclose(forces_perm, forces[perm], atol=TOL, rtol=0)


# --------------------------------------------------------------------------
# 4. Conservative field: F is exactly -dE/dR
# --------------------------------------------------------------------------

def test_forces_equal_negative_gradient_of_energy():
    # `predict_energy_and_forces` must not be a second head that merely
    # correlates with the gradient - it must *be* the gradient, or the
    # potential is not energy-conserving and an MD run heats up.
    #
    # The gradient is taken here independently, from `predict_energy`, and
    # compared against what the model returned.
    model = make_model()
    z, pos, edge_index, edge_attr, batch = make_molecule()

    pos_grad = pos.clone().requires_grad_(True)
    energy = model.predict_energy(z, pos_grad, edge_index, edge_attr, batch)
    (gradient,) = torch.autograd.grad(energy.sum(), pos_grad)

    _, forces = model.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)

    assert torch.allclose(forces, -gradient, atol=TOL, rtol=0)
    assert forces.abs().max() > 1e-6


def test_energy_agrees_between_the_two_entry_points():
    """`predict_energy` and `predict_energy_and_forces` must not drift apart."""
    model = make_model()
    z, pos, edge_index, edge_attr, batch = make_molecule()

    energy_only = model.predict_energy(z, pos, edge_index, edge_attr, batch)
    energy_with_forces, _ = model.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)

    assert torch.allclose(energy_only, energy_with_forces, atol=TOL, rtol=0)


def test_finite_difference_matches_autograd_forces():
    """Independent check that autograd is differentiating the right thing.

    ROADMAP.md's Phase 1 definition of done asks for exactly this: forces
    "verified against a finite-difference check of the energy gradient, not
    just 'the model runs'". Autograd and central differences are entirely
    separate code paths, so agreement between them is real evidence.

    Central differences are O(h^2) accurate; at h = 1e-5 in float64 the
    truncation and roundoff terms balance around 1e-9, hence the looser
    tolerance here than elsewhere in this file.
    """
    model = make_model()
    z, pos, edge_index, edge_attr, batch = make_molecule()
    _, forces = model.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)

    h = 1e-5
    numerical = torch.zeros_like(pos)
    for atom in range(pos.shape[0]):
        for axis in range(3):
            plus, minus = pos.clone(), pos.clone()
            plus[atom, axis] += h
            minus[atom, axis] -= h
            e_plus = model.predict_energy(z, plus, edge_index, edge_attr, batch)
            e_minus = model.predict_energy(z, minus, edge_index, edge_attr, batch)
            numerical[atom, axis] = -(e_plus - e_minus).item() / (2 * h)

    assert torch.allclose(forces, numerical, atol=1e-8, rtol=1e-6)


# --------------------------------------------------------------------------
# 5. Batching: N molecules at once == N separate passes
# --------------------------------------------------------------------------

def test_batched_results_match_single_molecule_passes():
    # This is the test that catches batch-index bugs, which are otherwise
    # completely silent: a wrong `index_add` target or a `batch` vector read
    # in the wrong place still returns correctly-shaped, finite, plausible
    # energies - it just mixes atoms between molecules.
    model = make_model()
    n_molecules = 4
    molecules = [make_molecule(seed=10 + k) for k in range(n_molecules)]

    single_energies, single_forces = [], []
    for z, pos, edge_index, edge_attr, batch in molecules:
        energy, forces = model.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)
        single_energies.append(energy)
        single_forces.append(forces)

    # Collate by hand, the way PyG does: concatenate node tensors and shift
    # each molecule's edge_index by the number of atoms already placed.
    z_parts, pos_parts, edge_parts, attr_parts, batch_parts = [], [], [], [], []
    offset = 0
    for index, (z, pos, edge_index, edge_attr, _) in enumerate(molecules):
        z_parts.append(z)
        pos_parts.append(pos)
        edge_parts.append(edge_index + offset)
        attr_parts.append(edge_attr)
        batch_parts.append(torch.full((pos.shape[0],), index, dtype=torch.int64))
        offset += pos.shape[0]

    energy_batched, forces_batched = model.predict_energy_and_forces(
        torch.cat(z_parts),
        torch.cat(pos_parts),
        torch.cat(edge_parts, dim=1),
        torch.cat(attr_parts),
        torch.cat(batch_parts),
    )

    assert energy_batched.shape == (n_molecules,)
    assert torch.allclose(energy_batched, torch.cat(single_energies), atol=TOL, rtol=0)
    assert torch.allclose(forces_batched, torch.cat(single_forces), atol=TOL, rtol=0)

    # The molecules must not be identical, or the test would pass even if
    # every molecule in the batch received molecule 0's result.
    assert energy_batched.std() > 1e-6


def test_batching_with_unequal_molecule_sizes():
    """Different atom counts per molecule - the case a fixed stride would break."""
    model = make_model()
    sizes = [4, 9, 6]
    molecules = [make_molecule(seed=40 + k, n_atoms=n) for k, n in enumerate(sizes)]

    single = [
        model.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)
        for z, pos, edge_index, edge_attr, batch in molecules
    ]

    z_parts, pos_parts, edge_parts, attr_parts, batch_parts = [], [], [], [], []
    offset = 0
    for index, (z, pos, edge_index, edge_attr, _) in enumerate(molecules):
        z_parts.append(z)
        pos_parts.append(pos)
        edge_parts.append(edge_index + offset)
        attr_parts.append(edge_attr)
        batch_parts.append(torch.full((pos.shape[0],), index, dtype=torch.int64))
        offset += pos.shape[0]

    energy, forces = model.predict_energy_and_forces(
        torch.cat(z_parts), torch.cat(pos_parts), torch.cat(edge_parts, dim=1),
        torch.cat(attr_parts), torch.cat(batch_parts),
    )

    assert energy.shape == (len(sizes),)
    assert forces.shape == (sum(sizes), 3)
    assert torch.allclose(energy, torch.cat([e for e, _ in single]), atol=TOL, rtol=0)
    assert torch.allclose(forces, torch.cat([f for _, f in single]), atol=TOL, rtol=0)


# --------------------------------------------------------------------------
# 6. The gradient path to `pos` must be intact
# --------------------------------------------------------------------------

def test_gradient_path_from_energy_to_pos_is_intact():
    """`pos.requires_grad` is honoured and the energy really depends on it.

    If the graph from energy back to atomic positions were severed, this
    fails here at test time rather than silently producing zero or wrong
    forces in a results table.
    """
    model = make_model()
    z, pos, edge_index, edge_attr, batch = make_molecule()
    pos = pos.clone().requires_grad_(True)

    energy = model.predict_energy(z, pos, edge_index, edge_attr, batch)
    assert energy.requires_grad, "energy is detached from its inputs"
    assert energy.grad_fn is not None

    energy.sum().backward()
    assert pos.grad is not None, "no gradient reached the caller's own pos tensor"
    assert torch.isfinite(pos.grad).all()
    # Every atom must feel a force: an all-zero row would mean that atom's
    # coordinates never entered the energy.
    assert (pos.grad.abs().sum(dim=1) > 1e-9).all()


def test_forces_flow_back_to_the_callers_pos_tensor():
    """When the caller owns a grad-enabled `pos`, forces attach to *that* tensor."""
    model = make_model()
    z, pos, edge_index, edge_attr, batch = make_molecule()
    pos = pos.clone().requires_grad_(True)

    model.train()  # create_graph=True, as the force loss needs during training
    _, forces = model.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)

    # A force loss has to be differentiable w.r.t. the parameters, which
    # requires the double-backward graph to still be attached.
    assert forces.grad_fn is not None, "forces are detached; a force loss cannot train"
    forces.pow(2).sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_eval_mode_does_not_retain_the_double_backward_graph():
    """Evaluation must not hold the graph, or the eval loop leaks memory."""
    model = make_model()
    z, pos, edge_index, edge_attr, batch = make_molecule()

    model.eval()
    _, forces = model.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)
    assert forces.grad_fn is None


def test_forces_still_computed_under_no_grad():
    """`ml/training/evaluate.py` wraps its loop in `torch.no_grad()`."""
    model = make_model()
    z, pos, edge_index, edge_attr, batch = make_molecule()

    expected_energy, expected_forces = model.predict_energy_and_forces(
        z, pos, edge_index, edge_attr, batch
    )
    with torch.no_grad():
        energy, forces = model.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)

    assert torch.allclose(energy, expected_energy, atol=TOL, rtol=0)
    assert torch.allclose(forces, expected_forces, atol=TOL, rtol=0)
    assert forces.abs().max() > 1e-6


# --------------------------------------------------------------------------
# 7. `edge_attr` must not reach the energy (it is detached from `pos`)
# --------------------------------------------------------------------------

def test_edge_attr_does_not_influence_energy_or_forces():
    """Regression test for the reason PaiNN recomputes its own radial basis.

    `MD17Dataset.get()` builds `edge_attr` through `pos.numpy()`, so it
    arrives with `grad_fn=None` - severed from the autograd graph. A model
    that fed it into the energy would compute

        -dE/dpos   instead of   -(dE/dpos + dE/drbf * drbf/dpos)

    and report forces that are not the gradient of the energy it predicts.
    Note that `test_forces_equal_negative_gradient_of_energy` above would
    *still pass* in that case, because both sides of that comparison
    inherit the same detached tensor - which is exactly why this separate
    test exists.

    Feeding the model deliberately wrong edge features must therefore
    change nothing at all.
    """
    model = make_model()
    z, pos, edge_index, edge_attr, batch = make_molecule()

    energy, forces = model.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)

    for label, corrupted in (
        ("zeros", torch.zeros_like(edge_attr)),
        ("noise", torch.randn_like(edge_attr) * 100.0),
        ("none", None),
    ):
        energy_other, forces_other = model.predict_energy_and_forces(
            z, pos, edge_index, corrupted, batch
        )
        assert torch.equal(energy_other, energy), f"edge_attr={label} changed the energy"
        assert torch.equal(forces_other, forces), f"edge_attr={label} changed the forces"


def test_dataset_edge_attr_is_detached_from_pos():
    """Pins the upstream fact the design above depends on.

    If `ml/data/datasets.py` ever starts building `edge_attr` differentiably,
    this test fails and the PaiNN design comment should be revisited.
    """
    _, pos, _, edge_attr, _ = make_molecule()
    assert not edge_attr.requires_grad
    assert edge_attr.grad_fn is None


# --------------------------------------------------------------------------
# 8. Numerical health at initialisation
# --------------------------------------------------------------------------

def test_outputs_are_finite_at_initialisation():
    """`v` starts at exactly zero, where `d||x||/dx` is undefined.

    A plain `torch.linalg.norm` in the update block produces NaN gradients
    on the very first forward pass because `||V v_i||` is exactly zero
    there. `_safe_norm` is what prevents it; this test is what proves it,
    across several random initialisations.
    """
    for seed in range(4):
        model = make_model(seed=seed)
        z, pos, edge_index, edge_attr, batch = make_molecule(seed=seed)
        energy, forces = model.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)
        assert torch.isfinite(energy).all(), f"non-finite energy at seed {seed}"
        assert torch.isfinite(forces).all(), f"non-finite forces at seed {seed}"


def test_energy_is_size_extensive():
    """Two far-apart copies inside *one* graph have twice the energy of one.

    Sum-pooling plus the cutoff envelope guarantee it. A mean-pooled
    readout would return the single-molecule energy instead, and would be
    wrong for every molecule whose atom count differs from training.

    Both copies must sit in the **same** graph (`batch` all zeros) for this
    to bite. Putting them in two separate graphs of equal size divides the
    mean by the same atom count twice over, so a mean-pooled readout would
    pass - that version of this test was silently vacuous.
    """
    model = make_model()
    z, pos, edge_index, edge_attr, batch = make_molecule()
    n_atoms = pos.shape[0]

    energy_single = model.predict_energy(z, pos, edge_index, edge_attr, batch)

    # Far enough apart that no pair crosses the 5 A cutoff, so the two
    # copies genuinely do not interact and the total is exactly additive.
    far = torch.tensor([500.0, 0.0, 0.0], dtype=torch.float64)
    energy_pair = model.predict_energy(
        torch.cat([z, z]),
        torch.cat([pos, pos + far]),
        torch.cat([edge_index, edge_index + n_atoms], dim=1),
        torch.cat([edge_attr, edge_attr]),
        torch.zeros(2 * n_atoms, dtype=torch.int64),
    )

    assert energy_pair.shape == (1,)
    assert torch.allclose(energy_pair, 2.0 * energy_single, atol=TOL, rtol=0)


def test_non_interacting_copies_split_across_graphs_are_independent():
    """The same two copies as separate graphs each score the single-molecule energy."""
    model = make_model()
    z, pos, edge_index, edge_attr, batch = make_molecule()
    n_atoms = pos.shape[0]

    energy_single = model.predict_energy(z, pos, edge_index, edge_attr, batch)
    far = torch.tensor([500.0, 0.0, 0.0], dtype=torch.float64)
    energy_two_graphs = model.predict_energy(
        torch.cat([z, z]),
        torch.cat([pos, pos + far]),
        torch.cat([edge_index, edge_index + n_atoms], dim=1),
        torch.cat([edge_attr, edge_attr]),
        torch.cat([batch, batch + 1]),
    )

    assert torch.allclose(energy_two_graphs, energy_single.repeat(2), atol=TOL, rtol=0)


def test_message_direction_distinguishes_neighbour_from_centre():
    """`edge_index` row 0 is the neighbour, row 1 the centre receiving the message.

    Aggregating into the wrong column - or gathering features from the
    wrong one - is completely invisible to every symmetry test in this
    file: the result stays rotation-, translation- and permutation-
    equivariant, batches correctly, and remains a conservative field. It is
    simply the wrong physics, with each atom messaging itself.

    A directed single-edge graph pins it. `build_graph` always emits a
    symmetric edge set, which is exactly why the bug hides there.

    The two candidate edges must be **equidistant**, or the radial filter
    alone makes the energies differ and the assertion passes without
    testing anything - the first version of this test had exactly that
    hole. Atoms 1 and 2 are placed at +/- the same offset from atom 0 and
    given different elements, so distance is held fixed and only the
    identity of the atom at each end of the edge varies.
    """
    model = make_model()
    # d(0,1) == d(0,2) == 1.3 exactly; C, H, O are all distinct elements.
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [-1.3, 0.0, 0.0]], dtype=torch.float64
    )
    z = torch.tensor([6, 1, 8], dtype=torch.int64).view(-1, 1)
    batch = torch.zeros(3, dtype=torch.int64)

    def energy_of(neighbour: int, centre: int) -> torch.Tensor:
        directed = torch.tensor([[neighbour], [centre]], dtype=torch.int64)
        return model.predict_energy(z, pos, directed, None, batch)

    # Same neighbour and same distance, different centres: only the centre's
    # features are updated, and H differs from O, so these must differ.
    # Scattering into row 0 would send both messages to atom 0 instead and
    # make them identical.
    assert not torch.allclose(energy_of(0, 1), energy_of(0, 2), atol=1e-9, rtol=0)

    # Same centre and same distance, different neighbours: the message
    # carries the neighbour's features, so these must differ too. Gathering
    # from row 1 would read atom 0 both times and make them identical.
    assert not torch.allclose(energy_of(1, 0), energy_of(2, 0), atol=1e-9, rtol=0)


def test_centrosymmetric_atom_feels_exactly_zero_force():
    """An atom at the centre of a symmetric pair must feel no net force.

    Atom 0 sits at the origin between two *identical* atoms at +/- x. The
    configuration is invariant under reflection through the yz-plane
    combined with swapping atoms 1 and 2, so the x-component of the force
    on atom 0 must be exactly zero by symmetry - not merely small.

    This is also where the vector features cancel exactly (``v_0 == 0``),
    which is the case `_safe_norm` exists for. `torch.linalg.norm` returns
    a zero gradient at the origin but jumps to 1.0 one ulp away, and that
    discontinuity shows up here as a spurious force on a perfectly
    balanced atom.
    """
    model = make_model()
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [-1.2, 0.0, 0.0]], dtype=torch.float64
    )
    z = torch.tensor([6, 1, 1], dtype=torch.int64).view(-1, 1)  # the outer pair is identical
    batch = torch.zeros(3, dtype=torch.int64)
    distances = torch.cdist(pos, pos)
    mask = (distances <= CUTOFF) & ~torch.eye(3, dtype=torch.bool)
    edge_index = torch.stack(mask.nonzero(as_tuple=True))

    _, forces = model.predict_energy_and_forces(z, pos, edge_index, None, batch)

    assert torch.allclose(forces[0], torch.zeros(3, dtype=torch.float64), atol=TOL, rtol=0)
    # The outer two must feel equal and opposite forces, and non-zero ones -
    # otherwise the assertion above is satisfied by a model that predicts
    # no forces at all.
    assert torch.allclose(forces[1], -forces[2], atol=TOL, rtol=0)
    assert forces[1].abs().max() > 1e-6


def test_energy_depends_on_bond_angle_at_fixed_bond_lengths():
    """PaiNN must see angles, not just distances - its whole reason for being here.

    No symmetry test can catch the loss of this. If the vector message
    dropped its direction factor (``m_vs * r_hat`` becoming a bare
    ``m_vs``), the model would collapse into a distance-only potential:
    still rotation-invariant, still translation-invariant, still
    conservative, still passing every other test in this file - and no
    longer PaiNN. The bake-off would then be comparing two invariant
    models and reporting it as equivariant-vs-invariant.

    Two configurations are built with *identical* edge lengths and
    identical elements, differing only in the angle between the two bonds.
    The edge set is restricted to the two bonds so the 1-2 distance, which
    does differ, never enters the graph. A distance-only model cannot tell
    these apart; PaiNN can, because atom 0's vector feature accumulates
    ``r_hat_01 + r_hat_02`` whose norm depends on the angle.
    """
    model = make_model()
    bond = 1.3
    z = torch.tensor([8, 1, 1], dtype=torch.int64).view(-1, 1)  # both neighbours are H
    batch = torch.zeros(3, dtype=torch.int64)
    # Only the two bonds, in both directions - never the 1-2 pair.
    edge_index = torch.tensor([[1, 0, 2, 0], [0, 1, 0, 2]], dtype=torch.int64)

    def positions(angle_degrees: float) -> torch.Tensor:
        angle = math.radians(angle_degrees)
        return torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [bond, 0.0, 0.0],
                [bond * math.cos(angle), bond * math.sin(angle), 0.0],
            ],
            dtype=torch.float64,
        )

    right_angle, wide_angle = positions(90.0), positions(120.0)

    # Precondition: every distance the graph actually uses is identical.
    for edge in range(edge_index.shape[1]):
        a, b = edge_index[0, edge], edge_index[1, edge]
        assert torch.allclose(
            (right_angle[a] - right_angle[b]).norm(),
            (wide_angle[a] - wide_angle[b]).norm(),
            atol=1e-12, rtol=0,
        )

    energy_90 = model.predict_energy(z, right_angle, edge_index, None, batch)
    energy_120 = model.predict_energy(z, wide_angle, edge_index, None, batch)

    assert not torch.allclose(energy_90, energy_120, atol=1e-9, rtol=0), (
        "energy is identical at 90 and 120 degrees with the same bond lengths - "
        "the model has collapsed to a distance-only potential"
    )


def test_energy_is_continuous_across_the_cutoff():
    """An atom crossing the cutoff must not step-change the energy.

    The neighbour list is rebuilt every frame, so an edge appears and
    disappears discretely. The cosine envelope is what makes the message
    already vanish by the time that happens; without it the potential has a
    jump discontinuity at 5 A, the forces have a delta there, and an MD
    trajectory driven by it gains energy every time a pair crosses.
    """
    model = make_model()
    z = torch.tensor([6, 1], dtype=torch.int64).view(-1, 1)
    batch = torch.zeros(2, dtype=torch.int64)
    both_ways = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    no_edges = torch.zeros(2, 0, dtype=torch.int64)

    def pair_at(distance: float) -> torch.Tensor:
        return torch.tensor([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=torch.float64)

    # Just inside the cutoff, the edge still exists; just outside,
    # `build_graph` has dropped it. The two must agree.
    inside = model.predict_energy(z, pair_at(CUTOFF - 1e-3), both_ways, None, batch)
    outside = model.predict_energy(z, pair_at(CUTOFF + 1e-3), no_edges, None, batch)
    assert torch.allclose(inside, outside, atol=1e-6, rtol=0)

    # Not vacuous: a genuinely bonded pair must differ from no bond at all,
    # or the envelope could be zeroing every message at every distance.
    bonded = model.predict_energy(z, pair_at(1.1), both_ways, None, batch)
    assert (bonded - outside).abs().max() > 1e-3


def test_forces_sum_to_zero():
    """Newton's third law: no net force on an isolated molecule.

    Follows from translation invariance - if the energy cannot depend on
    the molecule's absolute position, the total force must vanish - so
    this is a second, independent read on the same property.
    """
    model = make_model()
    z, pos, edge_index, edge_attr, batch = make_molecule()
    _, forces = model.predict_energy_and_forces(z, pos, edge_index, edge_attr, batch)

    assert torch.allclose(forces.sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=TOL, rtol=0)


# --------------------------------------------------------------------------
# 9. Interface contract against what MD17Dataset actually emits
# --------------------------------------------------------------------------

@pytest.fixture
def tiny_gold(tmp_path):
    """A gold-format .npz with the same key layout `preprocessing.py` writes.

    Real gold data is 229 MB and is not committed, so the shape contract is
    pinned against a miniature file rather than skipped. The split indices
    are contiguous blocks, as `trajectory_block_split` produces.
    """
    rng = np.random.default_rng(0)
    n_configs, n_atoms = 12, len(_Z)
    path = tmp_path / "toy_dft.npz"
    np.savez_compressed(
        path,
        z=np.array(_Z, dtype=np.int64),
        R=(rng.normal(size=(n_configs, n_atoms, 3)) * 1.4).astype(np.float32),
        E=rng.normal(size=n_configs).astype(np.float32),
        F=rng.normal(size=(n_configs, n_atoms, 3)).astype(np.float32),
        train_idx=np.arange(0, 8),
        val_idx=np.arange(8, 10),
        test_idx=np.arange(10, 12),
        molecule="toy",
        theory="dft",
        used_literature_split=False,
    )
    return path


def test_accepts_a_real_pyg_batch_from_md17dataset(tiny_gold):
    """The argument names and shapes must match what the loader really emits.

    `Data.x` is (N, 1) int64 atomic numbers, not a float feature matrix -
    the model indexes an embedding table with it. This test is what stops
    that assumption from silently rotting.
    """
    from torch_geometric.loader import DataLoader

    from ml.data.datasets import MD17Dataset

    dataset = MD17Dataset(tiny_gold, split="train", cutoff_radius=CUTOFF, num_rbf=16)
    batch = next(iter(DataLoader(dataset, batch_size=3, shuffle=False)))

    assert batch.x.dtype == torch.int64 and batch.x.shape[1] == 1
    assert batch.pos.shape[1] == 3
    assert batch.edge_index.shape[0] == 2

    model = PaiNN(hidden_channels=16, num_layers=1, num_rbf=8, cutoff_radius=CUTOFF)
    energy, forces = model.predict_energy_and_forces(
        batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch
    )

    # Exactly the shapes ml/utils/contract.py requires of a ModelOutput:
    # energy (B,) - a (B, 1) would broadcast to (B, B) against the
    # reference - and forces (N, 3), concatenated rather than padded.
    assert energy.shape == (batch.num_graphs,)
    assert forces.shape == batch.force.shape == (batch.num_nodes, 3)
    assert energy.dtype == batch.y.dtype


def test_model_output_contract_accepts_the_predictions(tiny_gold):
    """Predictions must pass `ModelOutput`'s validation unmodified."""
    from torch_geometric.loader import DataLoader

    from ml.data.datasets import MD17Dataset
    from ml.utils.contract import ModelOutput, ReferenceData, require_compatible

    dataset = MD17Dataset(tiny_gold, split="test", cutoff_radius=CUTOFF, num_rbf=16)
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    model = PaiNN(hidden_channels=16, num_layers=1, num_rbf=8, cutoff_radius=CUTOFF)
    energy, forces = model.predict_energy_and_forces(
        batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch
    )

    prediction = ModelOutput(E=energy.detach(), F=forces.detach())
    reference = ReferenceData(
        E=batch.y, F=batch.force, batch=batch.batch,
        molecule=dataset.molecule, theory=dataset.theory,
    )
    require_compatible(prediction, reference)  # raises on any mismatch


# --------------------------------------------------------------------------
# 10. Input validation
# --------------------------------------------------------------------------

def test_flat_z_and_column_z_agree():
    """(N,) and (N, 1) atomic numbers must give identical results."""
    model = make_model()
    z, pos, edge_index, edge_attr, batch = make_molecule()

    from_column = model.predict_energy(z, pos, edge_index, edge_attr, batch)
    from_flat = model.predict_energy(z.view(-1), pos, edge_index, edge_attr, batch)
    assert torch.equal(from_column, from_flat)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda kw: kw.update(pos=kw["pos"][:, :2]), "pos must have shape"),
        (lambda kw: kw.update(edge_index=kw["edge_index"][0]), "edge_index must have shape"),
        (lambda kw: kw.update(batch=kw["batch"][:-1]), "batch must have shape"),
        (lambda kw: kw.update(edge_attr=kw["edge_attr"][:-1]), "edge_attr has"),
        (lambda kw: kw.update(z=kw["z"].double()), "must be an integer tensor"),
    ],
)
def test_malformed_inputs_raise_rather_than_guess(mutate, message):
    """Shape bugs must fail loudly; a silent broadcast prints a plausible number."""
    model = make_model()
    z, pos, edge_index, edge_attr, batch = make_molecule()
    kwargs = dict(z=z, pos=pos, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    mutate(kwargs)

    with pytest.raises(ValueError, match=message):
        model.predict_energy(**kwargs)


def test_atomic_number_beyond_the_embedding_table_raises():
    model = make_model(max_atomic_number=8)
    z, pos, edge_index, edge_attr, batch = make_molecule()
    z = z.clone()
    z[0] = 17  # Cl, in _PERIODIC_TABLE but past this model's table

    with pytest.raises(ValueError, match="exceeds max_atomic_number"):
        model.predict_energy(z, pos, edge_index, edge_attr, batch)
