"""Split-leakage measurements for MD17 gold files (SCRUM-47).

Adjacent MD frames are near-duplicates, so a split leaks if a held-out
frame has a training frame that is, structurally, the same configuration.
This module measures that: for each query frame (test or validation), the
aligned RMSD to its closest frame in a reference set (train), and, as a
scale reference, the RMSD between frames a given number of steps apart in
the trajectory.

**Distance.** RMSD after optimal superposition (Kabsch: best rotation,
centroids removed), minimised over relabellings of chemically equivalent
atoms. Proper rotations only by default; `allow_reflection=True` also
admits mirror images, which every distance-based model in this repo
cannot tell apart, as a sensitivity check.

**Equivalent atoms.** Hydrogens bonded to the same heavy atom can swap
labels without changing the molecule (ethanol: the three methyl H, the two
methylene H), and a methyl rotation does exactly that over a trajectory.
Bonding is read from one reference frame (each H belongs to its nearest
heavy atom) and checked on others by `check_groups_stable`. Heavy-atom
symmetry (e.g. benzene's ring) is not handled; ethanol has none.

**Search.** Exact all-pairs RMSD is too expensive (ethanol DFT: 55,509
test frames x 444,074 train frames). Candidates are shortlisted by the
distance between interatomic-distance vectors (`distance_vector`: every
pair, in fixed atom order - invariant to rotation, translation and
reflection), taking for each reference frame the best of the query's
equivalent-atom relabellings; exact RMSD is then computed for the `k`
closest. `exact_nearest` is the brute-force version (every reference
frame, every relabelling) that the shortlist is validated against, and
`shortlist_recall` checks that a larger `k` would not have found a closer
frame. (A fully permutation-invariant descriptor - distances sorted within
each pair of groups, `invariant_descriptor` - was tried first and ranked
candidates too poorly at this scale: a larger shortlist found a closer
frame for 72% of queries.)

Numbers are float64 for the RMSD itself: near-duplicate frames differ by
thousandths of an angstrom, below float32's resolution for this formula.
"""
from __future__ import annotations

import itertools
from typing import Optional, Sequence

import numpy as np
import torch

__all__ = [
    "equivalent_atom_groups", "check_groups_stable", "group_permutations",
    "invariant_descriptor", "distance_vector", "kabsch_rmsd", "min_rmsd_over_permutations",
    "nearest_neighbours", "exact_nearest", "lag_rmsd", "shortlist_recall", "split_gaps",
]


def equivalent_atom_groups(z: np.ndarray, R_ref: np.ndarray) -> list[np.ndarray]:
    """Atom index groups whose members can swap labels: each heavy atom on
    its own, then the hydrogens bonded to each heavy atom (nearest heavy
    atom in `R_ref`, shape [n_atoms, 3])."""
    z = np.asarray(z)
    heavy = np.flatnonzero(z > 1)
    light = np.flatnonzero(z == 1)
    groups = [np.array([i]) for i in heavy]
    if len(light):
        d = np.linalg.norm(R_ref[light][:, None, :] - R_ref[heavy][None, :, :], axis=-1)
        parent = heavy[d.argmin(axis=1)]
        groups += [light[parent == h] for h in heavy if np.any(parent == h)]
    return groups


def check_groups_stable(z: np.ndarray, R: np.ndarray, groups: Sequence[np.ndarray]) -> bool:
    """True if every frame in `R` ([n_frames, n_atoms, 3]) gives the same
    hydrogen-to-heavy-atom assignment as `groups` (no bond broke)."""
    reference = [tuple(g) for g in groups]
    return all([tuple(g) for g in equivalent_atom_groups(z, frame)] == reference for frame in R)


def group_permutations(groups: Sequence[np.ndarray], n_atoms: int) -> np.ndarray:
    """Every relabelling that only swaps atoms within a group, as index
    arrays [n_perms, n_atoms]; the identity is row 0."""
    per_group = [list(itertools.permutations(g)) for g in groups]
    perms = []
    for combo in itertools.product(*per_group):
        p = np.arange(n_atoms)
        for g, gp in zip(groups, combo):
            p[np.asarray(g)] = gp
        perms.append(p)
    return np.stack(perms)


def invariant_descriptor(R: np.ndarray, groups: Sequence[np.ndarray]) -> np.ndarray:
    """[n_frames, n_pairs] interatomic distances, sorted within each pair of
    groups, so the descriptor is unchanged by rotation, translation,
    reflection and any within-group relabelling."""
    R = np.asarray(R, dtype=np.float64)
    blocks = []
    for a, b in itertools.combinations_with_replacement(range(len(groups)), 2):
        ga, gb = np.asarray(groups[a]), np.asarray(groups[b])
        if a == b:
            if len(ga) < 2:
                continue
            i, j = np.triu_indices(len(ga), 1)
            d = np.linalg.norm(R[:, ga[i]] - R[:, ga[j]], axis=-1)
        else:
            d = np.linalg.norm(R[:, ga][:, :, None, :] - R[:, gb][:, None, :, :], axis=-1)
            d = d.reshape(len(R), -1)
        blocks.append(np.sort(d, axis=1))
    return np.concatenate(blocks, axis=1)


def distance_vector(R: np.ndarray | torch.Tensor) -> torch.Tensor:
    """[n_frames, n_atoms*(n_atoms-1)/2] interatomic distances in fixed atom
    order (upper triangle). Invariant to rotation, translation and
    reflection, but not to relabelling - see `nearest_neighbours`."""
    R = torch.as_tensor(R)
    i, j = torch.triu_indices(R.shape[-2], R.shape[-2], 1, device=R.device)
    return (R[..., i, :] - R[..., j, :]).norm(dim=-1)


def _det3(M: torch.Tensor) -> torch.Tensor:
    return (M[..., 0, 0] * (M[..., 1, 1] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 1])
            - M[..., 0, 1] * (M[..., 1, 0] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 0])
            + M[..., 0, 2] * (M[..., 1, 0] * M[..., 2, 1] - M[..., 1, 1] * M[..., 2, 0]))


def _sym3_eigvals(A: torch.Tensor) -> torch.Tensor:
    """Eigenvalues of symmetric 3x3 matrices [..., 3, 3], ascending, in
    closed form (trigonometric method). Element-wise arithmetic only, so
    it runs at full speed in large batches, where torch.linalg.eigvalsh's
    per-matrix solver managed about 1.2M matrices/s on CPU and GPU alike."""
    p1 = A[..., 0, 1] ** 2 + A[..., 0, 2] ** 2 + A[..., 1, 2] ** 2
    q = (A[..., 0, 0] + A[..., 1, 1] + A[..., 2, 2]) / 3
    p2 = (A[..., 0, 0] - q) ** 2 + (A[..., 1, 1] - q) ** 2 + (A[..., 2, 2] - q) ** 2 + 2 * p1
    p = (p2 / 6).sqrt()
    eye = torch.eye(3, dtype=A.dtype, device=A.device)
    safe_p = torch.where(p > 0, p, torch.ones_like(p))
    B = (A - q[..., None, None] * eye) / safe_p[..., None, None]
    r = (_det3(B) / 2).clamp(-1, 1)
    phi = torch.acos(r) / 3
    big = q + 2 * p * torch.cos(phi)
    small = q + 2 * p * torch.cos(phi + 2 * torch.pi / 3)
    mid = 3 * q - big - small
    return torch.stack([small, mid, big], dim=-1)


def kabsch_rmsd(P: torch.Tensor, Q: torch.Tensor, allow_reflection: bool = False) -> torch.Tensor:
    """RMSD between P and Q ([..., n_atoms, 3], broadcastable) after
    removing centroids and applying the best rotation.

    From the singular values of the cross-covariance H = P^T Q (via the
    eigenvalues of H^T H): MSD = (|P|^2 + |Q|^2 - 2(s1 + s2 + d*s3)) / n,
    with d = sign(det H) for a proper rotation, +1 if reflections count.
    """
    P = P.to(torch.float64)
    Q = Q.to(torch.float64)
    P = P - P.mean(dim=-2, keepdim=True)
    Q = Q - Q.mean(dim=-2, keepdim=True)
    H = P.transpose(-1, -2) @ Q
    s = _sym3_eigvals(H.transpose(-1, -2) @ H).clamp_min(0).sqrt()  # ascending
    if allow_reflection:
        trace = s.sum(dim=-1)
    else:
        sign = torch.sign(_det3(H))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        trace = s[..., 2] + s[..., 1] + sign * s[..., 0]
    e0 = (P * P).sum(dim=(-1, -2)) + (Q * Q).sum(dim=(-1, -2))
    return ((e0 - 2 * trace).clamp_min(0) / P.shape[-2]).sqrt()


def min_rmsd_over_permutations(query: torch.Tensor, candidates: torch.Tensor, perms: torch.Tensor,
                               allow_reflection: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """For query [n_q, n_atoms, 3] and candidates [n_q, k, n_atoms, 3]: the
    smallest RMSD over candidates and relabellings (perms [n_perms, n_atoms]),
    and which candidate gave it. Returns ([n_q], [n_q])."""
    q = query[:, perms]                                   # [n_q, n_perms, n_atoms, 3]
    r = kabsch_rmsd(q[:, None], candidates[:, :, None], allow_reflection)  # [n_q, k, n_perms]
    best_per_candidate = r.min(dim=-1).values
    best, which = best_per_candidate.min(dim=-1)
    return best, which


def nearest_neighbours(R_query: np.ndarray, R_ref: np.ndarray, groups: Sequence[np.ndarray],
                       k: int = 512, chunk: int = 64, device: Optional[str] = None,
                       allow_reflection: bool = False, refine_top: int = 0,
                       refine_window: int = 0, rmsd_device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """For every query frame, the RMSD to its closest reference frame and
    that frame's position in `R_ref`.

    Shortlist: for each reference frame, the smallest `distance_vector`
    distance over the query's relabellings; the `k` smallest go on to the
    exact RMSD (minimised over relabellings again).

    Refinement (only meaningful when `R_ref` is in trajectory order): the
    `refine_top` best candidates' neighbours within +-`refine_window`
    frames are scored too, since a configuration's closest match tends to
    sit next to another close one in time.

    The shortlist runs on `device` (float32 distance vectors, GPU if
    available); the exact RMSD runs on `rmsd_device` in float64, CPU by
    default because consumer GPUs are slow at float64."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    n_atoms = R_ref.shape[1]
    perms_np = group_permutations(groups, n_atoms)
    perms = torch.as_tensor(perms_np, device=device)
    perms_exact = torch.as_tensor(perms_np, device=rmsd_device)
    ref_dv = distance_vector(torch.as_tensor(np.asarray(R_ref), dtype=torch.float32, device=device))
    ref_xyz = torch.as_tensor(np.asarray(R_ref), dtype=torch.float64, device=rmsd_device)
    k = min(k, len(R_ref))
    best_rmsd = np.empty(len(R_query))
    best_index = np.empty(len(R_query), dtype=np.int64)
    for start in range(0, len(R_query), chunk):
        stop = min(start + chunk, len(R_query))
        q_np = np.asarray(R_query[start:stop])
        q_dv = distance_vector(torch.as_tensor(q_np, dtype=torch.float32, device=device)[:, perms])
        d = torch.cdist(q_dv.reshape(-1, q_dv.shape[-1]), ref_dv)         # [n_q*n_perms, n_ref]
        d = d.reshape(len(q_np), len(perms), -1).min(dim=1).values       # [n_q, n_ref]
        shortlist = d.topk(k, dim=1, largest=False).indices.to(rmsd_device)  # [n_q, k]
        q_xyz = torch.as_tensor(q_np, dtype=torch.float64, device=rmsd_device)
        if refine_top and refine_window:
            r0 = kabsch_rmsd(q_xyz[:, perms_exact][:, None], ref_xyz[shortlist][:, :, None]).min(dim=-1).values
            top = shortlist.gather(1, r0.topk(min(refine_top, k), dim=1, largest=False).indices)
            offsets = torch.arange(-refine_window, refine_window + 1, device=rmsd_device)
            near = (top[:, :, None] + offsets).reshape(len(q_np), -1).clamp(0, len(R_ref) - 1)
            shortlist = torch.cat([shortlist, near], dim=1)
        rmsd, which = min_rmsd_over_permutations(q_xyz, ref_xyz[shortlist], perms_exact, allow_reflection)
        best_rmsd[start:stop] = rmsd.numpy()
        best_index[start:stop] = shortlist.gather(1, which[:, None]).squeeze(1).numpy()
    return best_rmsd, best_index


def exact_nearest(query: np.ndarray, R_ref: np.ndarray, groups: Sequence[np.ndarray],
                  chunk: int = 50_000, device: Optional[str] = None) -> tuple[float, int]:
    """Brute force for one query frame [n_atoms, 3]: exact RMSD to every
    reference frame under every relabelling. Slow; for validating
    `nearest_neighbours` on a sample."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    perms = torch.as_tensor(group_permutations(groups, R_ref.shape[1]), device=device)
    q = torch.as_tensor(np.asarray(query), dtype=torch.float64, device=device)[perms]  # [n_perms, n_atoms, 3]
    best, best_i = float("inf"), -1
    for start in range(0, len(R_ref), chunk):
        ref = torch.as_tensor(np.asarray(R_ref[start:start + chunk]), dtype=torch.float64, device=device)
        r = kabsch_rmsd(q[None], ref[:, None]).min(dim=1).values           # [chunk]
        value, i = r.min(dim=0)
        if value.item() < best:
            best, best_i = value.item(), start + int(i)
    return best, best_i


def lag_rmsd(R: np.ndarray, lags: Sequence[int], groups: Sequence[np.ndarray], n_samples: int = 2000,
             seed: int = 0, device: Optional[str] = None) -> dict[int, np.ndarray]:
    """RMSD (same metric) between frame t and frame t + lag for `n_samples`
    random t, per lag - what "near-duplicate" means in this trajectory."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    perms = torch.as_tensor(group_permutations(groups, R.shape[1]), device=device)
    rng = np.random.default_rng(seed)
    out = {}
    for lag in lags:
        if lag >= len(R):
            continue
        t = rng.integers(0, len(R) - lag, size=min(n_samples, len(R) - lag))
        a = torch.as_tensor(np.asarray(R[t]), dtype=torch.float64, device=device)
        b = torch.as_tensor(np.asarray(R[t + lag]), dtype=torch.float64, device=device)
        rmsd, _ = min_rmsd_over_permutations(a, b[:, None], perms)
        out[int(lag)] = rmsd.cpu().numpy()
    return out


def shortlist_recall(R_query: np.ndarray, R_ref: np.ndarray, groups: Sequence[np.ndarray],
                     k: int, k_check: int, device: Optional[str] = None) -> float:
    """Fraction of queries whose nearest-neighbour RMSD with `k` candidates
    equals (to 1e-9 A) the one with `k_check` > k candidates. 1.0 means the
    shortlist missed nothing a larger one would have found."""
    small, _ = nearest_neighbours(R_query, R_ref, groups, k=k, device=device)
    large, _ = nearest_neighbours(R_query, R_ref, groups, k=k_check, device=device)
    return float(np.mean(np.abs(small - large) <= 1e-9))


def split_gaps(train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, Optional[int]]:
    """Frames strictly between consecutive blocks in index order (0 = the
    blocks touch). None where a pair of blocks interleaves rather than
    following one another."""
    def gap(first: np.ndarray, second: np.ndarray) -> Optional[int]:
        if first.max() < second.min():
            return int(second.min() - first.max() - 1)
        return None
    return {"train_to_val": gap(train_idx, val_idx), "val_to_test": gap(val_idx, test_idx),
            "train_to_test": gap(train_idx, test_idx)}
