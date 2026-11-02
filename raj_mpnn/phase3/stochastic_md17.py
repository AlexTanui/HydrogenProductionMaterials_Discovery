"""Phase 1 / 2 / 3 on one PaiNN backbone, one data path and one training loop.

  --model painn   Phase 1: deterministic PaiNN.
  --model blip    Phase 2: BLIP weight-space stochasticity (unchanged from
                  Blip_model_building.ipynb).
  --model graph   Phase 3: graph-space stochasticity. Weights are ordinary
                  deterministic layers; the noise sits on representations:
                    --stoch node   node features after every update block
                    --stoch edge   the edge filter in every message block
                    --stoch both   both of the above (RQ3 ablation)

Phase 3 reuses BLIP's noise family, posterior network and KL so that the
only thing that changes between Phase 2 and Phase 3 is WHERE the noise lives:

  BLIP (weights):    w  ~ N(theta, alpha*theta^2)          per row
  Phase 3 (reps):    h' = h * (1 + sqrt(alpha) * eps),  eps ~ N(0, 1)
                     i.e. h' ~ N(h, alpha*h^2)            per node / edge

alpha comes from the same invariant posterior E_psi (embedding of Z for
nodes, |r_ij|^2 for edges) and is regularised by the same
KL[N(x, a x^2) || N(0, c x^2)], c = p/(1-p), times beta.

Symmetry: every perturbed quantity is either an invariant scalar (s, the
edge filter) or a PaiNN vector feature v multiplied by one scalar per
channel shared across x/y/z. So E stays invariant and F = -dE/dR stays
equivariant and conservative for every noise draw. tests_stochastic_md17.py
checks this numerically.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import tarfile
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KCAL_TO_MEV = 43.3641  # 1 kcal/mol = 43.3641 meV

# Data
RMD17_URL = "https://figshare.com/ndownloader/files/23950376"  # rmd17.tar.bz2
MD17_URL = "http://www.quantum-machine.org/gdml/data/npz/md17_{}.npz"
RMD17_NAMES = {"aspirin", "azobenzene", "benzene", "ethanol", "malonaldehyde",
               "naphthalene", "paracetamol", "salicylic", "toluene", "uracil"}


def _download(url, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)


def _check_not_lfs_pointer(path):
    with open(path, "rb") as f:
        head = f.read(64)
    if head.startswith(b"version https://git-lfs"):
        raise SystemExit(
            f"{path} is a Git LFS pointer, not the real data.\n"
            "Run `git lfs install && git lfs pull` inside the repository (or "
            "`scripts/download_bronze_data.sh`), then `python -m ml.data.preprocessing`.")


def stride_subset(n, size, seed):
    """Team rule from ml/training/train.py: fixed stride with a seeded offset."""
    if size is None or size <= 0 or size >= n:
        return np.arange(n)
    stride = n // size
    offset = int(np.random.default_rng(seed).integers(stride))
    return offset + stride * np.arange(size)


def load_gold(args):
    """Team gold file (HydrogenProductionMaterials_Discovery, data/gold/md17/
    {molecule}_{theory}.npz): z, R, E, F + train_idx/val_idx/test_idx, already
    in A and kcal/mol. The shipped contiguous-block splits are used as is,
    never re-split; optional subsets use the team's stride rule."""
    path = args.gold
    if path is None:
        path = os.path.join(args.repo_dir, "data", "gold", "md17", f"{args.molecule}_{args.theory}.npz")
    if not os.path.exists(path):
        raise SystemExit(
            f"gold file not found: {path}\n"
            "Inside the repository run:  git lfs pull  &&  python -m ml.data.preprocessing")
    _check_not_lfs_pointer(path)
    d = np.load(path, allow_pickle=True)
    for k in ("molecule", "theory"):
        if k in d.files:
            print(f"gold {k}: {d[k]}")
    tr = d["train_idx"][stride_subset(len(d["train_idx"]), args.train_subset, args.subset_seed)]
    va = d["val_idx"][stride_subset(len(d["val_idx"]), args.val_subset, args.subset_seed)]
    te = d["test_idx"][stride_subset(len(d["test_idx"]), args.test_subset, args.subset_seed)]
    return (d["z"].astype(np.int64), d["R"].astype(np.float32), d["E"].reshape(-1).astype(np.float64),
            d["F"].astype(np.float32), (tr, va, te))


def load_md17(args):
    """Returns z (N,), R (M,N,3) [A], E (M,) [kcal/mol], F (M,N,3) [kcal/mol/A],
    and (pool_idx, test_idx); pool_idx is the 1000-structure training budget."""
    rng = np.random.default_rng(args.seed)
    if args.synthetic:
        return make_synthetic(rng)

    split = None
    if args.npz:
        _check_not_lfs_pointer(args.npz)
        d = np.load(args.npz)
    elif args.dataset == "rmd17":
        mol = "salicylic" if args.molecule.startswith("salicylic") else args.molecule
        if mol not in RMD17_NAMES:
            raise ValueError(f"unknown rMD17 molecule '{mol}', choose from {sorted(RMD17_NAMES)}")
        root = os.path.join(args.data_dir, "rmd17")
        npz = os.path.join(root, "npz_data", f"rmd17_{mol}.npz")
        if not os.path.exists(npz):
            arc = os.path.join(args.data_dir, "rmd17.tar.bz2")
            if not os.path.exists(arc):
                _download(RMD17_URL, arc)
            with tarfile.open(arc) as t:
                t.extractall(args.data_dir)
        d = np.load(npz)
        sp = os.path.join(root, "splits")
        split = (np.loadtxt(os.path.join(sp, f"index_train_0{args.split}.csv"), dtype=int),
                 np.loadtxt(os.path.join(sp, f"index_test_0{args.split}.csv"), dtype=int))
    else:
        npz = os.path.join(args.data_dir, f"md17_{args.molecule}.npz")
        if not os.path.exists(npz):
            _download(MD17_URL.format(args.molecule), npz)
        d = np.load(npz)

    if "nuclear_charges" in d.files:   # rMD17 format
        z, R, E, Fo = d["nuclear_charges"], d["coords"], d["energies"], d["forces"]
    else:                              # MD17 / sGDML format
        z, R, E, Fo = d["z"], d["R"], d["E"], d["F"]
    E = np.asarray(E).reshape(-1)
    if split is None:
        perm = rng.permutation(len(E))
        split = (perm[:1000], perm[1000:1000 + args.n_test])
    return (z.astype(np.int64), R.astype(np.float32), E.astype(np.float64),
            Fo.astype(np.float32), split)


def make_synthetic(rng, n=1300, natoms=9):
    """Toy molecule with a Morse pair potential, for offline testing."""
    z = rng.choice([1, 6, 8], natoms)
    base = rng.normal(0, 1.2, (natoms, 3))
    R = base[None] + rng.normal(0, 0.08, (n, natoms, 3))
    Rt = torch.tensor(R, requires_grad=True)
    eye = torch.eye(natoms, dtype=Rt.dtype)
    d = torch.cdist(Rt, Rt) + eye
    d0 = torch.cdist(torch.tensor(base[None]), torch.tensor(base[None])) + eye
    E = (20.0 * (1 - torch.exp(-1.5 * (d - d0))) ** 2 * (1 - eye)).sum((1, 2)) / 2
    Fo = -torch.autograd.grad(E.sum(), Rt)[0]
    perm = rng.permutation(n)
    return (z.astype(np.int64), R.astype(np.float32), E.detach().numpy().astype(np.float64),
            Fo.numpy().astype(np.float32), (perm[:1000], perm[1000:]))


class Batch:
    """MD17 frames of one molecule share their atoms, so batching is a reshape;
    the radius graph is rebuilt for every batch."""

    def __init__(self, z, R, E, Fo, cutoff, device):
        B, N, _ = R.shape
        self.n_graphs, self.n_atoms = B, N
        self.z = z.repeat(B).to(device)
        self.pos = R.reshape(B * N, 3).to(device).requires_grad_(True)
        self.E = E.to(device)
        self.F = Fo.reshape(B * N, 3).to(device)
        self.batch = torch.arange(B, device=device).repeat_interleave(N)
        with torch.no_grad():
            m = (torch.cdist(R, R) < cutoff) & ~torch.eye(N, dtype=torch.bool)
            b, i, j = m.nonzero(as_tuple=True)
        self.dst = (b * N + i).to(device)  # receiving atom i
        self.src = (b * N + j).to(device)  # neighbour j (both directions present)


def make_batches(idx, data, bs, shuffle, rng, cutoff, device):
    z, R, E, Fo = data
    idx = rng.permutation(idx) if shuffle else idx
    for k in range(0, len(idx), bs):
        ii = idx[k:k + bs]
        yield Batch(z, R[ii], E[ii], Fo[ii], cutoff, device)


# BLIP components
class BayesLinear(nn.Linear):
    """w ~ N(theta, alpha * theta^2) via the local reparameterisation trick.
    alpha: (rows, 1) tensor, or None for the deterministic (MAP) pass."""

    def forward(self, x, alpha=None):
        mean = F.linear(x, self.weight, self.bias)
        if alpha is None:
            return mean
        var = alpha * F.linear(x.pow(2), self.weight.pow(2))
        return mean + torch.sqrt(var + 1e-18) * torch.randn_like(mean)


class KL:
    """KL[N(theta, a theta^2) || N(0, c theta^2)], c = p/(1-p), averaged over
    all rows and layers, times beta (official blip/bayes/kl.py)."""

    def __init__(self, p=0.5, beta=10.0):
        self.c, self.logc, self.beta = p / (1 - p), math.log(p / (1 - p)), beta

    def __call__(self, alphas):
        return self.beta * (0.5 * ((alphas + 1) / self.c + self.logc - torch.log(alphas) - 1)).mean()


class Posterior(nn.Module):
    """Invariant inference network E_psi (official PosteriorModelPaiNN):
    node head on an embedding of Z, edge head on ||r_i - r_j||^2."""

    def __init__(self, n_node, n_edge, hidden=64, n_layers=2, clamp_max_p=0.8, init_p=0.05):
        super().__init__()
        self.emb = nn.Embedding(100, hidden // 2, padding_idx=0)

        def mlp(i, o):
            layers, d = [], i
            for _ in range(n_layers - 1):
                layers += [nn.Linear(d, hidden), nn.SiLU()]
                d = hidden
            return nn.Sequential(*layers, nn.Linear(d, o))

        # A head with zero outputs is skipped (Phase 3 node-only / edge-only);
        # BLIP always has both, so its construction order and init are unchanged.
        self.node = mlp(hidden // 2, n_node) if n_node else None
        self.edge = mlp(1, n_edge) if n_edge else None
        self.clamp_max_p = clamp_max_p
        # Start with small noise (p = init_p) so early training is stable; the
        # official ORB posterior does the same with a learned offset.
        logit = math.log(init_p / (1 - init_p))
        for head in (self.node, self.edge):
            if head is None:
                continue
            nn.init.zeros_(head[-1].weight)
            nn.init.constant_(head[-1].bias, logit)

    def _alpha(self, logits):
        p = logits.sigmoid().clamp(min=1e-6, max=self.clamp_max_p)
        return p / (1 - p)

    def forward(self, b):
        node = edge = None
        if self.node is not None:
            node = self._alpha(self.node(self.emb(b.z)))
        if self.edge is not None:
            pos = b.pos.detach()
            r2 = (pos[b.src] - pos[b.dst]).pow(2).sum(-1, keepdim=True)
            edge = self._alpha(self.edge(r2))
        return node, edge



# PaiNN with BLIP layers
class PainnRBF(nn.Module):
    def __init__(self, n, cutoff):
        super().__init__()
        self.register_buffer("freq", torch.arange(1, n + 1) * math.pi / cutoff)

    def forward(self, d):
        return torch.sin(self.freq * d[:, None]) / d[:, None]


def cosine_cutoff(d, rc):
    return 0.5 * (torch.cos(d * math.pi / rc) + 1.0) * (d < rc)


class AlphaRouter:
    """Hands each Bayesian layer its own alpha column (official alpha hooks)."""

    def __init__(self):
        self.n = {"node": 0, "edge": 0}

    def new(self, domain):
        self.n[domain] += 1
        return domain, self.n[domain] - 1


def _a(alphas, slot):
    if alphas is None:
        return None
    domain, k = slot
    return alphas[domain][:, k:k + 1]


class Message(nn.Module):
    def __init__(self, F_, n_rbf, cutoff, router):
        super().__init__()
        self.phi1, self.phi2 = BayesLinear(F_, F_), BayesLinear(F_, 3 * F_)
        self.W = BayesLinear(n_rbf, 3 * F_)
        self.s_phi1, self.s_phi2 = router.new("node"), router.new("node")
        self.s_W = router.new("edge")
        self.F_, self.cutoff = F_, cutoff

    def forward(self, s, v, b, rbf, d, unit, al):
        phi = self.phi2(F.silu(self.phi1(s, _a(al, self.s_phi1))), _a(al, self.s_phi2))
        x = phi[b.src] * self.W(rbf, _a(al, self.s_W)) * cosine_cutoff(d, self.cutoff)[:, None]
        ds, dvv, dvs = torch.split(x, self.F_, -1)
        dv = v[b.src] * dvv[:, None] + dvs[:, None] * unit[..., None]
        s = s + torch.zeros_like(s).index_add_(0, b.dst, ds)
        v = v + torch.zeros_like(v).index_add_(0, b.dst, dv)
        return s, v


class Update(nn.Module):
    def __init__(self, F_, router):
        super().__init__()
        self.U = nn.Linear(F_, F_, bias=False)   # deterministic
        self.V = nn.Linear(F_, F_, bias=False)
        self.a1, self.a2 = BayesLinear(2 * F_, F_), BayesLinear(F_, 3 * F_)
        self.s_a1, self.s_a2 = router.new("node"), router.new("node")
        self.F_ = F_

    def forward(self, s, v, al):
        Uv, Vv = self.U(v), self.V(v)
        Vn = torch.sqrt(Vv.pow(2).sum(1) + 1e-8)
        a = self.a2(F.silu(self.a1(torch.cat([s, Vn], -1), _a(al, self.s_a1))), _a(al, self.s_a2))
        avv, asv, ass = torch.split(a, self.F_, -1)
        return s + asv * (Uv * Vv).sum(1) + ass, v + avv[:, None] * Uv


class BLIPPaiNN(nn.Module):
    def __init__(self, F_=128, n_layers=3, n_rbf=20, cutoff=5.0, bayesian=True,
                 posterior_hidden=64, posterior_layers=2, clamp_max_p=0.8, init_p=0.05):
        super().__init__()
        router = AlphaRouter()
        self.cutoff, self.bayesian = cutoff, bayesian
        self.emb = nn.Embedding(100, F_, padding_idx=0)
        self.rbf = PainnRBF(n_rbf, cutoff)
        self.msg = nn.ModuleList([Message(F_, n_rbf, cutoff, router) for _ in range(n_layers)])
        self.upd = nn.ModuleList([Update(F_, router) for _ in range(n_layers)])
        self.r1, self.r2 = BayesLinear(F_, F_ // 2), BayesLinear(F_ // 2, 1)
        self.s_r1, self.s_r2 = router.new("node"), router.new("node")
        self.n_alpha = dict(router.n)
        self.posterior = (Posterior(router.n["node"], router.n["edge"], posterior_hidden,
                                    posterior_layers, clamp_max_p, init_p) if bayesian else None)
        self.register_buffer("e_shift", torch.zeros(()))  # mean energy per atom
        self.register_buffer("scale", torch.ones(()))     # force RMS

    def alphas(self, b):
        node, edge = self.posterior(b)
        return {"node": node, "edge": edge}

    def forward(self, b, alphas=None, create_graph=False):
        """alphas=None -> deterministic MAP pass; dict -> one stochastic sample."""
        rij = b.pos[b.src] - b.pos[b.dst]
        d = torch.sqrt(rij.pow(2).sum(-1) + 1e-12)
        unit, rbf = rij / d[:, None], self.rbf(d)
        s = self.emb(b.z)
        v = s.new_zeros(s.shape[0], 3, s.shape[1])
        for msg, upd in zip(self.msg, self.upd):
            s, v = msg(s, v, b, rbf, d, unit, alphas)
            s, v = upd(s, v, alphas)
        e_atom = self.r2(F.silu(self.r1(s, _a(alphas, self.s_r1))), _a(alphas, self.s_r2)).squeeze(-1)
        E = s.new_zeros(b.n_graphs).index_add_(0, b.batch, e_atom) * self.scale + self.e_shift * b.n_atoms
        Fp = -torch.autograd.grad(E.sum(), b.pos, create_graph=create_graph)[0]
        return E, Fp


# Phase 3: PaiNN with graph-space (representation) stochasticity
def rep_noise(x, alpha, shared_dims=()):
    """x' = x * (1 + sqrt(alpha) * eps), eps ~ N(0, 1)  =>  x' ~ N(x, alpha x^2).

    alpha: (rows, 1); broadcast over every non-row dimension of x.
    shared_dims: dimensions of x that share one eps draw. For PaiNN vector
    features v (rows, 3, F) pass shared_dims=(1,) so x, y and z of a channel
    are scaled by the same random number - that keeps v equivariant."""
    if alpha is None:
        return x
    shape = list(x.shape)
    for dim in shared_dims:
        shape[dim] = 1
    eps = torch.randn(shape, dtype=x.dtype, device=x.device)
    a = alpha.reshape(alpha.shape[0], *([1] * (x.dim() - 1)))
    return x * (1 + a.sqrt() * eps)


class GSMessage(nn.Module):
    """PaiNN message block with deterministic weights (same shapes as BLIP's
    Message). With edge noise on, the per-edge filter
        x_ij = phi(s_j) * W(rbf(d_ij)) * fcut(d_ij)
    - one invariant scalar per edge and channel, feeding ds, dvv and dvs -
    is perturbed before it is used, so every message from j to i is noisy."""

    def __init__(self, F_, n_rbf, cutoff, router, edge_noise):
        super().__init__()
        self.phi1, self.phi2 = nn.Linear(F_, F_), nn.Linear(F_, 3 * F_)
        self.W = nn.Linear(n_rbf, 3 * F_)
        self.s_edge = router.new("edge") if edge_noise else None
        self.F_, self.cutoff = F_, cutoff

    def forward(self, s, v, b, rbf, d, unit, al):
        phi = self.phi2(F.silu(self.phi1(s)))
        x = phi[b.src] * self.W(rbf) * cosine_cutoff(d, self.cutoff)[:, None]
        if self.s_edge is not None:
            x = rep_noise(x, _a(al, self.s_edge))
        ds, dvv, dvs = torch.split(x, self.F_, -1)
        dv = v[b.src] * dvv[:, None] + dvs[:, None] * unit[..., None]
        s = s + torch.zeros_like(s).index_add_(0, b.dst, ds)
        v = v + torch.zeros_like(v).index_add_(0, b.dst, dv)
        return s, v


class GSUpdate(nn.Module):
    """PaiNN update block with deterministic weights. With node noise on,
    the updated node state (s_i, v_i) is perturbed after the residual update:
        s_i' = s_i * (1 + sqrt(a_i) eps_s),   eps_s ~ N(0, I_F)
        v_i' = v_i * (1 + sqrt(a_i) eps_v),   eps_v ~ N(0, I_F), shared over xyz"""

    def __init__(self, F_, router, node_noise):
        super().__init__()
        self.U = nn.Linear(F_, F_, bias=False)
        self.V = nn.Linear(F_, F_, bias=False)
        self.a1, self.a2 = nn.Linear(2 * F_, F_), nn.Linear(F_, 3 * F_)
        self.s_node = router.new("node") if node_noise else None
        self.F_ = F_

    def forward(self, s, v, al):
        Uv, Vv = self.U(v), self.V(v)
        Vn = torch.sqrt(Vv.pow(2).sum(1) + 1e-8)
        a = self.a2(F.silu(self.a1(torch.cat([s, Vn], -1))))
        avv, asv, ass = torch.split(a, self.F_, -1)
        s, v = s + asv * (Uv * Vv).sum(1) + ass, v + avv[:, None] * Uv
        if self.s_node is not None:
            alpha = _a(al, self.s_node)
            s = rep_noise(s, alpha)
            v = rep_noise(v, alpha, shared_dims=(1,))
        return s, v


class GraphStochasticPaiNN(nn.Module):
    """Phase 3. Same embedding, RBF, cutoff, message/update/readout shapes,
    energy scaling and posterior network as BLIPPaiNN; every weight is
    deterministic and the noise lives on node and/or edge representations.

    Exposes the same interface the training/evaluation code uses for BLIP
    (.bayesian, .alphas(b), .n_alpha, forward(b, alphas=...)), so data loading,
    the training loop, MAP validation, MC testing and all metrics are shared
    code, not reimplementations."""

    def __init__(self, F_=128, n_layers=3, n_rbf=20, cutoff=5.0, stoch="both",
                 posterior_hidden=64, posterior_layers=2, clamp_max_p=0.8, init_p=0.05):
        super().__init__()
        if stoch not in ("node", "edge", "both"):
            raise ValueError(f"stoch must be node | edge | both, got {stoch!r}")
        node_noise, edge_noise = stoch in ("node", "both"), stoch in ("edge", "both")
        router = AlphaRouter()
        self.cutoff, self.bayesian, self.stoch = cutoff, True, stoch
        self.emb = nn.Embedding(100, F_, padding_idx=0)
        self.rbf = PainnRBF(n_rbf, cutoff)
        self.msg = nn.ModuleList([GSMessage(F_, n_rbf, cutoff, router, edge_noise) for _ in range(n_layers)])
        self.upd = nn.ModuleList([GSUpdate(F_, router, node_noise) for _ in range(n_layers)])
        self.r1, self.r2 = nn.Linear(F_, F_ // 2), nn.Linear(F_ // 2, 1)
        self.n_alpha = dict(router.n)
        self.posterior = Posterior(router.n["node"], router.n["edge"], posterior_hidden,
                                   posterior_layers, clamp_max_p, init_p)
        self.register_buffer("e_shift", torch.zeros(()))
        self.register_buffer("scale", torch.ones(()))

    def alphas(self, b):
        node, edge = self.posterior(b)
        return {k: a for k, a in (("node", node), ("edge", edge)) if a is not None}

    def forward(self, b, alphas=None, create_graph=False):
        """alphas=None -> deterministic (mean) pass; dict -> one stochastic sample."""
        rij = b.pos[b.src] - b.pos[b.dst]
        d = torch.sqrt(rij.pow(2).sum(-1) + 1e-12)
        unit, rbf = rij / d[:, None], self.rbf(d)
        s = self.emb(b.z)
        v = s.new_zeros(s.shape[0], 3, s.shape[1])
        for msg, upd in zip(self.msg, self.upd):
            s, v = msg(s, v, b, rbf, d, unit, alphas)
            s, v = upd(s, v, alphas)
        e_atom = self.r2(F.silu(self.r1(s))).squeeze(-1)
        E = s.new_zeros(b.n_graphs).index_add_(0, b.batch, e_atom) * self.scale + self.e_shift * b.n_atoms
        Fp = -torch.autograd.grad(E.sum(), b.pos, create_graph=create_graph)[0]
        return E, Fp


def build_model(args):
    common = dict(F_=args.features, n_layers=args.layers, n_rbf=args.n_rbf, cutoff=args.cutoff)
    post = dict(posterior_hidden=args.posterior_hidden, posterior_layers=args.posterior_layers,
                clamp_max_p=args.clamp_max_p, init_p=args.init_p)
    if args.model == "graph":
        return GraphStochasticPaiNN(**common, stoch=args.stoch, **post)
    return BLIPPaiNN(**common, bayesian=args.model == "blip", **post)


def model_tag(args):
    return {"painn": "painn", "blip": "blip", "graph": f"graph-{args.stoch}"}[args.model]


def kl_total(kl_fn, al):
    """Same KL for BLIP and Phase 3: the per-domain mean, summed over the
    domains (node / edge) the model actually has."""
    return sum(kl_fn(a) for a in al.values())


# Training, validation, testing
def data_loss(Ep, Fp, b, args):
    L = {"mae": F.l1_loss, "mse": F.mse_loss}[args.loss]
    return args.energy_weight * L(Ep, b.E) + args.force_weight * L(Fp, b.F)


def evaluate(model, idx, data, args, device, mc_samples=0):
    """Returns val/test loss and MAEs. mc_samples=0 -> MAP; >0 -> MC mean + variance."""
    model.eval()
    rng = np.random.default_rng(0)
    out = {k: [] for k in ["E", "Ep", "Ev", "F", "Fp", "Fv"]}
    alpha_seen = {}  # domain -> list of alpha tensors over the evaluated inputs
    loss_sum = n = 0
    for b in make_batches(idx, data, args.eval_batch, False, rng, args.cutoff, device):
        if mc_samples and model.bayesian:
            with torch.no_grad():
                al = model.alphas(b)  # amortised: posterior run once per input
            for dom, a in al.items():
                alpha_seen.setdefault(dom, []).append(a.detach().cpu())
            Es, Fs = zip(*[(e.detach(), f.detach()) for e, f in
                           (model(b, alphas=al) for _ in range(mc_samples))])
            Es, Fs = torch.stack(Es), torch.stack(Fs)
            Ep, Fp, Ev, Fv = Es.mean(0), Fs.mean(0), Es.var(0), Fs.var(0)
        else:
            Ep, Fp = model(b)
            Ep, Fp = Ep.detach(), Fp.detach()
            Ev, Fv = torch.zeros_like(Ep), torch.zeros_like(Fp)
        loss_sum += data_loss(Ep, Fp, b, args).item() * b.n_graphs
        n += b.n_graphs
        for k, t in zip(out, [b.E, Ep, Ev, b.F, Fp, Fv]):
            out[k].append(t.cpu())
    out = {k: torch.cat(v) for k, v in out.items()}
    res = {"loss": loss_sum / n,
           "E_MAE_meV": (out["E"] - out["Ep"]).abs().mean().item() * KCAL_TO_MEV,
           "F_MAE_meV/A": (out["F"] - out["Fp"]).abs().mean().item() * KCAL_TO_MEV,
           "E_RMSE_meV": (out["E"] - out["Ep"]).pow(2).mean().sqrt().item() * KCAL_TO_MEV,
           "F_RMSE_meV/A": (out["F"] - out["Fp"]).pow(2).mean().sqrt().item() * KCAL_TO_MEV}
    if mc_samples and model.bayesian:
        res.update(uncertainty_metrics(out, data[1].shape[1]))
        res.update(alpha_spread(alpha_seen))
    return res


# Metric definitions version, written into every result JSON. Version 1 (the
# inherited BLIP code) scored ECE on the per-frame MEAN force over atoms, which
# is ~0 for both DFT and any translation-invariant model, so it always came out
# near 0.25 regardless of the uncertainty. Results with metrics_version < 2
# must be re-scored (main("--eval-only <checkpoint.pt> ...")) before reporting.
METRICS_VERSION = 2
COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)  # same levels as ml/utils/uq_metrics.py


def _calibration(err, std):
    """Central-interval coverage and ECE, same definition as the team harness
    (ml/utils/uq_metrics.py: _coverage_table / _ece): z = err/std per value;
    coverage(p) = fraction with |z| <= Phi^-1(0.5 + p/2); ECE = mean |coverage(p) - p|."""
    from scipy.stats import norm
    z = np.abs(err / std)
    cov = {p: float(np.mean(z <= norm.ppf(0.5 + p / 2))) for p in COVERAGE_LEVELS}
    return cov, float(np.mean([abs(c - p) for p, c in cov.items()]))


def uncertainty_metrics(o, n_atoms):
    from scipy.stats import spearmanr
    var = o["Fv"].clamp_min(1e-12)
    err = o["F"] - o["Fp"]
    res = {"metrics_version": METRICS_VERSION,
           "F_NLL": (0.5 * err ** 2 / var + 0.5 * torch.log(2 * math.pi * var)).mean().item()}
    # Calibration per force component (frame x atom x xyz), like force_mae_component
    f_err, f_std = err.reshape(-1).double().numpy(), var.sqrt().reshape(-1).double().numpy()
    cov, res["F_ECE"] = _calibration(f_err, f_std)
    for p, c in cov.items():
        res[f"F_coverage_{int(p * 100)}"] = c
    e_err = (o["E"] - o["Ep"]).double().numpy()
    e_std = o["Ev"].clamp_min(1e-12).sqrt().double().numpy()
    _, res["E_ECE"] = _calibration(e_err, e_std)
    # Spearman: predicted variance vs squared error per structure (active-learning signal)
    se = (err ** 2).reshape(-1, n_atoms * 3).mean(1).numpy()
    pv = var.reshape(-1, n_atoms * 3).mean(1).numpy()
    res["F_spearman"] = float(spearmanr(pv, se)[0])
    res["E_spearman"] = float(spearmanr(e_std, np.abs(e_err))[0])
    res["E_std_mean_meV"] = o["Ev"].sqrt().mean().item() * KCAL_TO_MEV
    res["F_std_mean_meV/A"] = var.sqrt().mean().item() * KCAL_TO_MEV
    return res


def alpha_spread(alpha_seen):
    """How much the noise level alpha varies across the evaluated inputs.
    std/mean ~ 0 means the posterior gives every input the same noise, i.e. the
    uncertainty is not input-dependent (posterior collapsed to the prior)."""
    res = {}
    for dom, parts in alpha_seen.items():
        a = torch.cat(parts).reshape(-1).double()
        res[f"alpha_{dom}_mean"] = a.mean().item()
        res[f"alpha_{dom}_std"] = a.std().item()
        res[f"alpha_{dom}_min"] = a.min().item()
        res[f"alpha_{dom}_max"] = a.max().item()
        res[f"alpha_{dom}_rel_spread"] = (a.std() / a.mean()).item()
    return res


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        if self.decay <= 0:
            self.shadow.load_state_dict(model.state_dict())
            return
        for ps, pm in zip(self.shadow.parameters(), model.parameters()):
            ps.mul_(self.decay).add_(pm.detach(), alpha=1 - self.decay)
        for bs, bm in zip(self.shadow.buffers(), model.buffers()):
            bs.copy_(bm)


def plot_history(csv_path, png_path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot (history CSV is saved)")
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    col = lambda k: [float(r[k]) for r in rows]
    ep = col("epoch")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(ep, col("train_loss"), label="train", color="#2563eb")
    ax[0].plot(ep, col("val_loss"), label="validation", color="#dc2626")
    ax[0].set(title="Loss (0.1·L(E) + L(F), kcal/mol units)", xlabel="epoch", yscale="log")
    ax[1].plot(ep, col("val_F_MAE_meV/A"), color="#dc2626", label="validation")
    ax[1].set(title="Force MAE (meV/Å)", xlabel="epoch", yscale="log")
    ax[2].plot(ep, col("val_E_MAE_meV"), color="#dc2626", label="validation")
    ax[2].set(title="Energy MAE (meV)", xlabel="epoch", yscale="log")
    if any(float(r["train_kl"]) > 0 for r in rows):
        twin = ax[0].twinx()
        twin.plot(ep, col("train_kl"), color="#6b7280", ls="--", lw=1, label="KL (right)")
        twin.set_ylabel("β·KL")
        twin.legend(loc="upper center", frameon=False)
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"saved {png_path}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("data")
    g.add_argument("--dataset", choices=["rmd17", "md17"], default="rmd17")
    g.add_argument("--molecule", default="aspirin")
    g.add_argument("--npz", default=None, help="local .npz (rMD17 or sGDML format)")
    g.add_argument("--data-dir", default="data")
    g.add_argument("--split", type=int, default=1, help="rMD17 official split 1-5")
    g.add_argument("--n-train", type=int, default=950)
    g.add_argument("--n-val", type=int, default=50)
    g.add_argument("--n-test", type=int, default=1000, help="original MD17 only")
    g.add_argument("--synthetic", action="store_true")
    g = ap.add_argument_group("team repo data (HydrogenProductionMaterials_Discovery gold files)")
    g.add_argument("--repo-dir", default=None,
                   help="repository root; reads data/gold/md17/<molecule>_<theory>.npz")
    g.add_argument("--gold", default=None, help="explicit path to a gold .npz")
    g.add_argument("--theory", default="dft", help="dft | ccsd | ccsd_t (part of dataset identity)")
    g.add_argument("--train-subset", type=int, default=10000, help="team protocol: 10000 (0 = all)")
    g.add_argument("--val-subset", type=int, default=5000, help="team protocol: 5000 (0 = all)")
    g.add_argument("--test-subset", type=int, default=0, help="0 = whole test split (team protocol)")
    g.add_argument("--subset-seed", type=int, default=42)
    g = ap.add_argument_group("PaiNN")
    g.add_argument("--features", type=int, default=128)
    g.add_argument("--layers", type=int, default=3)
    g.add_argument("--n-rbf", type=int, default=20)
    g.add_argument("--cutoff", type=float, default=5.0)
    g = ap.add_argument_group("which phase")
    g.add_argument("--model", choices=["painn", "blip", "graph"], default="blip",
                   help="painn = Phase 1, blip = Phase 2 (weight noise), graph = Phase 3 (representation noise)")
    g.add_argument("--stoch", choices=["node", "edge", "both"], default="both",
                   help="Phase 3 only: where the representation noise goes")
    g = ap.add_argument_group("BLIP / Phase 3 noise (defaults = official BLIP PaiNN settings)")
    g.add_argument("--no-blip", action="store_true", help="old spelling of --model painn")
    g.add_argument("--prior-p", type=float, default=0.5)
    g.add_argument("--kl-beta", type=float, default=10.0, help="weight of the KL term")
    g.add_argument("--clamp-max-p", type=float, default=0.8, help="max p -> alpha <= 4")
    g.add_argument("--init-p", type=float, default=0.05, help="starting noise level p (alpha=p/(1-p))")
    g.add_argument("--posterior-hidden", type=int, default=64)
    g.add_argument("--posterior-layers", type=int, default=2)
    g.add_argument("--mc-samples", type=int, default=100)
    g = ap.add_argument_group("optimisation (defaults = official PaiNN settings)")
    g.add_argument("--epochs", type=int, default=2000)
    g.add_argument("--batch-size", type=int, default=10)
    g.add_argument("--eval-batch", type=int, default=50)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--loss", choices=["mae", "mse"], default="mae")
    g.add_argument("--energy-weight", type=float, default=0.1)
    g.add_argument("--force-weight", type=float, default=1.0)
    g.add_argument("--lr-factor", type=float, default=0.5)
    g.add_argument("--patience", type=int, default=30)
    g.add_argument("--min-lr", type=float, default=1e-7)
    g.add_argument("--ema", type=float, default=0.0, help="EMA decay, e.g. 0.99 (0 = off, as official)")
    g.add_argument("--grad-clip", type=float, default=0.0, help="0 = off")
    g.add_argument("--log-every", type=int, default=10)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--eval-only", default=None, metavar="CKPT",
                   help="no training: re-score a saved .pt on the test split with the current metrics. "
                        "Model/data settings come from the checkpoint; --repo-dir, --gold, --mc-samples, "
                        "--test-subset, --eval-batch, --device and --out can be overridden")
    g.add_argument("--out", default="runs")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # parse_known_args: ignore extra arguments Jupyter/Colab put on sys.argv (e.g. -f kernel.json)
    args, unknown = ap.parse_known_args(argv)
    unknown = [u for u in unknown if not (u == "-f" or u.endswith(".json"))]
    if unknown:
        ap.error(f"unrecognized arguments: {' '.join(unknown)}")
    if args.no_blip:
        args.model = "painn"
    return args


def _in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None and "IPKernelApp" in get_ipython().config
    except Exception:
        return False


RESCORE_OVERRIDES = ("repo_dir", "gold", "mc_samples", "test_subset", "eval_batch", "device", "out")


def rescore(cli):
    """Re-evaluate a saved checkpoint on its test split with the current metric
    code, writing <out>/<tag>.json exactly as a training run would. The test
    frames, model and settings are rebuilt from the checkpoint's own saved args,
    so the result is comparable with a fresh run of the same configuration."""
    state = torch.load(cli.eval_only, map_location="cpu", weights_only=False)
    saved = parse_args([])  # defaults, so old checkpoints missing newer options still work
    vars(saved).update(state["args"])
    if "model" not in state["args"]:  # checkpoints from Blip_model_building.ipynb
        saved.model = "painn" if state["args"].get("no_blip") else "blip"
    defaults = vars(parse_args([]))
    for k in RESCORE_OVERRIDES:
        if getattr(cli, k) != defaults[k]:
            setattr(saved, k, getattr(cli, k))
    if saved.repo_dir is None and saved.gold is None and not saved.synthetic:
        raise SystemExit("checkpoint has no data location; pass --repo-dir (or --gold)")
    saved.eval_only, device = cli.eval_only, torch.device(saved.device)

    if saved.gold or saved.repo_dir:
        z, R, E, Fo, (_, _, test_idx) = load_gold(saved)
    else:  # same seeded split as training
        z, R, E, Fo, (_, test_idx) = load_md17(saved)
    data = (torch.tensor(z), torch.tensor(R), torch.tensor(E, dtype=torch.float32), torch.tensor(Fo))

    model = build_model(saved).to(device)
    model.load_state_dict(state["model"])  # includes e_shift / scale
    torch.manual_seed(saved.seed)
    print(f"re-scoring {cli.eval_only} ({model_tag(saved)}, best epoch {state['epoch']}) "
          f"on {len(test_idx)} test frames, {saved.mc_samples} MC samples")
    results = {"best_epoch": state["epoch"], "MAP": evaluate(model, test_idx, data, saved, device)}
    if model.bayesian:
        results["MC"] = evaluate(model, test_idx, data, saved, device, mc_samples=saved.mc_samples)
    print("TEST RESULTS\n" + json.dumps(results, indent=2))
    tag = os.path.splitext(os.path.basename(cli.eval_only))[0]
    os.makedirs(saved.out, exist_ok=True)
    with open(os.path.join(saved.out, tag + ".json"), "w") as f:
        json.dump({"args": vars(saved), "test": results, "rescored_from": cli.eval_only}, f, indent=2)
    return results


def main(argv=None):
    """Command line:  python stochastic_md17.py --repo-dir . --molecule ethanol --model graph --stoch node
    Notebook:      main("--repo-dir . --molecule ethanol --model graph --stoch node")
    Returns the test-results dict (also written to <out>/<tag>.json)."""
    if isinstance(argv, str):
        argv = argv.split()
    args = parse_args(argv)
    if args.eval_only:
        return rescore(args)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    if args.gold or args.repo_dir:
        z, R, E, Fo, (tr_idx, va_idx, test_idx) = load_gold(args)
    else:
        z, R, E, Fo, (pool, test_idx) = load_md17(args)
        pool = rng.permutation(pool)
        tr_idx, va_idx = pool[:args.n_train], pool[args.n_train:args.n_train + args.n_val]
    data = (torch.tensor(z), torch.tensor(R), torch.tensor(E, dtype=torch.float32), torch.tensor(Fo))

    model = build_model(args).to(device)
    model.e_shift.fill_(float(E[tr_idx].mean() / len(z)))
    model.scale.fill_(float(np.sqrt((Fo[tr_idx] ** 2).mean())))
    kl_fn = KL(args.prior_p, args.kl_beta)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=args.lr_factor, patience=args.patience,
                                                       threshold=1e-4, min_lr=args.min_lr)
    ema = EMA(model, args.ema)

    if args.synthetic:
        name = "synthetic"
    elif args.gold or args.repo_dir:
        name = f"gold_{args.molecule}_{args.theory}" if not args.gold else \
            "gold_" + os.path.splitext(os.path.basename(args.gold))[0]
    else:
        name = f"{args.dataset}_{args.molecule}"
    tag = f"{name}_{model_tag(args)}_seed{args.seed}"
    os.makedirs(args.out, exist_ok=True)
    ckpt, hist_csv = os.path.join(args.out, tag + ".pt"), os.path.join(args.out, tag + "_history.csv")
    fields = ["epoch", "lr", "train_loss", "train_kl", "val_loss", "val_E_MAE_meV",
              "val_F_MAE_meV/A", "alpha_node_mean", "alpha_edge_mean", "time_s"]
    with open(hist_csv, "w", newline="") as f:
        csv.writer(f).writerow(fields)

    print(f"{tag}: atoms={len(z)} train={len(tr_idx)} val={len(va_idx)} test={len(test_idx)} "
          f"params={sum(p.numel() for p in model.parameters()):,} device={device}")
    if model.bayesian:
        where = "noisy representations" if args.model == "graph" else "Bayesian layers"
        print(f"{where}: {model.n_alpha['node']} node-wise, {model.n_alpha['edge']} edge-wise")

    best, t0 = float("inf"), time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = tot_kl = n = 0
        a_node = a_edge = 0.0
        for b in make_batches(tr_idx, data, args.batch_size, True, rng, args.cutoff, device):
            al = model.alphas(b) if model.bayesian else None
            Ep, Fp = model(b, alphas=al, create_graph=True)
            dl = data_loss(Ep, Fp, b, args)
            kl = kl_total(kl_fn, al) if model.bayesian else torch.zeros((), device=device)
            loss = dl + kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            ema.update(model)
            tot += dl.item() * b.n_graphs
            tot_kl += kl.item() * b.n_graphs
            n += b.n_graphs
            if model.bayesian:  # a domain the model has no noise in is logged as 0
                a_node += al["node"].mean().item() * b.n_graphs if "node" in al else 0.0
                a_edge += al["edge"].mean().item() * b.n_graphs if "edge" in al else 0.0

        val = evaluate(ema.shadow, va_idx, data, args, device)  # MAP, as official validation
        sched.step(val["loss"])
        if val["loss"] < best:
            best = val["loss"]
            torch.save({"model": ema.shadow.state_dict(), "args": vars(args), "epoch": ep}, ckpt)
        lr = opt.param_groups[0]["lr"]
        row = [ep, lr, tot / n, tot_kl / n, val["loss"], val["E_MAE_meV"], val["F_MAE_meV/A"],
               a_node / n, a_edge / n, time.time() - t0]
        with open(hist_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)
        if ep == 1 or ep % args.log_every == 0:
            print(f"ep {ep:5d} | train loss {tot / n:.4f} KL {tot_kl / n:.4f} | val loss {val['loss']:.4f} "
                  f"E {val['E_MAE_meV']:.2f} meV F {val['F_MAE_meV/A']:.2f} meV/Å | lr {lr:.1e} "
                  f"| alpha n/e {a_node / n:.3f}/{a_edge / n:.3f} | {time.time() - t0:.0f}s", flush=True)
        if lr <= args.min_lr * 1.0001:
            print("learning rate reached min-lr, stopping")
            break

    plot_history(hist_csv, os.path.join(args.out, tag + "_curves.png"), tag)

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    results = {"best_epoch": state["epoch"], "MAP": evaluate(model, test_idx, data, args, device)}
    if model.bayesian:
        results["MC"] = evaluate(model, test_idx, data, args, device, mc_samples=args.mc_samples)
    print("TEST RESULTS\n" + json.dumps(results, indent=2))
    with open(os.path.join(args.out, tag + ".json"), "w") as f:
        json.dump({"args": vars(args), "test": results}, f, indent=2)
    return results


if __name__ == "__main__":
    if _in_notebook():
        # Pasted into a notebook cell: don't start training with default settings.
        print("Phase 1/2/3 code loaded. Start training in a new cell, e.g.:\n"
              '  main("--repo-dir . --molecule ethanol --theory dft --model graph --stoch node --mc-samples 20")')
    else:
        main()