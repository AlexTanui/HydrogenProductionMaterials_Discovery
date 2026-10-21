"""Phase 3 - graph-space stochasticity (SCRUM-32).

**Status: first draft, not yet a team decision.** CLAUDE.md says Phase 3's
exact mechanism should come out of the literature review (SCRUM-33/34),
and that review hadn't landed as of 2026-10-22 (SCRUM-33 was reset to To
Do - the file it was supposed to produce, docs/literature_review.md, is
still empty; SCRUM-34 is Dongxiao's in-progress deep-dive). This file
exists so Phase 3 isn't blocked on the literature review's own kickoff -
it is one reasonable mechanism, not the mechanism. Revisit once Dongxiao's
write-up lands; `RepresentationNoise` below is the one piece to swap if
the team picks something else - nothing else in this file or the shared
harness (ml/training/train.py, evaluate.py, benchmark.py) needs to change
for that.

Extends the Phase 1 backbone (ml/models/mpnn.py) the same way Phase 2
does: identical embedding, RBF, and message/update shapes, so any
difference between phases is attributable to *where the randomness
lives*, not to three unrelated architectures (CLAUDE.md's three-phase-plan
table). Compare the two directly:

  - Phase 2 (ml/models/blip.py): deterministic representations, stochastic
    *weights* - every nn.Linear in the message/update MLPs is replaced by
    StochasticLinear, which samples a fresh weight matrix per forward pass.
  - Phase 3 (here): deterministic weights - ordinary nn.Linear throughout
    message/update, identical to MPNNLayer - stochastic *representations*.
    Each layer's node embedding gets input-dependent Gaussian noise added
    after the deterministic update step.

Concretely, per layer: h_new = h + update_mlp(...) is exactly MPNNLayer's
deterministic step; RepresentationNoise then returns
`h_new + scale(h_new) * sigma * eps`, with `scale` input-dependent (one
learned gate per node, same pattern as BLIP's `noise_gate`) and `sigma`
a learned, shared-per-channel magnitude (analogous to BLIP's `weight_rho`,
but there is one representation per node rather than one weight per
connection, so a single learned magnitude plus a per-node gate is the
natural analogue rather than a full per-entry parameter).

**Learned from SCRUM-61, applied here from the start:** BLIP's bug was
that `noise_gate` carried no KL term at all, so plain MSE training shrank
it unopposed until the predicted variance collapsed. Here the KL term
below is computed from `scale(h) * sigma`, not from `sigma` alone - so the
gate's output is inside the regularized quantity and gets a KL gradient
on every forward pass, not just the magnitude parameter. This does not
make Phase 3 correctly calibrated by construction (nothing about a KL
term guarantees that), but it avoids walking into the exact same
unregularized-gate hole on day one.

The mean path is untouched by any of this: `h_new` above is bit-for-bit
what Phase 1's MPNNLayer would have produced from the same input, so a
`kl_weight` of 0 (the default) reduces this file to Phase 1's backbone
plus zero-centered noise around the mean - not an architecture change in
disguise.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_ATOMIC_NUMBER = 100

# Same choice/reasoning as ml/models/blip.py's: wide enough not to dominate
# the data term, not so wide the KL term is meaningless.
PRIOR_SIGMA = 0.1


class GaussianRBF(nn.Module):
    """Identical to ml/models/mpnn.py's - kept local so this file has no
    import-time dependency on the Phase 1 module, only a structural one."""

    def __init__(self, num_rbf: int = 50, cutoff: float = 5.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_rbf)
        width = centers[1] - centers[0] if num_rbf > 1 else torch.tensor(cutoff)
        self.register_buffer("centers", centers)
        self.width = width

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        d = distances.view(-1, 1)
        return torch.exp(-((d - self.centers.view(1, -1)) ** 2) / (2 * self.width ** 2))


class RepresentationNoise(nn.Module):
    """Adds input-dependent Gaussian noise to a node representation.

        scale_i = softplus(gate(h_i))     # per node, input-dependent - BLIP's noise_gate pattern
        sigma   = softplus(log_sigma)     # learned, shared across channels
        h_out_i = h_i + scale_i * sigma * eps_i,   eps_i ~ N(0, I)

    A fresh `eps` is sampled on every forward call, train or eval - same
    reasoning as BLIP's weight noise: the uncertainty estimate comes from
    explicitly repeated sampling (`predict_distribution` below), not from
    disabling noise at `eval()` the way dropout does.
    """

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.gate = nn.Linear(hidden_channels, 1)
        # Small at init (softplus(-3) ~= 0.049), same spirit as BLIP's
        # weight_rho init: noise starts small rather than swamping the
        # signal before training has shaped it.
        self.log_sigma = nn.Parameter(torch.full((hidden_channels,), -3.0))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.gate(h))       # (n_nodes, 1)
        sigma = F.softplus(self.log_sigma)      # (hidden_channels,)
        eps = torch.randn_like(h)
        return h + scale * sigma * eps

    def kl_divergence(self, h: torch.Tensor) -> torch.Tensor:
        """KL(N(0, (scale(h)*sigma)^2) || N(0, PRIOR_SIGMA^2)), summed over
        nodes and channels. Zero-mean on both sides (the noise this module
        adds is zero-mean by construction - h_out's mean is h itself), so
        this is the var-ratio-only term of the usual two-Gaussian KL, not
        the full form ml/models/blip.py's StochasticLinear.kl_divergence
        uses for its non-zero-mean weight posterior.

        Takes `h` explicitly - the KL depends on this forward call's own
        per-node gate output, not just the shared `sigma` parameter - see
        the module docstring for why that matters.
        """
        scale = F.softplus(self.gate(h))
        sigma = F.softplus(self.log_sigma)
        sigma_eff = scale * sigma
        var_ratio = (sigma_eff / PRIOR_SIGMA) ** 2
        return 0.5 * (var_ratio - 1 - torch.log(var_ratio)).sum()


class GraphStochasticLayer(nn.Module):
    """Same structure as ml/models/mpnn.py's MPNNLayer - ordinary
    deterministic nn.Linear throughout - plus a RepresentationNoise
    perturbation of the updated node representation. Compare to
    ml/models/blip.py's BLIPLayer, which keeps representations
    deterministic and makes the *weights* stochastic instead."""

    def __init__(self, hidden_channels: int, num_rbf: int):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_channels + num_rbf, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.noise = RepresentationNoise(hidden_channels)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_rbf: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        messages = self.message_mlp(torch.cat([h[col], edge_rbf], dim=-1))

        aggregated = torch.zeros_like(h)
        aggregated.index_add_(0, row, messages)

        h_new = h + self.update_mlp(torch.cat([h, aggregated], dim=-1))
        return self.noise(h_new), self.noise.kl_divergence(h_new)


class GraphStochastic(nn.Module):
    """Phase 3: Phase 1's MPNN backbone with noise added to node
    representations instead of weights (see module docstring).

    Exposes the same predict_energy / predict_energy_and_forces interface
    as every other phase/bake-off entry (one stochastic sample per call -
    a drop-in for the existing deterministic train/eval loop in
    ml/training/train.py), plus predict_distribution for repeated-sampling
    uncertainty, same contract as ml/models/blip.py's.

    `kl_weight` (default 0.0) is read by ml/training/train.py's batch_loss
    exactly the way it reads BLIP's - `getattr(inner_model, "kl_weight",
    0.0)` - so a model that doesn't define it is never penalised; no
    harness change needed to add this model. Deliberately NOT wiring up
    `train_num_mc_samples`/`nll_warmup_epochs` yet (BLIP's NLL-training
    hooks, SCRUM-61): this file ships Phase 2's *first* real shape -
    deterministic-mean training plus eval-time MC sampling for
    uncertainty - before attempting NLL training, the same order Phase 2
    itself went through. Revisit once this mechanism itself is settled;
    there is no reason to debug two new things (the mechanism and
    NLL-training-on-it) at once.

    **`kl_divergence()` takes no arguments**, matching
    ml/training/train.py's `batch_loss` exactly (`inner.kl_divergence()`,
    called right after `predict_energy_and_forces` in the same batch) -
    but unlike BLIP's version, the KL here genuinely depends on this
    forward's own per-node gate outputs (module docstring), which a
    no-arg method can't take as a parameter. Resolved the same way a
    `grad` cache would be: `predict_energy` computes and stashes it on
    `self._last_kl` during its layer loop; `kl_divergence()` just returns
    that. Safe only because the one real caller (`batch_loss`) always
    calls it immediately after this same forward pass, same batch, train
    mode - true today and enforced by there being no other caller
    anywhere in ml/ (checked at the time this was written). A future
    caller that reads `kl_divergence()` without an immediately preceding
    forward pass would get a stale value with no error - worth an
    assertion if a second caller ever appears.
    """

    def __init__(self, hidden_channels: int = 128, num_layers: int = 4,
                 num_rbf: int = 50, cutoff: float = 5.0,
                 num_mc_samples: int = 20, kl_weight: float = 0.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.num_mc_samples = num_mc_samples
        self.kl_weight = kl_weight
        self.embedding = nn.Embedding(MAX_ATOMIC_NUMBER + 1, hidden_channels)
        self.rbf = GaussianRBF(num_rbf=num_rbf, cutoff=cutoff)
        self.layers = nn.ModuleList([GraphStochasticLayer(hidden_channels, num_rbf) for _ in range(num_layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )
        # Set by predict_energy, read by kl_divergence() - see that
        # method's docstring for why a no-arg KL method needs a cache.
        self._last_kl: torch.Tensor = torch.tensor(0.0)

    def config(self) -> dict:
        """Constructor kwargs, for embedding in a checkpoint. See ml/models/registry.py."""
        return {
            "hidden_channels": self.hidden_channels,
            "num_layers": self.num_layers,
            "num_rbf": self.num_rbf,
            "cutoff": self.cutoff,
            "num_mc_samples": self.num_mc_samples,
            "kl_weight": self.kl_weight,
        }

    def predict_energy(self, z: torch.Tensor, edge_index: torch.Tensor,
                        edge_attr: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """One stochastic sample. `edge_attr` must already be RBF-expanded
        distance - see ml/models/mpnn.py's docstring, same contract here."""
        h = self.embedding(z.view(-1))

        kl_total = torch.tensor(0.0, device=h.device)
        for layer in self.layers:
            h, layer_kl = layer(h, edge_index, edge_attr)
            kl_total = kl_total + layer_kl
        self._last_kl = kl_total

        atomic_energies = self.readout(h).view(-1)

        if batch is None:
            return atomic_energies.sum().view(1)

        n_graphs = int(batch.max().item()) + 1
        total_energy = torch.zeros(n_graphs, dtype=atomic_energies.dtype, device=atomic_energies.device)
        total_energy.index_add_(0, batch, atomic_energies)
        return total_energy

    def predict_energy_and_forces(self, z: torch.Tensor, pos: torch.Tensor,
                                   edge_index: torch.Tensor, edge_attr: torch.Tensor,
                                   batch: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Same live-RBF-recompute pattern as mpnn.py/blip.py - forces need
        a real gradient path from pos, not the precomputed edge_attr."""
        pos = pos.requires_grad_(True)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        live_edge_attr = self.rbf(dist)

        energy = self.predict_energy(z, edge_index, live_edge_attr, batch=batch)

        grad_outputs = torch.ones_like(energy)
        (dE_dR,) = torch.autograd.grad(
            outputs=energy, inputs=pos, grad_outputs=grad_outputs,
            create_graph=self.training, retain_graph=True,
        )
        forces = -dE_dR
        return energy, forces

    @torch.no_grad()
    def predict_distribution(self, z: torch.Tensor, pos: torch.Tensor,
                              edge_index: torch.Tensor, edge_attr: torch.Tensor,
                              batch: Optional[torch.Tensor] = None,
                              num_mc_samples: Optional[int] = None,
                              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run `num_mc_samples` independent stochastic forward passes and
        return (E_mean, E_var, F_mean, F_var). Same contract and same
        no_grad/enable_grad reasoning as ml/models/blip.py's."""
        n = num_mc_samples or self.num_mc_samples
        energies, forces = [], []
        for _ in range(n):
            with torch.enable_grad():
                pos_sample = pos.detach().requires_grad_(True)
                e, f = self.predict_energy_and_forces(z, pos_sample, edge_index, edge_attr, batch=batch)
            energies.append(e.detach())
            forces.append(f.detach())

        e_stack = torch.stack(energies, dim=0)
        f_stack = torch.stack(forces, dim=0)
        return e_stack.mean(0), e_stack.var(0, unbiased=True), f_stack.mean(0), f_stack.var(0, unbiased=True)

    def kl_divergence(self) -> torch.Tensor:
        """Total KL from the most recent `predict_energy` call. See the
        class docstring for why this is a cache read rather than a fresh
        computation - matches BLIP's no-arg `kl_divergence()` contract
        that ml/training/train.py's `batch_loss` calls against."""
        return self._last_kl
