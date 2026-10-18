"""Phase 2 — BLIP-style weight-space stochasticity (SCRUM-31).

Extends the Phase 1 backbone (`ml/models/mpnn.py`) per CLAUDE.md's
three-phase plan: same message-passing structure, but the message/update
MLPs' linear layers carry an **input-dependent Gaussian** distribution
over their weights instead of a fixed weight matrix. Every forward pass
samples a fresh weight matrix via the reparameterization trick, so
running the same input through the model `num_mc_samples` times produces
a *distribution* of energy/force predictions - the spread of that
distribution is BLIP's uncertainty estimate.

This file intentionally extends MPNN's exact layer structure (same
embedding, same RBF, same message/update shapes) rather than introducing
new architecture, so that any difference between Phase 1 and Phase 2
results is attributable to *where the randomness lives*, not to two
unrelated models - this is the whole point of the three-phase comparison
(CLAUDE.md's three-phase-plan table).

**Status (updated once the three gaps below were closed):**
  - The KL-divergence / complexity-cost term is wired into the training
    loss: `ml/training/train.py`'s `batch_loss` adds
    `kl_weight * model.kl_divergence() / n_batches` whenever the wrapped
    model defines both `kl_weight` and `kl_divergence()`. `kl_weight`
    defaults to 0.0 (see `__init__`) and is set per-run in
    `experiments/configs/phase2_blip.yaml`'s `model:` block - starting
    value, not a tuned one; watch `train_kl`/`val_kl` in the history CSV.
  - `ml/training/evaluate.py` now calls `predict_distribution` (via
    `ScaledEnergyModel.supports_distribution`) instead of one stochastic
    sample, so `ModelOutput.E_var`/`F_var` are populated for real.
  - `ml/utils/uq_metrics.py` implements ECE, prediction-interval coverage,
    Spearman uncertainty-error correlation, a risk-coverage curve,
    Gaussian NLL/CRPS, and sharpness, reading those `E_var`/`F_var`
    fields. Its distributional assumption (Normal, from mean/variance,
    not raw MC samples) is provisional pending Dongxiao's sign-off - see
    that module's docstring and SCRUM-39.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_ATOMIC_NUMBER = 100

# Gaussian prior over weights, N(0, PRIOR_SIGMA^2) - standard Bayes-by-backprop
# choice: wide enough not to dominate the data term, not so wide the KL term
# is meaningless. Used only by kl_divergence(); see the module docstring.
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


class StochasticLinear(nn.Module):
    """A linear layer whose weight is input-dependent Gaussian.

    For one forward call, per row r (r is one edge for the message MLPs,
    one atom for the update MLPs):

        scale_r = softplus(noise_gate(x_r))                    # own row, not the batch mean
        y_r     = x_r @ W_mu.T + b + scale_r * (x_r @ (softplus(W_rho) * eps).T)

    `W_mu`/`W_rho` are learned, per-weight parameters (standard
    Bayes-by-backprop); `noise_gate` is a small learned function of each
    row's own input that scales how much weight-noise that row picks up,
    which is what makes the stochasticity *input-dependent* rather than a
    fixed per-weight prior applied identically regardless of what's being
    processed. `softplus` keeps both the weight's noise magnitude and the
    input-dependent scale positive without a hard clamp.

    Gating per row rather than per batch-mean is deliberate (see
    docs/reviews/blip-batch-dependence.md): gating on `x.mean(dim=0)`
    made one molecule's energy/forces depend on whichever other molecules
    shared its batch, which changed test results with `--batch-size` and
    confounded the Phase 1 vs Phase 2 comparison. Gating per row removes
    that - a molecule's prediction no longer depends on its batch-mates
    (verified to float64 roundoff; see the review doc's repro script).
    One `eps` draw is still shared across all rows in a call, which only
    correlates their noise samples, not their inputs, so it's unaffected.

    A fresh `eps` is sampled on every forward call, train or eval - BLIP's
    uncertainty comes from explicitly repeated sampling
    (`predict_distribution`, below), not from switching off noise at
    `eval()` the way dropout does.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.noise_gate = nn.Linear(in_features, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Same fan-in init as nn.Linear's default for the mean weight, so
        # this starts out behaving like an ordinary linear layer; rho
        # starts very negative (softplus(-5) ~= 0.0067) so weight noise is
        # small at initialisation rather than swamping the signal before
        # training has had a chance to shape it.
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -5.0)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_sigma = F.softplus(self.weight_rho)
        eps = torch.randn_like(self.weight_mu)

        row_scale = F.softplus(self.noise_gate(x))  # (rows, 1): each row's own input
        return F.linear(x, self.weight_mu, self.bias) + row_scale * F.linear(x, weight_sigma * eps)

    def kl_divergence(self) -> torch.Tensor:
        """KL(q(W) || N(0, PRIOR_SIGMA^2)), summed over all weight entries.

        Closed form for two diagonal Gaussians - see the module docstring
        for how this is meant to be combined into the training loss (not
        done automatically here).
        """
        sigma = F.softplus(self.weight_rho)
        var_ratio = (sigma / PRIOR_SIGMA) ** 2
        mu_term = (self.weight_mu / PRIOR_SIGMA) ** 2
        return 0.5 * (var_ratio + mu_term - 1 - torch.log(var_ratio)).sum()


class BLIPLayer(nn.Module):
    """Same structure as ml/models/mpnn.py's MPNNLayer, with StochasticLinear
    in place of nn.Linear in both the message and update MLPs - the
    "stochasticity in message/update weights" CLAUDE.md specifies."""

    def __init__(self, hidden_channels: int, num_rbf: int):
        super().__init__()
        self.message_lin1 = StochasticLinear(hidden_channels + num_rbf, hidden_channels)
        self.message_lin2 = StochasticLinear(hidden_channels, hidden_channels)
        self.update_lin1 = StochasticLinear(2 * hidden_channels, hidden_channels)
        self.update_lin2 = StochasticLinear(hidden_channels, hidden_channels)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_rbf: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        m = F.silu(self.message_lin1(torch.cat([h[col], edge_rbf], dim=-1)))
        messages = self.message_lin2(m)

        aggregated = torch.zeros_like(h)
        aggregated.index_add_(0, row, messages)

        u = F.silu(self.update_lin1(torch.cat([h, aggregated], dim=-1)))
        return h + self.update_lin2(u)

    def kl_divergence(self) -> torch.Tensor:
        return (self.message_lin1.kl_divergence() + self.message_lin2.kl_divergence()
                + self.update_lin1.kl_divergence() + self.update_lin2.kl_divergence())


class BLIP(nn.Module):
    """Phase 2: Phase 1's MPNN backbone with BLIP-style stochastic weights.

    Exposes the same predict_energy / predict_energy_and_forces interface
    as every other phase/bake-off entry (one stochastic sample per call,
    so it's a drop-in for the existing deterministic train/eval loop in
    ml/training/train.py), plus predict_distribution for the actual
    Phase 2 deliverable: multiple MC forward passes producing a predictive
    mean and variance.

    `kl_weight` lives here rather than in `train:` because the six Phase
    1/2 configs' `train:` blocks are locked byte-identical
    (tests/ml/test_harness.py) and this coefficient only exists for BLIP -
    same reasoning as `num_mc_samples` already being a `model:` key. Zero
    by default (matches every other model having no KL term at all);
    ml/training/train.py reads it via `getattr(inner_model, "kl_weight",
    0.0)`, so a model that doesn't define it is simply never penalised.
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
        self.layers = nn.ModuleList([BLIPLayer(hidden_channels, num_rbf) for _ in range(num_layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

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

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

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
        """One stochastic sample's energy/forces. Same live-RBF-recompute
        pattern as mpnn.py/schnet.py - forces need a real gradient path
        from pos, not the precomputed edge_attr. The weight noise sampled
        inside predict_energy is a fixed random draw *within* this one
        forward pass, so it doesn't interfere with the pos->energy
        gradient at all - autograd just treats it as a constant."""
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
        """The actual Phase 2 deliverable: run `num_mc_samples` independent
        stochastic forward passes and return (E_mean, E_var, F_mean, F_var).

        @torch.no_grad() here because this is for inference/evaluation, not
        training - forces are obtained per-sample via a local
        enable_grad() block (same reasoning as ml/training/evaluate.py's
        grad-context fix: don't assume the caller's grad context, and don't
        leak a graph across num_mc_samples passes).
        """
        n = num_mc_samples or self.num_mc_samples
        energies, forces = [], []
        for _ in range(n):
            with torch.enable_grad():
                pos_sample = pos.detach().requires_grad_(True)
                e, f = self.predict_energy_and_forces(z, pos_sample, edge_index, edge_attr, batch=batch)
            energies.append(e.detach())
            forces.append(f.detach())

        e_stack = torch.stack(energies, dim=0)  # [n, B]
        f_stack = torch.stack(forces, dim=0)    # [n, N, 3]
        return e_stack.mean(0), e_stack.var(0, unbiased=True), f_stack.mean(0), f_stack.var(0, unbiased=True)

    def kl_divergence(self) -> torch.Tensor:
        """Total KL(q(W) || prior) across every stochastic layer. See the
        module docstring for how this should be combined into the
        training loss - not done automatically by this method."""
        return sum((layer.kl_divergence() for layer in self.layers), torch.tensor(0.0))
