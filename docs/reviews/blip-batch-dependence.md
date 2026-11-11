# Finding: BLIP's noise scale couples the molecules in a batch

| | |
|---|---|
| **Scope** | `StochasticLinear.forward` in [`ml/models/blip.py`](../../ml/models/blip.py#L121-L128) (Phase 2, SCRUM-31) |
| **Owner** | Alex (BLIP). `blip.py` is **not** changed by this finding |
| **Measured at** | commit `832a2d6` on `fazin` |
| **Date** | 2026-10-16 |
| **Status** | Open. The batching physics check is marked `xfail(strict=True)` for BLIP until it is fixed |

## Summary

BLIP computes the size of its weight noise from the mean of **every row in
the batch**. A molecule's predicted energy and forces therefore depend on
which other molecules share its batch, and its forces pick up gradient
contributions from those other molecules' energies. At initialisation the
effect is small (3.0e-4 kcal/mol, 6.8e-5 kcal/mol/Å at batch size 32). It
grows steeply with the learned noise level: about 100× larger at 7× the
initial noise. A three-line change, computing the scale per row, removes it
completely. That change was tested by patching the method at runtime in a
scratch script; `blip.py` itself is unchanged.

## What `blip.py:123` does

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:                                       # line 121
    weight_sigma = F.softplus(self.weight_rho)
    input_scale = F.softplus(self.noise_gate(x.mean(dim=0, keepdim=True))).view(())      # line 123
    eps = torch.randn_like(self.weight_mu)
    weight = self.weight_mu + weight_sigma * input_scale * eps
    return F.linear(x, weight, self.bias)
```

`x` holds every row the layer sees in one call. In
[`BLIPLayer.forward`](../../ml/models/blip.py#L155-L164) that is every edge
of every molecule in the batch (`message_lin1`, `message_lin2`) or every
atom of every molecule in the batch (`update_lin1`, `update_lin2`).
`x.mean(dim=0)` therefore averages over the whole batch, and the resulting
scalar `input_scale` sets the noise on one weight matrix that every
molecule in the batch then uses. So molecule A's weights, and through them
its energy and forces, depend on molecules B, C, and so on. The module
docstring describes this as "the batch's pooled input", so the pooling is
intended. What it does to the per-molecule prediction appears not to have
been noticed.

Sharing `eps` (one draw per layer per forward call) is a separate matter
and is harmless. It makes the samples of molecules in the same pass
correlated, but it does not make one molecule's prediction depend on
another's *input*.

## Measured

**Setup.**
- Model: built from `experiments/configs/phase2_blip.yaml` (hidden 128, 4 layers, 50 RBF) with init seed 42.
- Units: wrapped in the harness's `ScaledEnergyModel` with the protocol's train-subset statistics, so every number is in kcal/mol as the model is trained and scored.
- Precision: float64.
- Frames: the first 32 frames of the protocol's train subset. No val or test data is used.
- RNG: reseeded before every forward call, so every call draws identical `eps`. Any difference below therefore comes from batch composition alone.

| Comparison | max \|ΔE\| (kcal/mol) | max \|ΔF\| (kcal/mol/Å) |
|---|---|---|
| Batch of 2 vs each frame alone | 2.2e-4 | 4.5e-5 |
| Batch of 32 (the protocol's batch size) vs each frame alone | 3.0e-4 | 6.8e-5 |
| Cross-molecule gradient max \|∂E₁/∂R₀\| in a batch of 2 (should be 0) | | 6.6e-4 |
| Control: MPNN (`phase1_baseline.yaml`), batch of 32 vs alone | **0** exactly | |

For scale, the largest force component on these untrained frames is
3.3e-2 kcal/mol/Å. The batch effect on forces is about 0.2% of that, and the
cross-molecule gradient about 2%.

**Sensitivity to the learned noise level** (batch of 32 vs alone, same
setup, every `weight_rho` set to a constant):

| `weight_rho` | σ = softplus(ρ) | max \|ΔE\| (kcal/mol) | max \|ΔF\| (kcal/mol/Å) |
|---|---|---|---|
| −5 (the initial value) | 0.0067 | 3.0e-4 | 6.8e-5 |
| −3 | 0.049 | 3.0e-2 | 1.1e-2 |

At ρ = −1 the untrained network diverges under its own weight noise, so that
point says nothing about batching and is omitted.

**Caveats.**
- Every number is at initialisation; no trained BLIP checkpoint exists yet. The size after training depends on the learned `weight_rho` and `noise_gate`, and nothing in the current loss pins either one, because the KL term is not wired in (see the `blip.py` module docstring).
- In float32 the energy difference reads 7.8e-3 kcal/mol. That is exactly one float32 step at 97,196 kcal/mol, so it is rounding, not a measurement. The float32 force differences (7.1e-5 and 7.2e-5) agree with float64.

## Why it matters

1. **Test results depend on batch size and frame order.** `evaluate.py`
   scores contiguous test frames in batches (default 32, `shuffle=False`), so
   BLIP's reported MAE changes with `--batch-size`. For every other model,
   `tests/ml/test_evaluate.py` requires metrics that are bit-identical across
   batch sizes.
2. **The trained forces are not the molecule's own forces.**
   `predict_energy_and_forces` differentiates the batch's summed energy, so
   the force on an atom includes ∂E_b/∂R_a from every other molecule b in
   the batch. The force loss trains BLIP against that mixture.
3. **Inference sees a different function from the one that was scored.** A
   backend or MD driver that predicts one molecule at a time gets different
   energies and forces from the batch-of-32 numbers in the benchmark table.
4. **It confounds the Phase 1 vs Phase 2 comparison.** MPNN has no such
   coupling (zero in the control above), so part of any MPNN-vs-BLIP
   difference could come from batch composition rather than from *where
   stochasticity lives*, which is the comparison the three-phase plan exists
   to make.
5. **It fails the batching physics check by 10⁴–10⁵× its tolerance.**
   `tests/ml/test_painn_physics.py` requires batched and single-frame
   predictions to agree to 1e-9 kcal/mol. BLIP misses by 1.6e-5 and 5.9e-5
   kcal/mol at batch sizes 2 and 4 (that test's own initialisation, seed 0).
   The check is `xfail(strict=True)`, so it reports XPASS, and fails, once
   the coupling is fixed.

## Suggested fix

**Option A (smallest; recommended): one noise scale per row.** Gate the
noise term with each row's own input instead of the batch mean:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    weight_sigma = F.softplus(self.weight_rho)
    eps = torch.randn_like(self.weight_mu)
    row_scale = F.softplus(self.noise_gate(x))              # (rows, 1): each row's own input
    return F.linear(x, self.weight_mu, self.bias) + row_scale * F.linear(x, weight_sigma * eps)
```

For each row r this computes `x_r (W_mu + σ · s_r · ε)ᵀ + b`, which is the
current formula with the pooled mean replaced by the row itself. It costs
one extra matrix multiply per layer. The batch-of-32 comparison above,
rerun with this method patched in at runtime, gives max |ΔE| = 0 and
max |ΔF| = 3e-17 (float64 roundoff). It does change the model's meaning,
however: the noise scale becomes per atom and per edge rather than per
molecule, even at batch size 1.

**Option B (keeps today's single-molecule behaviour): one scale per
molecule.** Average `x` over each molecule's own rows (a scatter-mean with
`batch` for atom-level layers and `batch[edge_index[1]]` for edge-level
layers), then broadcast each molecule's scale back to its rows. At batch
size 1 this is identical to the current model. It needs the batch index
threaded through `BLIPLayer.forward` into `StochasticLinear.forward`, so it
touches three methods instead of one.

Whether "input-dependent" was meant per molecule (B) or per input row (A)
is a modelling decision for the owner. Either option makes a molecule's
prediction independent of its batch-mates.

## Reproducing

The measurement scripts were run from the repo root. The core of the check:

```python
import torch
from torch_geometric.loader import DataLoader
from ml.config import ExperimentConfig
from ml.data.datasets import MD17Dataset
from ml.models.scaling import ScaledEnergyModel
from ml.training.train import build_model, select_subsets

cfg = ExperimentConfig.from_yaml("experiments/configs/phase2_blip.yaml")
ds = MD17Dataset("data/gold/md17/ethanol_dft.npz", split="train")
pos = select_subsets(cfg)["train"]["positions"]
shift, scale = ScaledEnergyModel.statistics_from_energies(ds.E[torch.as_tensor(pos)], 9)
frames = torch.utils.data.Subset(ds, pos[:32].tolist())
torch.manual_seed(42)
model = ScaledEnergyModel(build_model(cfg), shift, scale).double().eval()

def run(batch):
    torch.manual_seed(123)  # identical eps on every call
    e, f = model.predict_energy_and_forces(batch.x, batch.pos.double(), batch.edge_index, None, batch.batch)
    return e.detach(), f.detach()

alone = [run(b) for b in DataLoader(frames, batch_size=1)]
batched = [run(b) for b in DataLoader(frames, batch_size=32)]
print((torch.cat([b[0] for b in batched]) - torch.cat([a[0] for a in alone])).abs().max())  # 3.0e-4
```
