"""Tests for ml/training/mc_stability.py (SCRUM-49). No trained checkpoint or
gold data needed: an untrained stochastic model on small synthetic molecules
is enough to check the sampling, and the criterion is pure arithmetic."""
from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

from ml.data.md17 import build_graph
from ml.models.registry import build_model
from ml.models.scaling import ScaledEnergyModel
from ml.training.mc_stability import prefix_distributions, stable_k


def _molecules(n: int = 3, n_atoms: int = 5) -> Batch:
    generator = torch.Generator().manual_seed(0)
    graphs = []
    for _ in range(n):
        z = torch.tensor([6, 6, 8, 1, 1])[:n_atoms]
        pos = torch.rand(n_atoms, 3, generator=generator) * 3.0
        edge_index, edge_attr = build_graph(pos.numpy(), cutoff=5.0, num_rbf=16)
        graphs.append(Data(x=z.view(-1, 1), pos=pos, edge_index=edge_index, edge_attr=edge_attr,
                           y=torch.zeros(1), force=torch.zeros(n_atoms, 3)))
    return Batch.from_data_list(graphs)


@pytest.mark.parametrize("phase", ["blip", "graph_stochastic"])
def test_each_prefix_is_what_predict_distribution_returns(phase):
    torch.manual_seed(0)
    model = ScaledEnergyModel(build_model(phase, hidden_channels=16, num_layers=2, num_rbf=8),
                              shift=-1000.0, scale=0.5).eval()
    batch = _molecules()
    args = (batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch)
    torch.manual_seed(7)
    prefixes = prefix_distributions(model, batch, (2, 5))
    for k in (2, 5):
        torch.manual_seed(7)
        expected = model.predict_distribution(*args, num_mc_samples=k)
        for got, want, name in zip(prefixes[k], expected, ("E_mean", "E_var", "F_mean", "F_var")):
            torch.testing.assert_close(got, want, rtol=0, atol=0, msg=f"K={k} {name}")


def _results(values: dict[int, list[float]], spearman=None) -> dict:
    """{seed: {k: {...}}} with every judged metric equal to values[k][seed]."""
    seeds = range(len(next(iter(values.values()))))
    return {
        s: {k: {q: {"ece": v[s], "coverage_90": v[s],
                    "spearman": v[s] if spearman is None else spearman,
                    "sharpness": 1.0, "nll_gaussian": 1.0}
                for q in ("energy", "force")}
            for k, v in values.items()}
        for s in seeds
    }


def test_stable_k_is_the_smallest_k_from_which_everything_stays_close():
    results = _results({2: [0.30, 0.32, 0.28], 5: [0.215, 0.212, 0.209], 10: [0.205, 0.206, 0.204],
                        20: [0.201, 0.2, 0.202]})
    verdict = stable_k(results, tolerance=0.01)
    assert verdict["reference_k"] == 20
    assert [row["stable"] for row in verdict["rows"]] == [False, False, True, True]
    assert verdict["stable_k"] == 10


def test_stable_k_needs_every_larger_k_to_be_stable_too():
    # K=5 happens to land on the reference, but K=10 doesn't: 5 is not "stable from here on".
    results = _results({2: [0.5, 0.5, 0.5], 5: [0.2, 0.2, 0.2], 10: [0.25, 0.25, 0.25], 20: [0.2, 0.2, 0.2]})
    assert stable_k(results, tolerance=0.01)["stable_k"] == 20


def test_seed_spread_alone_can_make_a_k_unstable():
    results = _results({5: [0.17, 0.2, 0.23], 20: [0.2, 0.2, 0.2]})  # mean on target, spread 0.024
    verdict = stable_k(results, tolerance=0.01)
    assert verdict["rows"][0]["max_gap_to_reference"] == pytest.approx(0.0)
    assert verdict["rows"][0]["stable"] is False and verdict["stable_k"] == 20


def test_an_undefined_metric_is_never_called_stable():
    results = _results({5: [0.2, 0.2, 0.2], 20: [0.2, 0.2, 0.2]}, spearman=None)
    for seed in results:
        for k in results[seed]:
            results[seed][k]["energy"]["spearman"] = None
    verdict = stable_k(results)
    assert verdict["stable_k"] is None
    assert "energy.spearman" in verdict["rows"][0]["undefined"]
