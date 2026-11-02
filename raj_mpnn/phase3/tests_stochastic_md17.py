"""Sanity checks for stochastic_md17.py (Phase 1/2/3 on one PaiNN backbone).

Run:  python tests_stochastic_md17.py        (CPU, about a minute)
  or: pytest -q tests_stochastic_md17.py

These are the "physical/symmetry sanity checks" the scoping document asks
Phase 3 to pass (O3, risk R5), plus checks that the Phase 2 vs Phase 3
comparison really differs only in where the noise lives.
"""
import math
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stochastic_md17 as sm  # noqa: E402

torch.set_default_dtype(torch.float64)  # finite-difference checks need float64
CUTOFF = 5.0
SMALL = dict(F_=16, n_layers=2, n_rbf=8, cutoff=CUTOFF)


def ethanol_like(n_frames=3, seed=0):
    rng = np.random.default_rng(seed)
    z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1])
    base = rng.normal(0, 1.0, (9, 3))
    R = torch.tensor(base[None] + rng.normal(0, 0.05, (n_frames, 9, 3)))
    return z, R


def batch(z, R):
    B, N, _ = R.shape
    return sm.Batch(z, R, torch.zeros(B), torch.zeros(B, N, 3), CUTOFF, "cpu")


def random_rotation(seed=1):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return torch.tensor(q)


def all_models():
    models = {"painn": sm.BLIPPaiNN(**SMALL, bayesian=False),
              "blip": sm.BLIPPaiNN(**SMALL, bayesian=True, init_p=0.3)}
    for mode in ("node", "edge", "both"):
        models[f"graph-{mode}"] = sm.GraphStochasticPaiNN(**SMALL, stoch=mode, init_p=0.3)
    for m in models.values():
        m.eval()
    return models


def sample(model, b, seed):
    """One forward pass with a fixed noise draw (None alphas for PaiNN)."""
    al = model.alphas(b) if model.bayesian else None
    torch.manual_seed(seed)
    return model(b, alphas=al)


def test_rep_noise_moments():
    """x' = x(1 + sqrt(a) eps) has mean x and variance a x^2, per row."""
    torch.manual_seed(0)
    x = torch.tensor([[1.0, -2.0, 0.5]]).repeat(200_000, 1)
    a = torch.full((200_000, 1), 0.25)
    y = sm.rep_noise(x, a)
    assert torch.allclose(y.mean(0), x[0], atol=0.01)
    assert torch.allclose(y.var(0), 0.25 * x[0] ** 2, rtol=0.02)
    v = torch.randn(4, 3, 5)
    vy = sm.rep_noise(v, torch.full((4, 1), 0.5), shared_dims=(1,))
    ratio = vy / v  # one random factor per (row, channel), the same for x, y and z
    assert torch.allclose(ratio, ratio[:, :1, :].expand_as(ratio))


def test_noise_site_counts():
    m = all_models()
    assert m["graph-node"].n_alpha == {"node": 2, "edge": 0}
    assert m["graph-edge"].n_alpha == {"node": 0, "edge": 2}
    assert m["graph-both"].n_alpha == {"node": 2, "edge": 2}
    b = batch(*ethanol_like())
    assert set(m["graph-node"].alphas(b)) == {"node"}
    assert set(m["graph-edge"].alphas(b)) == {"edge"}
    assert set(m["graph-both"].alphas(b)) == {"node", "edge"}


def test_invariance_and_equivariance():
    """For one fixed noise draw: E(QR + t) = E(R) and F(QR + t) = F(R) Q^T."""
    z, R = ethanol_like()
    Q, t = random_rotation(), torch.tensor([0.7, -1.3, 2.1])
    Rt = R @ Q.T + t
    for name, model in all_models().items():
        E1, F1 = sample(model, batch(z, R), seed=123)
        E2, F2 = sample(model, batch(z, Rt), seed=123)
        F1 = F1.reshape(R.shape)
        F2 = F2.reshape(R.shape)
        assert torch.allclose(E1, E2, atol=1e-8), f"{name}: energy not invariant"
        assert torch.allclose(F1 @ Q.T, F2, atol=1e-8), f"{name}: forces not equivariant"


def test_permutation_invariance():
    """Relabelling atoms leaves the energy unchanged. Compared on the mean
    pass (alphas=None): a single noise draw is tied to atom order by design."""
    z, R = ethanol_like()
    perm = torch.tensor([1, 0, 2, 4, 3, 5, 8, 7, 6])
    for name, model in all_models().items():
        E1, _ = model(batch(z, R))
        E2, _ = model(batch(z[perm], R[:, perm]))
        assert torch.allclose(E1, E2, atol=1e-8), name


def test_forces_are_conservative():
    """F = -dE/dR matches a central finite difference of the SAME noise draw."""
    z, R = ethanol_like(n_frames=1)
    for name, model in all_models().items():
        _, Fp = sample(model, batch(z, R), seed=7)
        h, fd = 1e-5, torch.zeros(9, 3)
        for i in range(9):
            for k in range(3):
                Rp, Rm = R.clone(), R.clone()
                Rp[0, i, k] += h
                Rm[0, i, k] -= h
                Ep, _ = sample(model, batch(z, Rp), seed=7)
                Em, _ = sample(model, batch(z, Rm), seed=7)
                fd[i, k] = -(Ep - Em).item() / (2 * h)
        # BLIP and the graph model compute alpha from detached positions, so
        # autograd treats alpha as constant; the finite difference does not.
        # That gap is second order for smooth alpha and stays tiny here.
        assert torch.allclose(Fp.reshape(9, 3), fd, atol=1e-4, rtol=1e-3), \
            f"{name}: max |F - FD| = {(Fp.reshape(9, 3) - fd).abs().max():.2e}"


def test_noise_off_equals_deterministic_painn():
    """With alphas=None, Phase 3 is exactly Phase 1 PaiNN given the same weights.
    So any Phase 1 vs Phase 3 difference comes from the noise and its training,
    not from a different architecture."""
    torch.manual_seed(0)
    painn = sm.BLIPPaiNN(**SMALL, bayesian=False).eval()
    z, R = ethanol_like()
    E0, F0 = painn(batch(z, R))
    for mode in ("node", "edge", "both"):
        g = sm.GraphStochasticPaiNN(**SMALL, stoch=mode).eval()
        missing, unexpected = g.load_state_dict(painn.state_dict(), strict=False)
        assert not unexpected, unexpected
        assert all(k.startswith("posterior.") for k in missing), missing
        E1, F1 = g(batch(z, R))
        assert torch.allclose(E0, E1, atol=1e-10) and torch.allclose(F0, F1, atol=1e-10), mode


def test_same_backbone_parameter_count():
    """Phase 3 has exactly Phase 1's backbone weights plus a posterior network."""
    painn = sm.BLIPPaiNN(**SMALL, bayesian=False)
    n_painn = sum(p.numel() for p in painn.parameters())
    for mode in ("node", "edge", "both"):
        g = sm.GraphStochasticPaiNN(**SMALL, stoch=mode)
        n_backbone = sum(p.numel() for k, p in g.named_parameters() if not k.startswith("posterior."))
        assert n_backbone == n_painn, (mode, n_backbone, n_painn)


def test_mc_spread_and_posterior_gets_gradient():
    """Stochastic passes differ, and the data loss trains the posterior
    (alpha) through the noise - not only the KL term."""
    z, R = ethanol_like()
    b = batch(z, R)
    for mode in ("node", "edge", "both"):
        g = sm.GraphStochasticPaiNN(**SMALL, stoch=mode, init_p=0.3).train()
        al = g.alphas(b)
        Es = torch.stack([g(b, alphas=al)[0].detach() for _ in range(20)])
        assert Es.std(0).min() > 0, mode
        b2 = batch(z, R)
        al = g.alphas(b2)
        E, Fp = g(b2, alphas=al, create_graph=True)
        (E.pow(2).mean() + Fp.pow(2).mean()).backward()  # data-type loss, no KL
        grads = [p.grad.abs().sum().item() for k, p in g.named_parameters()
                 if k.startswith("posterior.") and p.grad is not None]
        assert grads and sum(grads) > 0, mode


def test_short_training_runs_end_to_end():
    """main() trains, validates, checkpoints and MC-tests every phase."""
    torch.set_default_dtype(torch.float32)
    try:
        with tempfile.TemporaryDirectory() as out:
            for extra in ("--model painn", "--model blip", "--model graph --stoch node",
                          "--model graph --stoch edge", "--model graph --stoch both"):
                # --out passed as its own list item: temp paths can contain spaces
                # (e.g. a Windows user folder), which a split string would break.
                res = sm.main(f"--synthetic --epochs 2 --features 16 --layers 2 --n-rbf 8 "
                              f"--batch-size 50 --mc-samples 5 --device cpu {extra}".split() + ["--out", out])
                vals = list(res["MAP"].values()) + list(res.get("MC", {}).values())
                assert all(math.isfinite(v) for v in vals), (extra, res)
                if "painn" not in extra:
                    assert "F_ECE" in res["MC"] and "F_spearman" in res["MC"]
            names = sorted(f for f in os.listdir(out) if f.endswith(".json"))
            assert names == sorted(["synthetic_blip_seed1.json", "synthetic_graph-both_seed1.json",
                                    "synthetic_graph-edge_seed1.json", "synthetic_graph-node_seed1.json",
                                    "synthetic_painn_seed1.json"]), names
    finally:
        torch.set_default_dtype(torch.float64)


def test_ece_measures_calibration():
    """ECE ~ 0 when the predicted std matches the real error spread, and large
    when it is 5x too small. The old structure-mean ECE gave ~0.25 for both."""
    rng = np.random.default_rng(0)
    std = rng.uniform(0.5, 2.0, 200_000)
    err = rng.normal(0, std)
    _, ece_good = sm._calibration(err, std)
    cov_bad, ece_bad = sm._calibration(err, std / 5)
    assert ece_good < 0.01, ece_good
    assert ece_bad > 0.4 and cov_bad[0.95] < 0.4, (ece_bad, cov_bad)


def test_rescore_checkpoints():
    """--eval-only rebuilds model and test split from a checkpoint: MAP metrics
    equal the training run's exactly. Also loads checkpoints written by the
    original Blip_model_building.ipynb code (no 'model' key in their args)."""
    import importlib.util
    torch.set_default_dtype(torch.float32)
    small = "--synthetic --epochs 2 --features 16 --layers 2 --n-rbf 8 --batch-size 50 --mc-samples 5 --device cpu"
    try:
        with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as out2:
            trained = sm.main(small.split() + ["--model", "graph", "--stoch", "edge", "--out", out])
            ck = os.path.join(out, "synthetic_graph-edge_seed1.pt")
            res = sm.main(["--eval-only", ck, "--out", out2])
            assert res["MAP"] == trained["MAP"]
            assert res["MC"]["metrics_version"] == sm.METRICS_VERSION
            assert "alpha_edge_rel_spread" in res["MC"] and "alpha_node_mean" not in res["MC"]
            assert os.path.exists(os.path.join(out2, "synthetic_graph-edge_seed1.json"))

            orig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blip_md17_original.py")
            try:
                import scipy.stats  # noqa: F401  (the original notebook code needs SciPy)
                have_scipy = True
            except ImportError:
                have_scipy = False
            if os.path.exists(orig_path) and have_scipy:  # original notebook code, if kept next to this file
                spec = importlib.util.spec_from_file_location("blip_orig", orig_path)
                orig = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(orig)
                orig.main(small.split() + ["--out", out])
                res = sm.main(["--eval-only", os.path.join(out, "synthetic_blip_seed1.pt"), "--out", out2])
                assert "F_coverage_95" in res["MC"] and "alpha_node_mean" in res["MC"]
    finally:
        torch.set_default_dtype(torch.float64)


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"\nall {len(tests)} checks passed")
