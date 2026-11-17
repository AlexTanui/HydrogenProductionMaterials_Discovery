"""Checks for phase3_showcase.py. Run:  python tests_showcase.py   (CPU, ~1-2 min)

Uses small stand-in data written in the team's gold .npz format, so the same
code paths run here as on the real ethanol / malonaldehyde gold files in Colab.
"""
import json
import math
import os
import sys
import tempfile

import numpy as np
import torch

# Folder holding this file - or, when the code is pasted into a notebook cell
# (no __file__ there), the current folder plus Colab's /content, where the
# notebooks save stochastic_md17.py and phase3_showcase.py.
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
for _p in (HERE, "/content"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import phase3_showcase as ps  # noqa: E402
import stochastic_md17 as sm  # noqa: E402

TMP = tempfile.mkdtemp(prefix="showcase_test_")
SMALL = "--features 32 --layers 2 --n-rbf 8 --batch-size 50 --mc-samples 5 --device cpu --prior-p 0.1"


def write_gold(name, seed, natoms):
    """Morse-potential toy molecule saved like data/gold/md17/<name>.npz."""
    z, R, E, F, _ = sm.make_synthetic(np.random.default_rng(seed), n=1300, natoms=natoms)
    z = np.where(z == 0, 1, z)
    path = os.path.join(TMP, "repo", "data", "gold", "md17", f"{name}.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, z=z, R=R, E=E, F=F, train_idx=np.arange(0, 1000), val_idx=np.arange(1000, 1150),
             test_idx=np.arange(1150, 1300), molecule=name.split("_")[0], theory="dft")
    return path


GOLD_A = write_gold("toyA_dft", 1, 9)
GOLD_B = write_gold("toyB_dft", 7, 12)  # a different "molecule" the model never sees
CKPT = os.path.join(TMP, "runs", "gold_toyA_dft_graph-both-env_seed1.pt")


def trained_checkpoint():
    if not os.path.exists(CKPT):
        sm.main(SMALL.split() + ["--repo-dir", os.path.join(TMP, "repo"), "--molecule", "toyA", "--theory", "dft",
                                 "--epochs", "12", "--model", "graph", "--stoch", "both", "--posterior", "env",
                                 "--out", os.path.join(TMP, "runs")])
    return CKPT


def test_acceleration_unit():
    """1 kcal/mol/A on 1 amu, in A/fs^2, from SI constants."""
    kcal_per_mol = 4184 / 6.02214076e23          # J
    a = kcal_per_mol / 1e-10 / 1.66053907e-27    # m/s^2
    assert math.isclose(a * 1e10 / 1e30, ps.ACC_UNIT, rel_tol=1e-3)


def test_load_frames_and_predict():
    z, R, E, F, idx = ps.load_frames(GOLD_A, "test", max_frames=40)
    assert len(idx) == 40 and np.all(np.diff(idx) > 0) and idx.min() >= 1150
    _, _, Eh, _, idx_h = ps.load_frames(GOLD_A, "test_high_energy", max_frames=0)
    d = np.load(GOLD_A)
    assert len(idx_h) == math.ceil(0.05 * 150) and Eh.min() >= np.quantile(d["E"][d["test_idx"]], 0.95) - 1e-9
    model, _ = ps.load_checkpoint(trained_checkpoint())
    p = ps.predict(model, z, R, mc=6)
    assert p["F_std"].shape == R.shape and p["atom_unc"].shape == R.shape[:2] and p["E_std"].min() > 0
    assert p["alpha_node"].shape == R.shape[:2] and p["alpha_node"].std() > 0  # env posterior varies


def test_ood_study():
    out = os.path.join(TMP, "ood")
    res = ps.ood_study(trained_checkpoint(), {"toy A (test)": (GOLD_A, "test"),
                                              "toy A (top-5% energy)": (GOLD_A, "test_high_energy"),
                                              "toy B (unseen)": (GOLD_B, "all")},
                       mc=5, max_frames=60, out=out)
    assert set(res) == {"toy A (test)", "toy A (top-5% energy)", "toy B (unseen)"}
    for r in res.values():
        assert all(math.isfinite(r[k]) for k in ("F_MAE_meV/A", "F_std_mean_meV/A", "F_ECE", "frame_spearman"))
    assert res["toy A (test)"]["error_ratio_vs_ref"] == 1.0
    # an unseen molecule is much harder - the model's error must reflect that
    assert res["toy B (unseen)"]["F_MAE_meV/A"] > res["toy A (test)"]["F_MAE_meV/A"]
    tag = "gold_toyA_dft_graph-both-env_seed1"
    js = json.load(open(os.path.join(out, f"ood_{tag}.json")))
    assert "frame_rms_err" not in js["results"]["toy A (test)"]
    assert os.path.getsize(os.path.join(out, f"ood_{tag}.png")) > 10_000


def test_active_learning():
    out = os.path.join(TMP, "al")
    base = SMALL + f" --repo-dir {os.path.join(TMP, 'repo')} --molecule toyA --theory dft --model graph --stoch node"
    curves = ps.active_learning(base.split(), init=40, step=40, rounds=2, epochs=3, pool_size=600,
                                val_size=100, candidates=200, mc=4, out=out)
    for strat in ("uncertainty", "random"):
        assert [p["n_train"] for p in curves[strat]] == [40, 80, 120]
        assert all(math.isfinite(p["val_F_MAE_meV/A"]) for p in curves[strat])
    # same starting frames and seeds: round 0 is identical for both strategies
    assert curves["uncertainty"][0] ["val_F_MAE_meV/A"] == curves["random"][0]["val_F_MAE_meV/A"]
    assert os.path.exists(os.path.join(out, "al_graph-node.png"))


def test_uncertainty_pick_is_top_k():
    """The 'uncertainty' strategy adds exactly the highest-variance candidates."""
    model, _ = ps.load_checkpoint(trained_checkpoint())
    z, R, _, _, idx = ps.load_frames(GOLD_A, "train", max_frames=50)
    pred = ps.predict(model, z, R, 4, seed=3)
    score = (pred["F_std"] ** 2).mean((1, 2))
    top = idx[np.argsort(-score)[:5]]
    assert set(top) == set(idx[score >= np.sort(score)[-5]])


def test_md_energy_conservation_and_breakdown():
    """No thermostat (huge tau, flat temperature): total energy is conserved by
    velocity Verlet with the model's forces - checks force/energy/unit consistency.
    Then a hot ramp must be detected as a breakdown with a reason."""
    model_ck = trained_checkpoint()
    z, R, _, _, _ = ps.load_frames(GOLD_A, "test", max_frames=1)
    res = ps.run_md(model_ck, (z, R[0]), steps=400, dt=0.1, t_start=50, t_end=50, tau=1e12,
                    mc=3, mc_every=10, out=os.path.join(TMP, "md_nve"), break_stretch=10)
    tot = np.array([r["E_pot"] + r["E_kin"] for r in res["log"]])
    kin = np.array([r["E_kin"] for r in res["log"]])
    assert np.abs(tot - tot[0]).max() < 0.02 * kin.mean() + 1e-3, (np.abs(tot - tot[0]).max(), kin.mean())
    assert res["summary"]["t_break_fs"] is None and len(res["frames"]) == 41

    hot = ps.run_md(model_ck, (z, R[0]), steps=300, dt=0.5, t_start=300, t_end=300, tau=10,
                    mc=3, mc_every=5, out=os.path.join(TMP, "md_hot"), break_stretch=1.05)
    s = hot["summary"]
    assert s["t_break_fs"] is not None and s["break_reason"].startswith("bond stretched")
    assert hot["log"][-1]["max_bond_stretch"] > 1.05
    assert os.path.exists(os.path.join(TMP, "md_hot", "md_gold_toyA_dft_graph-both-env_seed1.png"))


def test_viewer_html():
    model_ck = trained_checkpoint()
    z, R, _, _, _ = ps.load_frames(GOLD_A, "test", max_frames=1)
    res = ps.run_md(model_ck, (z, R[0]), steps=200, dt=0.5, t_start=300, t_end=1500, mc=3, mc_every=5,
                    save_every=10, out=os.path.join(TMP, "md_view"))
    path = ps.write_md_viewer(res, os.path.join(TMP, "md_view", "viewer.html"))
    html = open(path, encoding="utf-8").read()
    assert ps.THREEDMOL in html and "__DATA__" not in html and "__TITLE__" not in html
    data = json.loads(html.split("const D = ", 1)[1].split(";\n", 1)[0])
    assert len(data["frames"]) == len(res["frames"]) and data["z"] == z.tolist()


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"\nall {len(tests)} showcase checks passed  (scratch files in {TMP})")
