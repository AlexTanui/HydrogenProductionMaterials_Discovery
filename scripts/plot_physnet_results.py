"""Figures for the PhysNet bake-off entry, from a trained checkpoint.

Produces the two things a reader of an interatomic-potential result
actually wants to see, which a table of MAEs cannot show:

  1. Parity plots - predicted against reference, for energy and for force
     components. A single MAE hides *where* the model is wrong; parity
     shows whether error is uniform, or concentrated in high-force
     configurations, or a systematic slope (a scale error, not a noise
     error).
  2. Error distributions - whether error is symmetric noise or a
     one-sided bias. A model that is always 0.5 kcal/mol low has the same
     MAE as one scattered +/-0.5, and they are different problems.

    python -m scripts.plot_physnet_results --config experiments/configs/phase1_physnet.yaml

Writes PNGs into reports/figures/, which docs/technical_report.md can
reference. Needs only the saved checkpoint - no retraining.

The plotting functions take plain numpy arrays and import no torch, so
they can be tested without a GPU, a checkpoint, or a data download.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Palette shared with notebooks/eda_md17.ipynb so every figure in the
# project reads as one set. Validated on the light surface: passes the
# lightness band, chroma floor, CVD separation and normal-vision floor.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
RULE = "#8a5c00"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRIDLINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "grid.color": GRIDLINE,
    "font.size": 9,
    "axes.grid": True,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_parity(ax, ref, pred, label, unit, color=BLUE, max_points=8000, seed=0):
    """Predicted against reference, with the ideal y = x line.

    Points are thinned to `max_points`: a scatter of 55,000 overplots into
    a solid block that hides exactly the outlier structure the plot exists
    to reveal.
    """
    ref = np.asarray(ref).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    if ref.size > max_points:
        idx = np.random.default_rng(seed).choice(ref.size, max_points, replace=False)
        ref, pred = ref[idx], pred[idx]

    lo = float(min(ref.min(), pred.min()))
    hi = float(max(ref.max(), pred.max()))
    pad = 0.03 * (hi - lo) if hi > lo else 1.0

    # Ideal line first, so data sits on top of it.
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=RULE,
            linestyle="--", linewidth=1.2, zorder=1)
    ax.scatter(ref, pred, s=3, color=color, alpha=0.35, linewidths=0, zorder=2)

    mae = float(np.abs(pred - ref).mean())
    rmse = float(np.sqrt(((pred - ref) ** 2).mean()))
    r2 = 1.0 - ((pred - ref) ** 2).sum() / max(((ref - ref.mean()) ** 2).sum(), 1e-30)

    ax.set_xlabel(f"reference {label} ({unit})", fontsize=8)
    ax.set_ylabel(f"predicted {label} ({unit})", fontsize=8)
    ax.set_title(f"{label.capitalize()} parity — MAE {mae:.3f} {unit}", fontsize=9.5)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)

    # Numbers as text, not a legend: one series needs no colour key, and
    # the reader wants the values, not the name of the colour.
    ax.text(0.04, 0.96, f"MAE  {mae:.3f}\nRMSE {rmse:.3f}\nR²   {r2:.4f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
            color=INK_SECONDARY, family="monospace")
    return mae, rmse, r2


def plot_error_hist(ax, ref, pred, label, unit, color=BLUE):
    """Signed error. Centred on zero means noise; shifted means bias."""
    err = (np.asarray(pred).reshape(-1) - np.asarray(ref).reshape(-1))
    ax.hist(err, bins=60, color=color, edgecolor=SURFACE, linewidth=0.3)
    ax.axvline(0.0, color=RULE, linestyle="--", linewidth=1.2)
    ax.set_xlabel(f"predicted − reference {label} ({unit})", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.set_title(f"{label.capitalize()} error — mean {err.mean():+.3f}, "
                 f"std {err.std():.3f} {unit}", fontsize=9.5)
    ax.tick_params(labelsize=7)
    return float(err.mean()), float(err.std())


def make_figures(e_ref, e_pred, f_ref, f_pred, out_dir, n_configs=None):
    """Writes the three figures. Pure numpy in, file paths out."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5))
    plot_parity(axes[0], e_ref, e_pred, "energy", "kcal/mol", color=BLUE)
    plot_parity(axes[1], f_ref, f_pred, "force", "kcal/mol/Å", color=ORANGE)
    suffix = f" — {n_configs:,} validation configs" if n_configs else ""
    fig.suptitle(f"PhysNet: predicted vs reference{suffix}", color=INK_PRIMARY)
    fig.tight_layout()
    p = out_dir / "physnet_parity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    plot_error_hist(axes[0], e_ref, e_pred, "energy", "kcal/mol", color=BLUE)
    plot_error_hist(axes[1], f_ref, f_pred, "force", "kcal/mol/Å", color=ORANGE)
    fig.suptitle("PhysNet: signed error distributions", color=INK_PRIMARY)
    fig.tight_layout()
    p = out_dir / "physnet_error_distribution.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # Error against reference force magnitude: does accuracy degrade where
    # the forces are large? That is the regime an MD simulation spends its
    # time in, so a model that is only accurate near equilibrium is a
    # model that will drift.
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    f_ref_a = np.asarray(f_ref).reshape(-1)
    f_err = np.abs(np.asarray(f_pred).reshape(-1) - f_ref_a)
    if f_ref_a.size > 8000:
        idx = np.random.default_rng(1).choice(f_ref_a.size, 8000, replace=False)
        f_ref_a, f_err = f_ref_a[idx], f_err[idx]
    ax.scatter(np.abs(f_ref_a), f_err, s=3, color=ORANGE, alpha=0.3, linewidths=0)
    ax.set_xlabel("|reference force component| (kcal/mol/Å)", fontsize=8)
    ax.set_ylabel("absolute error (kcal/mol/Å)", fontsize=8)
    ax.set_title("Force error vs force magnitude", fontsize=9.5)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    p = out_dir / "physnet_force_error_vs_magnitude.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--limit-val", type=int, default=2000)
    args = p.parse_args(argv)

    import torch

    from ml.models.physnet import PhysNet
    from ml.training.train_physnet import ROOT, batch_forward, load_config, make_loaders

    cfg = load_config(args.config)
    device = torch.device("cpu")
    _train, val_loader = make_loaders(cfg, limit_train=1, limit_val=args.limit_val)

    ckpt_path = ROOT / "experiments" / "checkpoints" / f"{cfg['name']}.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PhysNet(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    e_ref, e_pred, f_ref, f_pred = [], [], [], []
    for data in val_loader:
        energy, forces, _z, _b = batch_forward(model, data, device)
        e_pred.append(energy.detach().cpu().numpy())
        e_ref.append(data.y.reshape(-1).cpu().numpy())
        f_pred.append(forces.detach().cpu().numpy().reshape(-1))
        f_ref.append(data.force.cpu().numpy().reshape(-1))

    e_ref, e_pred = np.concatenate(e_ref), np.concatenate(e_pred)
    f_ref, f_pred = np.concatenate(f_ref), np.concatenate(f_pred)

    written = make_figures(e_ref, e_pred, f_ref, f_pred,
                           ROOT / "reports" / "figures", n_configs=e_ref.size)
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
