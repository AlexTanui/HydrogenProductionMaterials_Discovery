"""Reproduce SCRUM-36 descriptive comparisons from pinned repository records.

No retraining, new inference, p-values, or independent-sample assumptions.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = ["phase1_baseline", "phase2_blip", "phase2_blip_nll",
         "phase3_graph_stochastic", "phase3_graph_stochastic_nll"]
LABELS = ["MPNN", "BLIP MSE", "BLIP NLL", "Graph MSE", "Graph NLL"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    manifest = json.loads((ROOT/"source_manifest.json").read_text())
    for source in manifest["files"]:
        path = ROOT/source["local"]
        require(hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"],
                f"Source hash mismatch: {path}")
    rows, records, audit = [], {}, {"commit": manifest["commit"], "checks": []}
    train_ids = val_ids = fingerprint = train_config = data_config = None
    for name, label in zip(NAMES, LABELS):
        folder = ROOT/"sources"/name
        run = json.loads((folder/"run.json").read_text())
        test = json.loads((folder/"test_metrics.json").read_text())
        subsets = json.loads((folder/"subsets.json").read_text())
        tr, va = (np.asarray(subsets[s]["gold_indices"], dtype=np.int64) for s in ("train", "val"))
        require(not np.intersect1d(tr, va).size, f"{name}: train/validation overlap")
        for split, ids in (("train", tr), ("val", va)):
            require(len(np.unique(ids)) == len(ids), f"{name}: duplicate {split} IDs")
            require(hashlib.sha256(ids.tobytes()).hexdigest() ==
                    run["subsets"][split]["gold_indices_sha256"], f"{name}: {split} hash mismatch")
        if train_ids is None:
            train_ids, val_ids = tr, va
            fingerprint = test["data"]["fingerprint"]["sha256"]
            train_config, data_config = run["config"]["train"], run["config"]["data"]
        require(np.array_equal(tr, train_ids) and np.array_equal(va, val_ids),
                f"{name}: different training/validation frames")
        require(test["data"]["fingerprint"]["sha256"] == fingerprint, f"{name}: different test data")
        require(run["config"]["train"] == train_config and run["config"]["data"] == data_config,
                f"{name}: shared configuration mismatch")
        require(test["data"]["split"] == "test" and test["data"]["n_configs"] == 55509,
                f"{name}: unexpected test split")
        require(test["seed"] == 0 and test["timing"]["batch_size"] == 32,
                f"{name}: scoring protocol mismatch")
        require(test.get("num_mc_samples") == (None if name == NAMES[0] else 20),
                f"{name}: MC protocol mismatch")
        metric = test["metrics"]["per_key"][0]
        require(metric["units"]["energy"] == "kcal/mol" and metric["units"]["force"] == "kcal/mol/Ang",
                f"{name}: units differ")
        for key in ("energy_mae_total", "energy_rmse_total", "force_mae_component", "force_rmse_component"):
            require(metric[key] == test["metrics"]["overall"][key], f"{name}: aggregate mismatch")
        row = dict(run=name, label=label,
            energy_mae=metric["energy_mae_total"], energy_rmse=metric["energy_rmse_total"],
            force_mae=metric["force_mae_component"], force_rmse=metric["force_rmse_component"],
            energy_ece=None, force_ece=None, energy_spearman=None, force_spearman=None,
            energy_coverage90=None, force_coverage90=None,
            energy_mean_std=None, force_mean_std=None,
            parameters=run["n_parameters"], best_epoch=run["best_epoch"],
            training_seed=run["config"]["data"]["seed"], training_commit=run["git_commit"],
            git_dirty=run["git_dirty"], eval_mc=test.get("num_mc_samples"),
            train_mc=run["protocol"].get("train_num_mc_samples"),
            warmup_epochs=run["protocol"].get("nll_warmup_epochs"),
            train_seconds=run["wall_clock_s"],
            inference_ms=1000*test["timing"]["inference_s_per_config"])
        uq = test["uq_metrics"]
        if uq is not None:
            for quantity in ("energy", "force"):
                q = uq["per_key"][0][quantity]
                levels = uq["coverage_levels"]
                coverage = [q["coverage"][f"{100*x:g}%"] for x in levels]
                ece = float(np.mean(np.abs(np.asarray(coverage)-levels)))
                require(abs(ece-q["ece"])<1e-12, f"{name}: {quantity} ECE mismatch")
                row[quantity+"_ece"] = q["ece"]
                row[quantity+"_spearman"] = q["spearman_uncertainty_error"]
                row[quantity+"_coverage90"] = q["coverage"]["90%"]
                row[quantity+"_mean_std"] = q["sharpness_mean_std"]
        rows.append(row)
        records[name] = {"run": run, "test": test}
    deltas = []
    for row in rows[3:]:
        for base in rows[:3]:
            for metric in ("energy_mae", "energy_rmse", "force_mae", "force_rmse"):
                delta = row[metric]-base[metric]
                deltas.append(dict(phase3=row["label"], comparator=base["label"], metric=metric,
                    absolute_difference=delta, relative_difference_percent=100*delta/base[metric],
                    interpretation="higher error" if delta>0 else "lower error"))
    audit.update(test_fingerprint=fingerprint, n_test=55509, n_train=len(train_ids),
        n_validation=len(val_ids), training_seeds=sorted(set(r["training_seed"] for r in rows)),
        checks=["15 source hashes verified", "Exact training and validation ID arrays match",
                "Training and validation ID hashes match run records",
                "Shared data/train config blocks match", "Test fingerprints match across five runs",
                "Same energy/force definitions and units", "Same evaluation seed and batch size",
                "20 MC draws for every stochastic evaluation", "Eight ECE values independently recomputed"],
        limitations=["Source-reported test fingerprints; no checkpoint re-scoring performed",
                     "One training seed per configuration; no training-run uncertainty estimate",
                     "No aligned shared-model per-frame prediction exports in inspected tracked tree",
                     "Loss objectives, selection rules, model parameters and training code versions differ",
                     "Molecular trajectory frames and force components are not independent replicates"])
    out = ROOT/"results"
    out.mkdir(exist_ok=True)
    write_csv(out/"comparison.csv", rows)
    write_csv(out/"phase3_differences.csv", deltas)
    (out/"audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#475569", "#3b82f6", "#1d4ed8", "#14b8a6", "#0f766e"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, key, unit in zip(axes, ("energy_mae", "force_mae"), ("kcal/mol", "kcal/mol/Å")):
        bars=ax.bar(LABELS, [r[key] for r in rows], color=colors)
        ax.bar_label(bars, fmt="%.4f", padding=3)
        ax.set(ylabel=unit, title=key.replace("_", " ").title()+" (lower is better)")
        ax.set_ylim(0, max(r[key] for r in rows)*1.22)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Recorded ethanol DFT test results — one training seed per configuration")
    fig.tight_layout(); fig.savefig(out/"accuracy.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, quantity in zip(axes, ("energy", "force")):
        ax.plot([0,1],[0,1],"--",color="gray",label="Nominal")
        for name,label,color in zip(NAMES[1:],LABELS[1:],colors[1:]):
            uq=records[name]["test"]["uq_metrics"]
            cov=uq["per_key"][0][quantity]["coverage"]
            ax.plot(uq["coverage_levels"],[cov[f"{100*x:g}%"] for x in uq["coverage_levels"]],
                    "o-",label=label,color=color)
        ax.set(xlabel="Nominal central-interval coverage",ylabel="Observed coverage",
               title=quantity.title()+" — raw Gaussian intervals",xlim=(0,1),ylim=(0,1.02))
        ax.legend(fontsize=8)
    fig.suptitle("Calibration diagnostics from saved records; no post-hoc calibration applied")
    fig.tight_layout(); fig.savefig(out/"coverage.png",dpi=170); plt.close(fig)
    print(json.dumps({"rows":rows,"deltas":deltas,"audit":audit},indent=2))


if __name__ == "__main__":
    main()
