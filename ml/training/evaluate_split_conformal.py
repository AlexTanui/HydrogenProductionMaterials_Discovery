"""Evaluate frozen energy exports; provenance attestation is not proof of independence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from ml.utils.split_conformal import evaluate, absolute_residuals


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def indices(x, name, allow_empty=False):
    a = np.asarray(x)
    if a.ndim != 1 or (not a.size and not allow_empty):
        raise ValueError(f"{name}: expected one-dimensional IDs")
    if a.size and (a.dtype.kind not in "iu" or (a<0).any()):
        raise ValueError(f"{name}: expected nonnegative integer IDs")
    if len(np.unique(a)) != len(a):
        raise ValueError(f"{name}: duplicate IDs")
    return a.astype(np.int64)


def validate_manifest(m, cal, test):
    for field in ("dataset_sha256", "checkpoint_sha256", "calibration_sha256", "test_sha256"):
        v = m.get(field, "")
        if not isinstance(v,str) or len(v)!=64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError(f"Invalid {field}")
    if m.get("energy_unit")!="kcal/mol" or m.get("energy_kind")!="total_absolute":
        raise ValueError("Require total absolute energies in kcal/mol")
    for field in ("independence_evidence", "trajectory_limitations", "model_identity"):
        if not isinstance(m.get(field),str) or not m[field].strip():
            raise ValueError(f"Missing {field}")
    for field in ("calibration_unused_for_fit_selection_tuning", "test_unused_for_method_selection"):
        if m.get(field) is not True:
            raise ValueError(f"Unconfirmed {field}")
    used = indices(m["fit_selection_tuning_ids"], "used", True)
    reserved = indices(m["final_test_ids"], "final test")
    ci, ti = indices(cal["gold_index"], "calibration"), indices(test["gold_index"], "test")
    for name,a,b in (("cal/used",ci,used),("test/used",ti,used),("cal/final test",ci,reserved),("cal/test",ci,ti)):
        if np.intersect1d(a,b).size:
            raise ValueError(f"Overlap: {name}")
    if set(ti)!=set(reserved):
        raise ValueError("Test IDs must equal the preregistered final test IDs")
    for data in (cal,test):
        if len(absolute_residuals(data["E_ref"],data["E_pred"])) != len(data["gold_index"]):
            raise ValueError("Prediction/ID lengths differ")


def run(manifest_path, output):
    manifest_path, output = Path(manifest_path), Path(output)
    m = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    data = {}
    for role in ("calibration","test"):
        path = (manifest_path.parent/m[role]).resolve()
        if sha256(path)!=m[role+"_sha256"]:
            raise ValueError(f"{role}: export hash mismatch")
        with np.load(path,allow_pickle=False) as a:
            data[role] = {k:a[k] for k in ("gold_index","E_ref","E_pred")}
    validate_manifest(m,data["calibration"],data["test"])
    c,t = data["calibration"],data["test"]
    rows = evaluate(c["E_ref"],c["E_pred"],t["E_ref"],t["E_pred"])
    output.mkdir(parents=True,exist_ok=False)
    status = "ILLUSTRATIVE_SYNTHETIC_ONLY" if m.get("illustrative") is True else "OBSERVED_INPUT_PREDICTIONS"
    with (output/"comparison.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    clean=[{k:("Infinity" if isinstance(v,float) and np.isinf(v) else v) for k,v in r.items()} for r in rows]
    (output/"results.json").write_text(json.dumps(dict(status=status,metrics=clean,manifest=m,
        manifest_sha256=sha256(manifest_path),limitation="MD trajectory exchangeability is not established."),
        indent=2,allow_nan=False),encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,5))
    ax.plot([0,1],[0,1],"--",color="gray",label="Nominal")
    ax.plot([r["target_coverage"] for r in rows],[r["empirical_coverage"] for r in rows],"o-",label="Observed")
    ax.set(xlabel="Nominal coverage",ylabel="Empirical coverage",xlim=(0,1),ylim=(0,1),title=status)
    ax.legend(); fig.tight_layout(); fig.savefig(output/"coverage.png",dpi=160); plt.close(fig)
    order=np.argsort(t["gold_index"])
    selected=order[np.linspace(0,len(order)-1,min(150,len(order)),dtype=int)]
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,r in zip(axes,rows):
        y,p=t["E_ref"][selected],t["E_pred"][selected]
        if np.isfinite(r["quantile"]):
            ax.errorbar(y,p,yerr=r["quantile"],fmt=".",alpha=.45)
        else:
            ax.scatter(y,p,s=8); ax.text(.05,.9,"Unbounded intervals",transform=ax.transAxes)
        a,b=min(y.min(),p.min()),max(y.max(),p.max())
        ax.plot([a,b],[a,b],"--",color="gray")
        ax.set(xlabel="Reference energy (kcal/mol)",ylabel="Predicted energy (kcal/mol)",
               title=f"{r['target_coverage']:.0%} prediction intervals")
    fig.suptitle(status+" — at most 150 frames displayed")
    fig.tight_layout(); fig.savefig(output/"energy_intervals.png",dpi=160); plt.close(fig)
    return rows


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--output",required=True,help="New directory; existing outputs are never overwritten")
    args=p.parse_args()
    run(args.manifest,args.output)
