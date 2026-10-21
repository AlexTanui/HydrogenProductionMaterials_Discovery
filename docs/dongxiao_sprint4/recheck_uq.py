"""Recheck saved DimeNet draws and committed BLIP summary; no training.
Usage: python recheck_uq.py --team PATH --legacy PATH --out PATH
Requires numpy, scipy, torch and the existing team checkout.
"""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--team', type=Path, required=True)
    p.add_argument('--legacy', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.team.resolve()))
    from ml.utils.uq_metrics import _ece
    source = a.legacy / 'ml/training/dongxiao_edge_experiment.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('metrics', 'uq_metrics')]
    ns = dict(np=np, spearmanr=spearmanr)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), ns)
    results = []
    for seed in (42, 43, 44):
        folder = a.legacy / 'experiments/results' / ('dongxiao_edge_noise_20' + ('' if seed == 42 else f'_seed{seed}'))
        target = np.load(folder / 'validation_targets.npz')
        indices = target['indices']
        assert len(np.unique(indices)) == len(indices)
        arms = {}
        for arm in ('deterministic', 'edge_gaussian'):
            file = folder / f'{arm}_validation_samples.npz'
            pred = np.load(file)
            e, f = pred['energy'], pred['force']
            assert e.shape[1:] == target['energy'].shape
            assert f.shape[1:] == target['force'].shape
            assert np.isfinite(e).all() and np.isfinite(f).all()
            settings = dict(random_seed=42, random_repetitions=100, reject_fractions=[0,.1,.2,.3], coverage_levels=[.5,.8,.9])
            q = ns['uq_metrics'](e.astype(float), f.astype(float), target['energy'].astype(float), target['force'].astype(float), settings)
            q['mc_samples'] = len(e)
            q['max_energy_mc_std'] = float(e.std(0, ddof=1).max()) if len(e)>1 else None
            q['prediction_sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
            arms[arm] = q
        results.append(dict(seed=seed, n_frames=len(indices), arms=arms,
            force_mae_increase_percent=100*(arms['edge_gaussian']['force_mae']/arms['deterministic']['force_mae']-1),
            alignment='Shapes/order-producing code checked; prediction archive has no frame IDs, so independent ID matching is unavailable.'))
    committed = json.loads((a.team/'experiments/runs/phase2_blip/test_metrics.json').read_text())
    uq = committed['uq_metrics']['per_key'][0]
    for quantity in ('energy','force'):
        assert np.isclose(_ece(uq[quantity]['coverage']), uq[quantity]['ece'])
    report = dict(legacy_recomputed=results, phase2_blip_committed=uq,
        phase2_blip_nll=dict(status='not present in fetched run records; no checkpoint/raw draws available; not evaluated'),
        checks='Finite values, saved-array shapes, target IDs unique; shared ECE formula agrees with committed summary. No retraining.',
        mean_paired_force_increase_percent=float(np.mean([r['force_mae_increase_percent'] for r in results])))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(mean_force_increase_percent=report['mean_paired_force_increase_percent'],
        force_spearman=[r['arms']['edge_gaussian']['force_spearman'] for r in results]),indent=2))


if __name__ == '__main__':
    main()
