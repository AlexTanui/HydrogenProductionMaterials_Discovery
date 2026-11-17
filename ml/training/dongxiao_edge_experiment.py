"""Matched fine-tuning and validation-only UQ for the independent DimeNet adapter."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import torch
import yaml
from torch_geometric.loader import DataLoader

from ml.models.dimenet import energy_forces
from ml.models.dongxiao_edge_noise import EdgeNoiseDimeNet
from ml.training.train_dimenet import ROOT, load_gold, epoch


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def predict(model, dataset, batch_size, samples, seed):
    """One energy and its coordinate gradient per draw; no no_grad for forces."""
    model.eval()
    energies, forces, elapsed = [], [], []
    start = time.perf_counter()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for _ in range(samples):
            es, fs = [], []
            for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
                e, f = energy_forces(model, batch)
                es.append(e.detach().cpu().numpy())
                fs.append(f.detach().cpu().numpy().reshape(batch.num_graphs, -1, 3))
            energies.append(np.concatenate(es)); forces.append(np.concatenate(fs))
            elapsed.append(time.perf_counter()-start)
    e = np.stack(energies).astype(np.float64)
    f = np.stack(forces).astype(np.float64)
    if not np.isfinite(e).all() or not np.isfinite(f).all():
        raise RuntimeError('Non-finite Monte Carlo predictions')
    return e, f, elapsed


def metrics(e, f, target_e, target_f):
    de, df = e.mean(0)-target_e, f.mean(0)-target_f
    return dict(energy_mae=float(np.abs(de).mean()), force_mae=float(np.abs(df).mean()),
                energy_mse=float(np.square(de).mean()), force_mse=float(np.square(df).mean()))


def uq_metrics(e, f, target_e, target_f, settings):
    """Equal-weight structure scores; atom components are not independent samples."""
    result = metrics(e, f, target_e, target_f)
    if len(e) < 2:
        return dict(result, uncertainty_available=False)
    eu, fu = e.var(0, ddof=1), f.var(0, ddof=1).mean(axis=(1, 2))
    ee = np.square(e.mean(0)-target_e)
    fe = np.square(f.mean(0)-target_f).mean(axis=(1, 2))
    def rho(a, b):
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            return None
        return float(spearmanr(a, b).statistic)
    result.update(uncertainty_available=True, energy_spearman=rho(eu, ee), force_spearman=rho(fu, fe))
    random = np.random.default_rng(settings['random_seed'])
    per_structure_mae = np.abs(f.mean(0)-target_f).mean(axis=(1, 2))
    order = np.argsort(fu, kind='stable')
    rejection = []
    for fraction in settings['reject_fractions']:
        n = max(1, int(len(order)*(1-fraction)))
        random_scores = [float(per_structure_mae[random.permutation(len(order))[:n]].mean())
                         for _ in range(settings['random_repetitions'])]
        rejection.append(dict(reject_fraction=fraction, retained_count=n,
                              force_mae=float(per_structure_mae[order[:n]].mean()),
                              random_mean=float(np.mean(random_scores)),
                              random_p05=float(np.quantile(random_scores, .05)),
                              random_p95=float(np.quantile(random_scores, .95))))
    result['rejection'] = rejection
    coverage = []
    for level in settings['coverage_levels']:
        lower, upper = np.quantile(e, [(1-level)/2, (1+level)/2], axis=0)
        coverage.append(dict(nominal=level, observed=float(((target_e >= lower)&(target_e <= upper)).mean()),
                             mean_width=float((upper-lower).mean())))
    result['raw_energy_sample_interval_coverage'] = coverage
    result['raw_interval_coverage_gap'] = float(np.mean([abs(x['nominal']-x['observed']) for x in coverage]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    if (ROOT/'.git/HEAD').read_text().strip() != 'ref: refs/heads/dongxiao':
        raise RuntimeError('This independent experiment requires branch dongxiao')
    cfg_path = (ROOT/args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    tc, uc = cfg['train'], cfg['uncertainty']
    output = (ROOT/tc['output_dir']).resolve()
    if not output.is_relative_to((ROOT/'experiments/results').resolve()):
        raise ValueError('Output must be under experiments/results')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(tc['num_threads'])
    torch.use_deterministic_algorithms(True)
    datasets, mean, indices, gold_hash = load_gold(cfg)
    checkpoint_path = ROOT/cfg['initial_checkpoint']
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if abs(checkpoint['energy_mean']-mean) > 1e-8:
        raise ValueError('Initial checkpoint must use exactly the same train-only energy center')
    if checkpoint['config']['model'] != cfg['model']:
        raise ValueError('Initial checkpoint architecture differs from experiment config')
    target_e = datasets['val'].E.numpy().astype(np.float64).reshape(-1)
    target_f = datasets['val'].F.numpy().astype(np.float64)
    mc = dict(cfg['model']); mc.pop('name')
    report = dict(status='running', config=cfg, gold_sha256=gold_hash,
                  initial_checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  config_sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
                  selected_gold_indices=indices, energy_mean=mean,
                  torch_version=str(torch.__version__), test_evaluated=False,
                  units={'energy':'kcal/mol', 'force':'kcal/mol/Angstrom'}, arms={})
    report['source_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [Path(__file__), ROOT/'ml/models/dongxiao_edge_noise.py',
                                        ROOT/'ml/models/dimenet.py', ROOT/'ml/training/train_dimenet.py']}
    np.savez_compressed(output/'validation_targets.npz', energy=target_e, force=target_f, indices=indices['val'])
    reference = EdgeNoiseDimeNet(**mc, noise_std=0, cutoff=cfg['data']['cutoff_radius'])
    reference.load_state_dict(checkpoint['model_state'], strict=True)
    e, f, _ = predict(reference, datasets['val'], tc['batch_size'], 1, uc['selection_seed'])
    report['initial_validation'] = metrics(e, f, target_e, target_f)
    save_json(output/'metrics.json', report)
    try:
        for arm in cfg['arms']:
            torch.manual_seed(cfg['data']['seed'])
            model = EdgeNoiseDimeNet(**mc, noise_std=arm['noise_std'], cutoff=cfg['data']['cutoff_radius'])
            model.load_state_dict(checkpoint['model_state'], strict=True)
            # Both arms start new Adam states and receive the same shuffled batches.
            optimizer = torch.optim.Adam(model.parameters(), lr=tc['lr'])
            generator = torch.Generator().manual_seed(cfg['data']['seed'])
            loader = DataLoader(datasets['train'], batch_size=tc['batch_size'], shuffle=True, generator=generator)
            record = dict(noise_std=arm['noise_std'], history=[], best_score=None)
            report['arms'][arm['name']] = record
            n_mc = uc['selection_samples'] if arm['noise_std'] else 1
            for number in range(1, tc['epochs']+1):
                start = time.perf_counter()
                tr = epoch(model, loader, tc, optimizer)
                e, f, _ = predict(model, datasets['val'], tc['batch_size'], n_mc, uc['selection_seed'])
                val = metrics(e, f, target_e, target_f)
                score = tc['energy_loss_weight']*val['energy_mse']+tc['force_loss_weight']*val['force_mse']
                row = dict(epoch=number, train=tr, validation=val, selection_score=score, seconds=time.perf_counter()-start)
                record['history'].append(row)
                if record['best_score'] is None or score < record['best_score']:
                    record.update(best_score=score, best_epoch=number)
                    torch.save(dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                                    epoch=number, energy_mean=mean, config=cfg, arm=arm,
                                    torch_rng_state=torch.get_rng_state(), loader_rng_state=generator.get_state()),
                               output/(arm['name']+'_best.pt'))
                save_json(output/'metrics.json', report)
                print(json.dumps(dict(arm=arm['name'], **row)), flush=True)
            selected = torch.load(output/(arm['name']+'_best.pt'), map_location='cpu', weights_only=False)
            model.load_state_dict(selected['model_state'], strict=True)
            n_mc = max(uc['sample_counts']) if arm['noise_std'] else 1
            e, f, elapsed = predict(model, datasets['val'], tc['batch_size'], n_mc, uc['evaluation_seed'])
            np.savez_compressed(output/(arm['name']+'_validation_samples.npz'), energy=e, force=f, elapsed=elapsed)
            record['evaluation'] = uq_metrics(e, f, target_e, target_f, uc)
            counts = uc['sample_counts'] if arm['noise_std'] else [1]
            record['mc_prefix'] = [dict(samples=n, seconds=elapsed[n-1],
                                        energy_mean_shift_to_largest=float(np.abs(e[:n].mean(0)-e.mean(0)).mean()),
                                        force_mean_shift_to_largest=float(np.abs(f[:n].mean(0)-f.mean(0)).mean()),
                                        **uq_metrics(e[:n], f[:n], target_e, target_f, uc)) for n in counts]
            save_json(output/'metrics.json', report)
        if hashlib.sha256((ROOT/cfg['data']['gold_path']).read_bytes()).hexdigest() != gold_hash:
            raise RuntimeError('Gold data changed during experiment')
        report['status'] = 'completed'
        save_json(output/'metrics.json', report)
    except BaseException as exc:
        report['status'] = 'failed'; report['error'] = repr(exc)
        save_json(output/'metrics.json', report)
        raise
    print('Completed validation-only experiment: '+str(output), flush=True)


if __name__ == '__main__':
    main()
