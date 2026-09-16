"""Independent Gold-only DimeNet++ smoke training; shared modules unchanged."""
import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch_geometric
import yaml
from torch_geometric.loader import DataLoader

from ml.data.datasets import MD17Dataset
from ml.models.dimenet import GoldDimeNetPlusPlus, energy_forces

ROOT = Path(__file__).resolve().parents[2]


def load_gold(config):
    dc = config['data']
    path = (ROOT / dc['gold_path']).resolve()
    if not path.is_relative_to((ROOT / 'data/gold/md17').resolve()):
        raise ValueError('DimeNet training requires a file under data/gold/md17')
    with np.load(path, allow_pickle=False) as d:
        if str(d['molecule']) != dc['molecule'] or str(d['theory']) != dc['theory']:
            raise ValueError('Gold molecule/theory does not match config')
        splits = [d[f'{s}_idx'] for s in ('train', 'val', 'test')]
        if not np.array_equal(np.sort(np.concatenate(splits)), np.arange(len(d['R']))):
            raise ValueError('Gold splits must be disjoint and cover all samples')
        selected = {}
        for split, idx in zip(('train', 'val'), splits[:2]):
            limit = dc[f'max_{split}_samples']
            if not isinstance(limit, int) or not 0 < limit <= len(idx):
                raise ValueError(f'Invalid {split} sample limit')
            selected[split] = idx[:limit].tolist()
        # Keep target subtraction in float64 to avoid losing small energy differences.
        energies = np.asarray(d['E'], dtype=np.float64).reshape(-1)
        mean = float(energies[selected['train']].mean())
        centered = {s: torch.tensor(energies[ix] - mean, dtype=torch.float32)
                    for s, ix in selected.items()}
    datasets = {}
    for split in ('train', 'val'):
        dataset = MD17Dataset(path, split=split, cutoff_radius=dc['cutoff_radius'])
        n = len(selected[split])
        dataset.R, dataset.F, dataset.E = dataset.R[:n], dataset.F[:n], centered[split]
        datasets[split] = dataset
    return datasets, mean, selected, hashlib.sha256(path.read_bytes()).hexdigest()


def epoch(model, loader, config, optimizer=None):
    model.train(optimizer is not None)
    totals = dict(energy_abs=0., energy_sq=0., force_abs=0., force_sq=0., ne=0, nf=0)
    for batch in loader:
        batch = batch.to(config['device'])
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        # Validation still needs position gradients for force predictions.
        energy, force = energy_forces(model, batch, create_graph=optimizer is not None)
        de, df = energy - batch.y.reshape(-1), force - batch.force
        loss = config['energy_loss_weight'] * de.square().mean() + config['force_loss_weight'] * df.square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        if optimizer is not None:
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Non-finite parameter gradient')
            optimizer.step()
        de, df = de.detach().double(), df.detach().double()
        totals['energy_abs'] += de.abs().sum().item()
        totals['energy_sq'] += de.square().sum().item()
        totals['force_abs'] += df.abs().sum().item()
        totals['force_sq'] += df.square().sum().item()
        totals['ne'] += de.numel()
        totals['nf'] += df.numel()
    return dict(energy_mae=totals['energy_abs']/totals['ne'],
                energy_rmse=(totals['energy_sq']/totals['ne'])**0.5,
                force_mae=totals['force_abs']/totals['nf'],
                force_rmse=(totals['force_sq']/totals['nf'])**0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != 'dongxiao':
        raise RuntimeError('This smoke runner is restricted to the dongxiao branch')
    config = yaml.safe_load((ROOT / args.config).read_text(encoding='utf-8'))
    tc = config['train']
    output = (ROOT / tc['output_dir']).resolve()
    if not output.is_relative_to((ROOT / 'experiments/results').resolve()):
        raise ValueError('Output must be under experiments/results')
    if output.exists():
        raise FileExistsError(f'Choose a new output_dir; preserving existing run: {output}')
    seed = config['data']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(tc['num_threads'])
    datasets, mean, selected, digest = load_gold(config)
    mc = dict(config['model'])
    if mc.pop('name') != 'dimenetplusplus':
        raise ValueError('Expected dimenetplusplus')
    model = GoldDimeNetPlusPlus(**mc, cutoff=config['data']['cutoff_radius']).to(tc['device'])
    loaders = {s: DataLoader(d, batch_size=tc['batch_size'], shuffle=s == 'train', num_workers=0)
               for s, d in datasets.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=tc['lr'])
    output.mkdir(parents=True)
    report = dict(status='running', purpose='smoke test, not benchmark', config=config,
                  branch=branch, commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  torch=torch.__version__, pyg=torch_geometric.__version__,
                  gold_sha256=digest, selected_gold_indices=selected, energy_mean=mean,
                  units={'energy':'kcal/mol','force':'kcal/mol/Angstrom'},
                  force_metric='mean over Cartesian components', test_evaluated=False,
                  baseline=epoch(model, loaders['val'], tc), history=[])
    best = float('inf')
    for number in range(1, tc['epochs'] + 1):
        train = epoch(model, loaders['train'], tc, optimizer)
        val = epoch(model, loaders['val'], tc)
        record = dict(epoch=number, train=train, validation=val)
        report['history'].append(record)
        score = tc['energy_loss_weight'] * val['energy_rmse']**2 + tc['force_loss_weight'] * val['force_rmse']**2
        if score < best:
            best = score
            report['best_epoch'] = number
            torch.save(dict(model_state=model.state_dict(), energy_mean=mean, config=config), output/'best.pt')
        (output/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(record), flush=True)
    report['status'] = 'completed'
    (output/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Completed: {output}', flush=True)


if __name__ == '__main__':
    main()
