"""Frozen follow-up evaluation of the completed 200-epoch checkpoint."""
from pathlib import Path
import argparse, hashlib, json, os, sys
import numpy as np
os.environ.setdefault('MPLCONFIGDIR', str(Path.cwd()/'work/mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
a = parser.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(a.repo))
from ml.data.datasets import MD17Dataset
from ml.models.dimenet import GoldDimeNetPlusPlus, energy_forces

run = a.repo / 'experiments/results/dongxiao_dimenet_ethanol_lr1e4_200'
report = json.loads((run/'metrics.json').read_text())
assert report['status'] == 'completed' and report['best_epoch'] == 200
checkpoint = run/'best.pt'
ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
config = ck['config']
gold = a.repo / config['data']['gold_path']
gold_sha = hashlib.sha256(gold.read_bytes()).hexdigest()
assert gold_sha == report['gold_sha256']
with np.load(gold, allow_pickle=False) as d:
    test_idx = d['test_idx'].copy()
    e_true = np.asarray(d['E'], dtype=np.float64)[test_idx]
    f_true = np.asarray(d['F'], dtype=np.float64)[test_idx]
    n_atoms = len(d['z'])
assert len(test_idx) == 1000 and len(np.unique(test_idx)) == 1000
mc = dict(config['model']); mc.pop('name')
model = GoldDimeNetPlusPlus(**mc, cutoff=config['data']['cutoff_radius'])
model.load_state_dict(ck['model_state'], strict=True)
model.eval()
torch.set_num_threads(config['train']['num_threads'])
test = MD17Dataset(gold, split='test', cutoff_radius=config['data']['cutoff_radius'])
energies, forces = [], []
for batch in DataLoader(test, batch_size=config['train']['batch_size'], shuffle=False, num_workers=0):
    energy, force = energy_forces(model, batch)
    energies.extend(energy.detach().double().tolist())
    forces.append(force.detach().double().numpy().reshape(-1, n_atoms, 3))
e_pred = np.asarray(energies) + float(ck['energy_mean'])
f_pred = np.concatenate(forces)
de, df = e_pred - e_true, f_pred - f_true
offset_file = a.repo/'notebooks/dongxiao_dimenet/performance_analysis/energy_calibration/energy_offset_epoch_200.json'
offset = float(json.loads(offset_file.read_text())['energy_offset_to_subtract_kcal_per_mol'])
de_cal = de - offset
def energy_metrics(x):
    return {'mae_total': float(np.abs(x).mean()), 'rmse_total': float(np.mean(x*x)**.5), 'mae_per_atom': float(np.abs(x/n_atoms).mean()), 'rmse_per_atom': float(np.mean((x/n_atoms)**2)**.5), 'signed_bias': float(x.mean())}
def force_metrics(x):
    return {'mae_component': float(np.abs(x).mean()), 'rmse_component': float(np.mean(x*x)**.5), 'mae_atom_norm': float(np.linalg.norm(x,axis=-1).mean()), 'rmse_atom_norm': float(np.mean(np.sum(x*x,axis=-1))**.5)}
results = {'status':'completed', 'selection':{'run':config['name'], 'best_epoch':200, 'criterion':'validation energy MSE + 100 * force component MSE', 'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'test_used_for_selection':False}, 'n_test':len(test_idx), 'n_atoms':n_atoms, 'gold_sha256':gold_sha, 'original_energy':energy_metrics(de), 'train_fitted_offset':{'offset_to_subtract':offset, 'fit_split':'train', 'fit_samples':889, 'test_evaluation':energy_metrics(de_cal)}, 'force':force_metrics(df), 'units':{'energy':'kcal/mol','force':'kcal/mol/Angstrom'}, 'test_used_for_tuning':False, 'follow_up_to_previous_test_inspection':True, 'test_policy':'Frozen follow-up evaluation after all training and offset decisions; no further tuning permitted.'}
assert np.isfinite(de).all() and np.isfinite(df).all() and np.isfinite(de_cal).all()
(a.out/'dimenet_200_final_test.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
np.savez_compressed(a.out/'dimenet_200_test_predictions.npz', gold_idx=test_idx, energy_true=e_true, energy_pred=e_pred, energy_pred_calibrated=e_pred-offset, force_true=f_true, force_pred=f_pred)
plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False, 'figure.facecolor':'#f7f9fc'})
fig, axs = plt.subplots(1,2,figsize=(12,5))
axs[0].hist(de, bins=40, alpha=.55, color='#275dad', label='Original')
axs[0].hist(de_cal, bins=40, alpha=.55, color='#00877c', label='Train-fitted offset')
axs[0].axvline(0,color='black',ls='--'); axs[0].set(title='200-epoch energy residuals', xlabel='Prediction − reference (kcal/mol)', ylabel='Test configurations'); axs[0].legend()
axs[1].hist(df.ravel(), bins=50, color='#275dad', alpha=.85, label='Original')
axs[1].axvline(0,color='black',ls='--'); axs[1].set(title='200-epoch force residuals', xlabel='Prediction − reference (kcal/mol/Å)', ylabel='Test components'); axs[1].legend()
fig.suptitle('Frozen follow-up evaluation | ethanol CCSD(T) test set', fontsize=18, fontweight='bold')
fig.text(.06,.02,'1,000 held-out Gold configurations. This is a disclosed follow-up evaluation of a previously inspected test set. No test-derived tuning; offset fitted on training data only.',fontsize=9)
fig.tight_layout(rect=[.01,.11,.99,.93]); fig.savefig(a.out/'dimenet_200_test_residuals.png', dpi=180); plt.close(fig)
print(json.dumps(results, indent=2))
