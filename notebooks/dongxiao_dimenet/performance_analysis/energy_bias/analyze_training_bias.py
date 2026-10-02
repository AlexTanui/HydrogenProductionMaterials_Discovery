"""Frozen-model energy residuals on training/validation only; no correction fitted."""
from pathlib import Path
import argparse, json, os, sys
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path.cwd()/'work/mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader

parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
a=parser.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(a.repo))
from ml.training.train_dimenet import load_gold
from ml.models.dimenet import GoldDimeNetPlusPlus
torch.set_num_threads(2)
metrics={}; residuals={}
for budget in (100,200):
    run=a.repo/f'experiments/results/dongxiao_dimenet_ethanol_lr1e4_{budget}'
    report=json.loads((run/'metrics.json').read_text()); assert report['status']=='completed'
    ckpt=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    config=ckpt['config']; datasets,mean,indices,digest=load_gold(config)
    assert mean==ckpt['energy_mean'] and digest==report['gold_sha256']
    mc=dict(config['model']); mc.pop('name')
    model=GoldDimeNetPlusPlus(**mc,cutoff=config['data']['cutoff_radius']); model.load_state_dict(ckpt['model_state']); model.eval()
    gold=np.load(a.repo/config['data']['gold_path'])
    metrics[str(budget)]={'selected_epoch':report['best_epoch'],'splits':{}}
    for split in ('train','val'):
        preds=[]
        with torch.no_grad():
            for batch in DataLoader(datasets[split],batch_size=config['train']['batch_size'],shuffle=False):
                preds.append(model(batch).double().numpy())
        pred=np.concatenate(preds)
        reference=gold['E'][indices[split]].astype(np.float64)-mean
        residuals[(budget,split)]=de=pred-reference
        m={'n':len(de),'signed_bias':float(de.mean()),'mae':float(abs(de).mean()),'rmse':float(np.mean(de**2)**.5),'fraction_overpredicted':float((de>0).mean()),'residual_std':float(de.std()),'mse_bias_fraction':float(de.mean()**2/np.mean(de**2)), 'median_absolute_error':float(np.median(abs(de))), 'p95_absolute_error':float(np.percentile(abs(de),95))}
        metrics[str(budget)]['splits'][split]=m
        if split=='val':
            h=next(h for h in report['history'] if h['epoch']==report['best_epoch'])
            np.testing.assert_allclose(m['mae'],h['validation']['energy_mae'],rtol=0,atol=1e-5)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f7f9fc'})
fig,axs=plt.subplots(2,2,figsize=(13,8),sharex=True)
combined=np.concatenate(list(residuals.values()))
bins=np.linspace(combined.min(),combined.max(),40)
for row,budget in enumerate((100,200)):
    for col,split in enumerate(('train','val')):
        ax=axs[row,col]; de=residuals[(budget,split)]; m=metrics[str(budget)]['splits'][split]
        ax.hist(de,bins=bins,color='#275dad' if budget==100 else '#00877c',alpha=.85)
        ax.axvline(0,color='black',ls='--'); ax.axvline(de.mean(),color='#dc7b24',lw=2,label=f'Mean error: {de.mean():+.3f}')
        ax.set(title=f'{budget}-epoch budget | {"Training" if split=="train" else "Validation"} | selected epoch {metrics[str(budget)]["selected_epoch"]}',xlabel='Energy prediction − reference (kcal/mol)',ylabel='Configurations'); ax.legend()
fig.suptitle('Energy bias | Frozen checkpoints on train and validation',fontsize=19,fontweight='bold')
fig.text(.06,.015,'All predictions use each selected checkpoint in evaluation mode. No test data used, no offset correction applied.\nTraining evaluation here uses a fixed checkpoint; it differs from online training metrics accumulated during optimization.',fontsize=9)
fig.tight_layout(rect=[.02,.07,.99,.95]); fig.savefig(a.out/'train_validation_energy_bias.png',dpi=180); plt.close(fig)
(a.out/'train_validation_energy_bias.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
np.savez_compressed(a.out/'train_validation_energy_residuals.npz',**{f'epochs_{budget}_{split}':de for (budget,split),de in residuals.items()})
print(json.dumps(metrics,indent=2))
