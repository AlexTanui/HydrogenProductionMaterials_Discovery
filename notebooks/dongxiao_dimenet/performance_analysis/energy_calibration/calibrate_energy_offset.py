"""Fit additive offsets on training residuals only; evaluate frozen offsets on validation."""
from pathlib import Path
import argparse, base64, hashlib, json, os, shutil, subprocess
import numpy as np
os.environ.setdefault('MPLCONFIGDIR', str(Path.cwd()/'work/mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
a=parser.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
data=np.load(a.source/'train_validation_energy_residuals.npz')
prior=json.loads((a.source/'train_validation_energy_bias.json').read_text())
def measures(x):
    return dict(mae=float(np.abs(x).mean()),rmse=float(np.mean(x*x)**.5),signed_bias=float(x.mean()),n=int(x.size))
results={}
for budget in (100,200):
    train=data[f'epochs_{budget}_train']; val=data[f'epochs_{budget}_val']
    assert len(train)==889 and len(val)==111 and np.isfinite(train).all() and np.isfinite(val).all()
    offset=float(train.mean())
    np.testing.assert_allclose(offset,prior[str(budget)]['splits']['train']['signed_bias'],rtol=0,atol=1e-12)
    checkpoint=a.repo/f'experiments/results/dongxiao_dimenet_ethanol_lr1e4_{budget}/best.pt'
    report=json.loads((checkpoint.parent/'metrics.json').read_text())
    assert report['best_epoch']==budget
    np.testing.assert_allclose(measures(val)['mae'],report['history'][budget-1]['validation']['energy_mae'],rtol=0,atol=1e-5)
    result={'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'selected_epoch':budget,'gold_sha256':report['gold_sha256'],'energy_offset_to_subtract_kcal_per_mol':offset,'fit_split':'train','fit_samples':len(train),'fit_objective':'minimum training energy MSE with one additive offset','application':'E_calibrated = E_original - offset; original energy includes its training-mean restoration','training':{'original':measures(train),'calibrated':measures(train-offset)},'validation':{'original':measures(val),'calibrated':measures(val-offset)},'test_evaluated':False,'force_effect':'A position-independent constant has zero derivative; no force correction is applied.'}
    results[str(budget)]=result
    (a.out/f'energy_offset_epoch_{budget}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(a.out/'energy_offset_comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f7f9fc'})
fig,axs=plt.subplots(1,2,figsize=(12,5))
x=np.arange(2); width=.34
for ax,key,title in zip(axs,['mae','rmse'],['Validation energy MAE','Validation energy RMSE']):
    for shift,mode,color,label in [(-width/2,'original','#275dad','Original'),(width/2,'calibrated','#00877c','Train-fitted offset')]:
        vals=[results[str(b)]['validation'][mode][key] for b in (100,200)]
        bars=ax.bar(x+shift,vals,width,color=color,label=label); ax.bar_label(bars,fmt='%.3f',padding=4)
    ax.set(xticks=x,xticklabels=['Epoch 100','Epoch 200'],ylabel='kcal/mol',title=title,ylim=(0,2.35)); ax.legend()
fig.suptitle('Energy offset calibration | Training fit, validation evaluation',fontsize=18,fontweight='bold')
fig.text(.06,.02,'One constant per frozen checkpoint, fitted on 889 training configurations only. Validation: 111 configurations.\nNo retraining, no test evaluation. Original checkpoints preserved; forces mathematically unchanged.',fontsize=9)
fig.tight_layout(rect=[.01,.11,.99,.93]); fig.savefig(a.out/'energy_offset_validation.png',dpi=180); plt.close(fig)
rows='\n'.join(f'| {b} | {results[str(b)]["energy_offset_to_subtract_kcal_per_mol"]:.6f} | {results[str(b)]["validation"]["original"]["mae"]:.6f} | {results[str(b)]["validation"]["calibrated"]["mae"]:.6f} | {results[str(b)]["validation"]["calibrated"]["rmse"]:.6f} | {results[str(b)]["validation"]["calibrated"]["signed_bias"]:+.6f} |' for b in (100,200))
md=f'''# Training-fitted energy offset experiment

Date: October 1, 2026. Ethanol CCSD(T), unchanged Gold splits.

For each already-selected checkpoint, fit one constant from its 889 training energy residuals: offset = mean(predicted energy - reference energy). Subtract that frozen constant from future predicted energies. This minimizes training MSE for an additive offset; it is not an MAE-optimal median fit.

## Validation results

All energy values are in kcal/mol. Validation contains 111 configurations. Validation data was used only for evaluation, not offset fitting.

| Epoch | Offset to subtract | Original MAE | Calibrated MAE | Calibrated RMSE | Calibrated mean error |
|---|---:|---:|---:|---:|---:|
{rows}

## Interpretation

This is post-processing of frozen checkpoints, not new training. Each offset belongs to its recorded checkpoint and this fixed-composition ethanol setup; do not transfer it to different models or molecules. Original checkpoints and raw metrics remain unchanged.

Forces are negative position derivatives of energy. The derivative of a position-independent constant is zero, so this offset does not change forces. No new force inference was needed for this experiment.

Reduced validation energy error supports the diagnosis of an additive energy offset. It does not establish a physical explanation for the original training bias or prove generalization to other molecules. Both runs use one seed and share the first 100 training epochs. No test set was evaluated. The earlier test set had already been inspected, so later reuse must be disclosed as follow-up evaluation.

## Reproduction

Run `python -B calibrate_energy_offset.py --repo REPOSITORY --source ENERGY_BIAS_DIRECTORY --out OUTPUT_DIRECTORY`. Source residuals are generated by `analyze_training_bias.py` from the frozen checkpoints. The source directory contains training/validation residuals only. Apply the recorded constant once to the original energy prediction, including the model's usual training-mean restoration.

Per-checkpoint JSON files record the checkpoint hash, Gold hash, fitting rule, offset, and original/calibrated metrics. No validation-derived correction was applied. The calibrated models have not been promoted or deployed.
'''
(a.out/'Energy_Offset_Report.md').write_text(md,encoding='utf-8')
image=base64.b64encode((a.out/'energy_offset_validation.png').read_bytes()).decode()
html='<!doctype html><html lang="en"><meta charset="utf-8"><title>Energy offset calibration</title><style>body{max-width:1100px;margin:35px auto;padding:24px;font:17px/1.6 system-ui;color:#182a41;background:#f7f9fc}img{width:100%}pre{white-space:pre-wrap}</style><h1>Training-fitted energy offset calibration</h1><p>October 1, 2026 · Ethanol CCSD(T) · Gold train/validation</p><img alt="Original and calibrated validation energy errors" src="data:image/png;base64,'+image+'"><p>Each constant was fitted on training data only and then frozen before validation evaluation. Original models remain unchanged. No new training or test evaluation.</p><p><a href="Energy_Offset_Report.md">Full experiment report</a></p><h2>Recorded results</h2><pre>'+json.dumps(results,indent=2)+'</pre></html>'
(a.out/'Energy_Offset_Review.html').write_text(html,encoding='utf-8')
print(json.dumps(results,indent=2))
