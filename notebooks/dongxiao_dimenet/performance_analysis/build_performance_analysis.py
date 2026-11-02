"""Reproduce descriptive figures from saved predictions; no fitting or test tuning."""
from pathlib import Path
import argparse, base64, hashlib, json, os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR', str(Path.cwd() / 'work' / 'mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--saved', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
a = parser.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
gold_path = a.repo / 'data/gold/md17/ethanol_ccsd_t.npz'
pred_path = a.saved / 'dimenet_test_predictions.npz'
final = json.loads((a.saved / 'dimenet_final_test.json').read_text())
assert hashlib.sha256(gold_path.read_bytes()).hexdigest() == final['gold_sha256']
g, p = np.load(gold_path), np.load(pred_path)
idx = p['gold_idx'].astype(int)
assert len(np.unique(idx)) == 1000 and np.array_equal(np.sort(idx), np.sort(g['test_idx']))
np.testing.assert_allclose(p['energy_true'], g['E'][idx], rtol=0, atol=1e-8)
np.testing.assert_allclose(p['force_true'], g['F'][idx], rtol=0, atol=1e-8)
e = p['energy_pred'] - p['energy_true']
f = p['force_pred'] - p['force_true']
np.testing.assert_allclose(np.abs(e).mean(), final['energy_mae_total'])
np.testing.assert_allclose(np.abs(f).mean(), final['force_mae_component'])
run_names = ['dongxiao_dimenet_ethanol_ccsd_t','dongxiao_dimenet_ethanol_lr1e4_20','dongxiao_dimenet_ethanol_lr1e4_100']
runs = [json.loads((a.repo / 'experiments/results' / n / 'metrics.json').read_text()) for n in run_names]
for r in runs:
    assert r['gold_sha256'] == final['gold_sha256']
    assert r['selected_gold_indices']['validation'] == runs[0]['selected_gold_indices']['validation'] if 'validation' in r['selected_gold_indices'] else r['selected_gold_indices']['val'] == runs[0]['selected_gold_indices']['val']
selected = [next(h for h in r['history'] if h['epoch'] == r['best_epoch'])['validation'] for r in runs]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'#f7f9fc','axes.facecolor':'white','savefig.facecolor':'#f7f9fc'})
blue, teal, orange = '#275dad', '#00877c', '#dc7b24'
def save(fig, name, title, note):
    fig.suptitle(title, fontsize=20, fontweight='bold', x=.06, ha='left', y=.99)
    fig.text(.06,.015,note,fontsize=9,color='#455469')
    fig.tight_layout(rect=[.025,.06,.99,.94],h_pad=3,w_pad=3)
    fig.savefig(a.out / name,dpi=180)
    plt.close(fig)

fig, axs = plt.subplots(2,2,figsize=(14,9))
for ax,key,label in zip(axs[0],['energy_mae','force_mae'],['Energy MAE (kcal/mol)','Force component MAE (kcal/mol/Å)']):
    h = runs[-1]['history']
    for split,c in [('train',blue),('validation',teal)]:
        ax.plot([v['epoch'] for v in h],[v[split][key] for v in h],label=split.title(),color=c,lw=2)
    ax.set(xlabel='Epoch',ylabel=label,title='Selected run: '+label.split(' (')[0]); ax.legend(); ax.grid(alpha=.18)
for ax,key,label in zip(axs[1],['energy_mae','force_mae'],['Energy MAE (kcal/mol)','Force component MAE (kcal/mol/Å)']):
    values = [v[key] for v in selected]
    bars=ax.bar(['5e-4 / 20 epochs','1e-4 / 20 epochs','1e-4 / 100 epochs'],values,color=[blue,orange,teal],width=.6)
    ax.bar_label(bars,fmt='%.3f',padding=5); ax.set(ylim=(0,max(values)*1.22),ylabel=label,title='Best validation checkpoints'); ax.tick_params(axis='x',labelsize=9)
save(fig,'01_training_and_comparison.png','DimeNet++ | Training and validation', 'Same Gold splits: 889 train / 111 validation. Checkpoints selected by energy MSE + 100 × force MSE.\nThe two 1e-4 runs share the same first 20 epochs; they are not independent replicates. Best epoch = 100, at the training budget limit.')

fig, axs = plt.subplots(2,2,figsize=(14,9))
center = p['energy_true'].mean()
for ax,x,y,label in [(axs[0,0],p['energy_true']-center,p['energy_pred']-center,'Energy relative to reference test mean (kcal/mol)'),(axs[0,1],p['force_true'].ravel(),p['force_pred'].ravel(),'Force component (kcal/mol/Å)')]:
    hb=ax.hexbin(x,y,gridsize=48,mincnt=1,bins='log',cmap='viridis'); fig.colorbar(hb,ax=ax,label='Count (log scale)')
    lo=min(x.min(),y.min()); hi=max(x.max(),y.max()); ax.plot([lo,hi],[lo,hi],'--',color=orange,lw=1.5)
    ax.set(xlabel='Reference: '+label,ylabel='Prediction',title='Energy parity' if ax is axs[0,0] else 'Force parity: all 27,000 components')
for ax,vals,unit,title in [(axs[1,0],e,'kcal/mol','Energy residuals'),(axs[1,1],f.ravel(),'kcal/mol/Å','Force component residuals')]:
    ax.hist(vals,bins=45,color=blue,alpha=.85); ax.axvline(0,color='black',ls='--',label='Zero'); ax.axvline(vals.mean(),color=orange,lw=2,label=f'Mean = {vals.mean():+.3f}')
    ax.set(xlabel=f'Prediction − reference ({unit})',ylabel='Count',title=title); ax.legend()
save(fig,'02_test_predictions_and_residuals.png','DimeNet++ | Held-out test performance',f"1,000 Gold test configurations. Energy MAE = {np.abs(e).mean():.3f} kcal/mol; force component MAE = {np.abs(f).mean():.3f} kcal/mol/Å.\nEnergy parity subtracts the same reference mean from both axes for display only. Residuals and metrics are uncorrected.")

per_frame = np.abs(f).mean(axis=(1,2))
by_element = {label:float(np.abs(f[:,g['z']==z,:]).mean()) for label,z in [('C',6),('O',8),('H',1)]}
top = np.argsort(per_frame)[-10:][::-1]
fig, axs = plt.subplots(2,2,figsize=(14,9))
bars=axs[0,0].bar(list(by_element),list(by_element.values()),color=[blue,orange,teal]); axs[0,0].bar_label(bars,fmt='%.3f',padding=5)
axs[0,0].set(title='Force error by element',ylabel='Component MAE (kcal/mol/Å)',ylim=(0,max(by_element.values())*1.2),xlabel='C: 2 atoms / O: 1 atom / H: 6 atoms per configuration')
axs[0,1].hist(per_frame,bins=35,color=teal); axs[0,1].axvline(np.quantile(per_frame,.95),color=orange,label=f'95th percentile = {np.quantile(per_frame,.95):.3f}'); axs[0,1].legend()
axs[0,1].set(title='Distribution of configuration-level force error',xlabel='Mean absolute error across 27 components (kcal/mol/Å)',ylabel='Configurations')
mag=np.linalg.norm(p['force_true'],axis=2).ravel(); atom_err=np.abs(f).mean(axis=2).ravel()
hb=axs[1,0].hexbin(mag,atom_err,gridsize=40,mincnt=1,bins='log',cmap='viridis'); fig.colorbar(hb,ax=axs[1,0],label='Atom count (log scale)')
axs[1,0].set(title='Force magnitude versus prediction error',xlabel='Reference force vector norm (kcal/mol/Å)',ylabel='Per-atom component MAE (kcal/mol/Å)')
axs[1,1].barh([str(int(idx[i])) for i in top][::-1],per_frame[top][::-1],color=blue); axs[1,1].set(title='Ten largest configuration-level force errors',xlabel='Force component MAE (kcal/mol/Å)',ylabel='Gold configuration index')
save(fig,'03_error_diagnostics.png','DimeNet++ | New error diagnostics','Descriptive analysis of the previously evaluated test set; no model tuning or offset correction.\nElement averages use all matching atoms and Cartesian components. Configuration indices refer to the unchanged Gold file.')

stats={'scope':'Descriptive reanalysis of saved test predictions; no new training','n_test':len(e),'energy_signed_bias':float(e.mean()),'fraction_energy_overpredicted':float((e>0).mean()),'energy_mse_bias_fraction':float(e.mean()**2/np.mean(e**2)),'force_component_mae_by_element':by_element,'configuration_force_mae_percentiles':{str(q):float(np.percentile(per_frame,q)) for q in [50,90,95,99]},'worst_force_configurations':[{'gold_index':int(idx[i]),'force_component_mae':float(per_frame[i]),'energy_absolute_error':float(abs(e[i]))} for i in top],'gold_sha256':final['gold_sha256'],'prediction_sha256':hashlib.sha256(pred_path.read_bytes()).hexdigest(),'selected_checkpoint_sha256':final['selection']['checkpoint_sha256'],'test_used_for_additional_tuning':False}
(a.out/'analysis_metrics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
findings=f"""This week I extended the existing DimeNet++ baseline with a reproducible performance analysis. I visualized training and validation, compared the saved experiments at their selected checkpoints, and examined held-out prediction errors. The test energy MAE is {np.abs(e).mean():.3f} kcal/mol and the force component MAE is {np.abs(f).mean():.3f} kcal/mol/angstrom. The energy predictions show a positive mean error of {e.mean():.3f} kcal/mol. The squared mean bias accounts for {stats['energy_mse_bias_fraction']:.1%} of energy mean squared error, so systematic bias is an important issue to investigate. I also broke down force errors by element and identified the ten configurations with the largest force errors. These are new diagnostics of the existing model, not results from a new training run. The next controlled experiment should use training and validation data to investigate energy calibration and loss weighting, with the current test results retained as the baseline."""
notes=f"""# DimeNet++ performance analysis\n\n## New work\n\n{findings}\n\n## Interpretation and limits\n\n- The best saved checkpoint occurs at epoch 100, the training budget limit; convergence is not established.\n- Training metrics aggregate predictions while parameters are changing within each epoch; validation metrics use the end-of-epoch model.\n- Force component MAE is different from force-vector norm error.\n- Element errors: {json.dumps(by_element)} kcal/mol/angstrom. Differences do not establish a chemical cause.\n- This is one molecule and one seed. Components and atoms within configurations are dependent; the plots are descriptive, with no independence-based confidence intervals.\n- No test-derived offset was applied. The bias decomposition is a diagnostic identity, not a corrected model score.\n- Test diagnostics are exploratory. Subsequent hyperparameters should be chosen on validation data; repeatedly consulted test data cannot provide a fresh unbiased final assessment.\n\n## Proposed next experiment (not run)\n\n1. Measure signed energy residuals on training and validation data for the frozen model.\n2. Investigate energy loss weighting and a longer training budget, changing one factor at a time and selecting on validation only.\n3. Repeat the chosen protocol with multiple seeds and report variation.\n4. Agree with the team on a fresh final evaluation protocol before claiming further generalization gains.\n\n## Reproduction\n\nRun build_performance_analysis.py with --repo REPOSITORY --saved DIRECTORY_WITH_SAVED_PREDICTIONS --out OUTPUT_DIRECTORY. Requires NumPy and Matplotlib. The saved predictions are a local artifact and are not currently committed to GitHub. Gold data is read only.\n"""
(a.out/'Performance_Analysis_Readme.md').write_text(notes,encoding='utf-8')
sections=[]
for name,caption in [('01_training_and_comparison.png','Training dynamics and validation-selected experiment comparison.'),('02_test_predictions_and_residuals.png','Original held-out predictions and signed errors; no bias correction.'),('03_error_diagnostics.png','New diagnostic breakdown by element, configuration, and force magnitude.')]:
    encoded=base64.b64encode((a.out/name).read_bytes()).decode()
    sections.append(f'<section><img src="data:image/png;base64,{encoded}" alt="{caption}"><p>{caption}</p></section>')
html='<!doctype html><html lang="en"><meta charset="utf-8"><title>DimeNet++ Performance Review</title><style>body{font:17px/1.6 system-ui;margin:35px auto;max-width:1200px;padding:0 24px;color:#182a41;background:#f7f9fc}h1{font-size:36px}section{margin:35px 0;break-inside:avoid}img{width:100%}article{background:white;padding:25px;border-radius:12px} @media print{body{margin:0}section{break-before:page}}</style><h1>DimeNet++ Performance Review</h1><p>Ethanol CCSD(T) · Gold data · Dongxiao · September 24, 2026</p><article><h2>New this week: performance diagnostics</h2><p>'+findings+'</p></article>'+''.join(sections)+'<article><h2>Next experiment</h2><p>Diagnose energy bias on train/validation, then compare a controlled loss-weight or training-budget change. Select on validation; retain the original test baseline. Multi-seed results remain future work.</p><p>Scope: one molecule, one seed, small CPU model. Existing saved results were reanalyzed; no new training was performed.</p></article></html>'
(a.out/'Performance_Review.html').write_text(html,encoding='utf-8')
print(json.dumps(stats,indent=2))
