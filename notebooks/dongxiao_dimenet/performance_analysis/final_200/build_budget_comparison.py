"""Compare completed 100- and 200-epoch runs without reading test predictions."""
from pathlib import Path
import argparse, base64, hashlib, json, os, contextlib, io
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path.cwd()/'work/mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
paths=[args.repo/f'experiments/results/dongxiao_dimenet_ethanol_lr1e4_{n}/metrics.json' for n in (100,200)]
old,new=[json.loads(p.read_text()) for p in paths]
assert old['status']==new['status']=='completed'
assert len(new['history'])==200
assert old['gold_sha256']==new['gold_sha256']
assert old['selected_gold_indices']==new['selected_gold_indices']
assert old['config']['data']==new['config']['data']
assert old['config']['model']==new['config']['model']
for key,value in old['config']['train'].items():
    if key not in ('epochs','output_dir'): assert new['config']['train'][key]==value
prefix=max(abs(x[split][key]-y[split][key]) for x,y in zip(old['history'],new['history'][:100]) for split in ('train','validation') for key in x[split])
def best(run):
    h=min(run['history'],key=lambda h:h['validation']['energy_rmse']**2+100*h['validation']['force_rmse']**2)
    assert h['epoch']==run['best_epoch']
    return h
b0,b1=best(old),best(new)
v0,v1=b0['validation'],b1['validation']
changes={k:100*(v0[k]-v1[k])/v0[k] for k in v0}
scores=[v['energy_rmse']**2+100*v['force_rmse']**2 for v in (v0,v1)]
summary={'scope':'Training-budget ablation; validation-only comparison','old_best_epoch':b0['epoch'],'new_best_epoch':b1['epoch'],'old_validation':v0,'new_validation':v1,'relative_reduction_percent':changes,'selection_score_old':scores[0],'selection_score_new':scores[1],'first_100_epochs_max_absolute_metric_difference':prefix,'same_data_model_seed_optimizer':True,'test_evaluated':new['test_evaluated'],'gold_sha256':new['gold_sha256'],'new_checkpoint_sha256':hashlib.sha256((paths[1].parent/'best.pt').read_bytes()).hexdigest()}
assert summary['test_evaluated'] is False
(args.out/'training_budget_comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f7f9fc','axes.titleweight':'bold'})
fig,axs=plt.subplots(2,2,figsize=(14,9))
for ax,key,title,unit in zip(axs[0],['energy_mae','force_mae'],['Validation energy MAE','Validation force component MAE'],['kcal/mol','kcal/mol/Å']):
    h=new['history']; ax.plot([x['epoch'] for x in h],[x['validation'][key] for x in h],color='#00877c',lw=1.6,label='200-epoch run')
    ax.axvline(100,color='#dc7b24',ls='--',label='Previous training budget')
    ax.scatter([b0['epoch'],b1['epoch']],[v0[key],v1[key]],color=['#275dad','#ba3a63'],zorder=4,s=65)
    ax.set(title=title,xlabel='Epoch',ylabel=unit); ax.legend(); ax.grid(alpha=.15)
for ax,key,title,unit in zip(axs[1],['energy_mae','force_mae'],['Selected-checkpoint energy MAE','Selected-checkpoint force MAE'],['kcal/mol','kcal/mol/Å']):
    values=[v0[key],v1[key]]; bars=ax.bar([f'100-epoch budget\nselected epoch {b0["epoch"]}',f'200-epoch budget\nselected epoch {b1["epoch"]}'],values,color=['#275dad','#00877c'],width=.55)
    ax.bar_label(bars,fmt='%.4f',padding=5); ax.set(title=title,ylabel=unit,ylim=(0,max(values)*1.22))
fig.suptitle('DimeNet++ | New training: 100 versus 200 epochs',x=.055,ha='left',fontsize=21,fontweight='bold')
fig.text(.055,.015,'Same Gold splits, model, seed 42, learning rate 1e-4, and loss weights. Selection: validation energy MSE + 100 × force MSE.\nFresh deterministic replay to 200 epochs; not a resumed optimizer or an independent seed replicate. Test set not evaluated.',fontsize=10,color='#455469')
fig.tight_layout(rect=[.02,.07,.99,.95],h_pad=3,w_pad=3)
fig.savefig(args.out/'training_budget_comparison.png',dpi=180); plt.close(fig)
table='\n'.join(f'| {k} | {v0[k]:.6f} | {v1[k]:.6f} | {changes[k]:+.2f}% |' for k in v0)
speech=f"Last week, I completed the 100-epoch DimeNet++ baseline. This week, I ran a controlled 200-epoch training experiment on the same ethanol Gold data. I kept the architecture, random seed, learning rate, loss weights, and data splits fixed. The first 100 epochs reproduced the previous run with a maximum absolute metric difference of {prefix:.3g}. I selected the new checkpoint at epoch {b1['epoch']} using the same validation criterion. Validation energy MAE changed from {v0['energy_mae']:.3f} to {v1['energy_mae']:.3f} kcal/mol, and force component MAE changed from {v0['force_mae']:.3f} to {v1['force_mae']:.3f} kcal/mol per angstrom. This is a training-budget comparison, not a multi-seed result. I have not evaluated the new model on the test set."
md=f'''# DimeNet++ training-budget experiment

## Completed experiment

The 200-epoch training-budget experiment is complete. Under the original joint validation selection rule, epoch {b1['epoch']} is selected. Energy MAE increased by {-changes['energy_mae']:.2f}%, while force component MAE decreased by {changes['force_mae']:.2f}%. The result is a trade-off, not an improvement in every target. This report supersedes the 144-epoch interim snapshot.

## Controlled protocol

- Ethanol CCSD(T), unchanged Gold data: 889 training and 111 validation configurations.
- Seed 42; learning rate 0.0001; energy/force loss weights 1/100; unchanged architecture and batch size.
- One changed training factor: epoch budget, 100 to 200.
- Started from the same seeded initialization because the old checkpoint has no optimizer state. This is not an exact optimizer resume and is not an independent replicate.
- Checkpoint selection: minimum validation energy RMSE squared plus 100 times force RMSE squared.
- New selected epoch: {b1['epoch']}. Budget completion alone does not establish convergence.
- Maximum metric difference over the first 100 epochs: {prefix:.12g}.
- No test evaluation or test-derived correction was performed for this experiment.

## Validation results

Energy units: kcal/mol. Force units: kcal/mol/angstrom; Cartesian component metrics.
Positive reduction means a lower error; negative reduction means a higher error.

| Metric | Previous selected checkpoint | New selected checkpoint | Relative reduction |
|---|---:|---:|---:|
{table}

Weighted selection score: {scores[0]:.6f} to {scores[1]:.6f}. Individual metrics need not all improve under a joint selection rule.

## Limitations and next step

This is one seed and one molecule. Shared-prefix runs are dependent. Validation was used for model selection, so these scores are not a fresh generalization estimate. Earlier test-error analysis informed possible follow-up work; do not present the old test set as an untouched new assessment. Repeat a fixed protocol with additional seeds and agree on a final evaluation protocol with the team.

## Reproduction

Training: `python -B -m ml.training.train_dimenet --config experiments/configs/dimenet_ethanol_lr1e4_200.yaml`

Run this only on dongxiao. The training runner deliberately refuses to overwrite an existing result directory. This command starts 200 epochs from scratch.

Analysis: `python build_budget_comparison.py --repo REPOSITORY --out OUTPUT_DIRECTORY`

The experiment reuses the existing runner, whose saved purpose field still says smoke test; this report describes the actual full-split training-budget comparison. Model weights stay local.

## Likely meeting questions

**What is new compared with last week?** A completed 200-epoch training run and a controlled validation comparison, in addition to the error diagnostics. Last week's budget was 100 epochs.

**Why did you restart training?** The old checkpoint did not save Adam's optimizer state. Restarting with the same seed and verifying the shared prefix makes the budget comparison transparent.

**Does this prove the model generalizes better?** No. These are validation-selected results from one molecule and one seed. A fresh final evaluation and multiple seeds are still needed for stronger claims.

**Why not change the architecture at the same time?** Holding other settings fixed isolates the effect of the additional training budget.
'''
main_report, questions = md.split('## Likely meeting questions', 1)
md=main_report
(args.out/'Training_Budget_Report.md').write_text(md,encoding='utf-8')
image=base64.b64encode((args.out/'training_budget_comparison.png').read_bytes()).decode()
rows=''.join(f'<tr><td>{k}</td><td>{v0[k]:.4f}</td><td>{v1[k]:.4f}</td><td>{changes[k]:+.2f}%</td></tr>' for k in v0)
html=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>New DimeNet++ Training Experiment</title><style>body{{max-width:1150px;margin:40px auto;padding:20px;font:17px/1.6 system-ui;color:#182a41;background:#f7f9fc}}img{{width:100%}}article{{background:white;padding:24px}}td,th{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}table{{width:100%;border-collapse:collapse}}</style><h1>New training: 100 versus 200 epochs</h1><p>Dongxiao | Ethanol CCSD(T) | Gold-only | Validation-only comparison</p><article><h2>Meeting script</h2><p>{speech}</p></article><img alt="Training budget comparison" src="data:image/png;base64,{image}"><table><tr><th>Metric</th><th>100-epoch budget</th><th>200-epoch budget</th><th>Error reduction</th></tr>{rows}</table><p>Energy: kcal/mol. Force: Cartesian components, kcal/mol/Å. Both checkpoints were selected using the same joint validation criterion.</p><h2>What this establishes</h2><p>A controlled comparison of training budgets for this seed and molecule. No new test score was produced. This does not establish multi-seed robustness or convergence.</p></html>'''
html=html.replace(f'<article><h2>Meeting script</h2><p>{speech}</p></article>', f'<article><h2>Completed training: 200/200 epochs</h2><p>Selected epoch: {b1["epoch"]}. Energy MAE increased by {-changes["energy_mae"]:.1f}%; force component MAE decreased by {changes["force_mae"]:.1f}%. This is a trade-off under the original joint validation criterion.</p><p>Report finalized October 1, 2026. Supersedes the September 24 snapshot at epoch 144.</p></article>')
html=html.replace('Both checkpoints were selected using the same joint validation criterion.', 'Both checkpoints were selected using validation energy MSE + 100 × force component MSE. Positive reduction means lower error; negative reduction means higher error.')
html=html.replace('</html>', '<p>The two runs share the same first 100 epochs and are not independent seed replicates. The selected epoch is at the budget limit, so convergence is not established. Earlier test diagnostics were already inspected; future evaluation on that same set must be disclosed as follow-up evaluation.</p></html>')
(args.out/'New_Training_Review.html').write_text(html,encoding='utf-8')
# Record the actual analysis invocation and its resulting summary in a notebook.
for budget,run in [(100,old),(200,new)]:
    (args.out/f'run_{budget}_metrics.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
code='''import json
from pathlib import Path
old = json.loads(Path('run_100_metrics.json').read_text())
new = json.loads(Path('run_200_metrics.json').read_text())
assert old['history'] == new['history'][:100]
assert old['selected_gold_indices'] == new['selected_gold_indices']
assert old['gold_sha256'] == new['gold_sha256']
def selected(run):
    return min(run['history'], key=lambda h:
        h['validation']['energy_rmse']**2 + 100*h['validation']['force_rmse']**2)
previous, current = selected(old), selected(new)
print('Selected epochs:', previous['epoch'], current['epoch'])
for metric, before in previous['validation'].items():
    after = current['validation'][metric]
    print(f'{metric}: {before:.6f} -> {after:.6f}; reduction = {100*(before-after)/before:+.2f}%')
print('First 100 epochs identical. No new test evaluation.')'''
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[{'cell_type':'markdown','metadata':{},'source':md.splitlines(keepends=True)},{'cell_type':'code','execution_count':None,'metadata':{},'source':code.splitlines(keepends=True),'outputs':[]},{'cell_type':'markdown','metadata':{},'source':['## Saved analysis output\n','The report below records the completed analysis; rerun the cell above to load it locally.\n','```json\n',json.dumps(summary,indent=2),'\n```']},{'cell_type':'markdown','metadata':{},'source':['## Training comparison\n','![Training comparison](attachment:training_budget_comparison.png)'],'attachments':{'training_budget_comparison.png':{'image/png':image}}}]}
captured=io.StringIO()
previous=Path.cwd()
try:
    os.chdir(args.out.resolve())
    with contextlib.redirect_stdout(captured):
        exec(compile(code,'budget_summary_cell','exec'),{})
finally:
    os.chdir(previous)
nb['cells'][1]['execution_count']=1
nb['cells'][1]['outputs']=[{'output_type':'stream','name':'stdout','text':captured.getvalue().splitlines(keepends=True)}]
for i,c in enumerate(nb['cells']): c['id']=f'budget-{i}'
(args.out/'Training_Budget_Analysis.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
