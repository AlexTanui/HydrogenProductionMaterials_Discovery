"""Build an English report and actually executed analysis notebook from saved runs.

Run from the repository root. Does not retrain or evaluate the test set.
The notebook cells are executed below in order and their real outputs captured.
"""
import base64
import contextlib
import hashlib
import html
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT/'experiments/results/dongxiao_edge_noise_20'


def main():
    report = json.loads((RUN/'metrics.json').read_text())
    if report['status'] != 'completed':
        raise RuntimeError('Training must complete before the final report is generated')
    (OUT/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    cells, blocks = [], []
    scope = {'__name__':'__notebook__'}
    def markdown(title, text):
        content = '## '+title+'\n\n'+text
        cells.append(dict(cell_type='markdown', metadata={}, source=content))
        blocks.append('<section><h2>'+html.escape(title)+'</h2><p>'+html.escape(text).replace('\n','<br>')+'</p></section>')
    def code(source):
        stdout = io.StringIO()
        outputs = []
        def display_png(path):
            encoded = base64.b64encode(Path(path).read_bytes()).decode()
            outputs.append(dict(output_type='display_data', data={'image/png':encoded}, metadata={}))
        scope['display_png'] = display_png
        with contextlib.redirect_stdout(stdout):
            exec(compile(source, '<executed-report-cell>', 'exec'), scope)
        text = stdout.getvalue()
        if text:
            outputs.insert(0, dict(output_type='stream', name='stdout', text=text))
        count = 1+sum(c['cell_type']=='code' for c in cells)
        cells.append(dict(cell_type='code', metadata={}, source=source, execution_count=count, outputs=outputs))
        rendered = '<details><summary>Executed analysis code</summary><pre>'+html.escape(source)+'</pre></details>'
        if text:
            rendered += '<pre>'+html.escape(text)+'</pre>'
        for item in outputs:
            if item['output_type']=='display_data':
                rendered += '<img alt="Validation performance analysis" src="data:image/png;base64,'+item['data']['image/png']+'">'
        blocks.append('<section>'+rendered+'</section>')

    markdown('DimeNet++: fixed edge-noise pilot',
             'Ethanol CCSD(T), Gold only. Two matched 20-epoch fine-tuning arms from the same 200-epoch checkpoint. '
             'This is an independent DimeNet++ experiment, not the team-wide three-phase MPNN benchmark or a BLIP implementation. '
             'These cells read saved results and generate analysis; they do not claim to rerun training.')
    code('''from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
def display_png(path):
    display(Image(filename=str(path)))
root = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'ml/models/dimenet.py').exists()), None)
if root is None:
    raise RuntimeError('Open this notebook inside the project repository')
import sys
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
out = root/'notebooks/dongxiao_dimenet/edge_noise'
run = root/'experiments/results/dongxiao_edge_noise_20'
r = json.loads((out/'metrics.json').read_text())
print('Status:', r['status'], '| Test evaluated:', r['test_evaluated'])
print('Saved Gold train / validation:', len(r['selected_gold_indices']['train']), '/', len(r['selected_gold_indices']['val']))
print('Gold SHA256:', r['gold_sha256'])
print('Initial checkpoint SHA256:', r['initial_checkpoint_sha256'])
'''.replace("from IPython.display import display, Image\ndef display_png(path):\n    display(Image(filename=str(path)))\n", ''))
    # Saved notebook uses standard IPython image display when opened and rerun.
    cells[-1]['source'] = cells[-1]['source'].replace('import matplotlib.pyplot as plt',
        'import matplotlib.pyplot as plt\nfrom IPython.display import display, Image\ndef display_png(path):\n    display(Image(filename=str(path)))')
    markdown('1. Controlled protocol',
             'Both arms use all 889 saved training structures and all 111 validation structures, seed 42, '
             'fresh Adam optimizers, identical batch order, learning rate 0.0001 and energy MSE + 100 × force-component MSE. '
             'Each arm receives exactly 20 additional epochs. Initial centering uses training energies only. '
             'No additive energy correction is applied here. Noise standard deviation 0.05 was fixed before this run; it was not tuned. '
             'Checkpoint selection considers epochs 1–20 using validation MC-mean joint MSE (8 draws for noise; 1 for control). '
             'Final validation uses a separate random stream with 32 draws. Validation still informed selection, so these are exploratory results.')
    code('''print(json.dumps(r['config'], indent=2))
''')
    markdown('2. Model implementation and physical checks',
             'The initial directed-edge embedding is multiplied by (1 + 0.05 × epsilon), with one independent standard-normal '
             'scalar per directed edge shared across channels. The same stochastic forward produces energy and its force gradient. '
             'Topology, distances, angles and atomic numbers are preserved. Noise is active during both training and MC evaluation. '
             'Geometry tests couple the same edge noise across transformations; permutation tests map noise with edge identity. '
             'This fixed-noise regularization baseline is not a learned Bayesian posterior or proof of calibrated uncertainty.')
    code('''print((root/'ml/models/dongxiao_edge_noise.py').read_text())
''')
    markdown('3. Matched validation performance',
             'The comparison below uses raw centered energy errors and force-component errors. '
             'Do not compare these energy values directly to the previous offset-corrected test results. Lower MAE is better.')
    code('''arms = r['arms']
labels = ['Initial 200 epochs', 'Deterministic +20', 'Edge noise +20']
values = [r['initial_validation'], arms['deterministic']['evaluation'], arms['edge_gaussian']['evaluation']]
for name, arm in arms.items():
    print(name, '| selected additional epoch:', arm['best_epoch'], '| validation:', json.dumps(arm['evaluation'], indent=2))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, metric, unit in zip(axes, ['energy_mae','force_mae'], ['kcal/mol','kcal/mol/Angstrom']):
    bars = ax.bar(labels, [x[metric] for x in values], color=['#94a3b8','#2563eb','#0d9488'])
    ax.bar_label(bars, fmt='%.3f'); ax.set_ylabel(unit); ax.set_title('Validation '+metric.replace('_',' '))
    ax.tick_params(axis='x', labelrotation=12); ax.margins(y=.2)
fig.tight_layout(); fig.savefig(out/'01_matched_validation.png', dpi=160); plt.close(fig)
display_png(out/'01_matched_validation.png')
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, arm in arms.items():
    for ax, key in zip(axes, ['energy_mae','force_mae']):
        ax.plot([h['epoch'] for h in arm['history']], [h['validation'][key] for h in arm['history']], label=name)
        ax.set_title('Validation '+key.replace('_',' ')); ax.set_xlabel('Additional epoch'); ax.legend()
axes[0].set_ylabel('kcal/mol'); axes[1].set_ylabel('kcal/mol/Angstrom')
fig.tight_layout(); fig.savefig(out/'02_training_curves.png', dpi=160); plt.close(fig)
display_png(out/'02_training_curves.png')
''')
    markdown('4. Does uncertainty identify larger errors?',
             'Each point is one molecular structure. Energy uncertainty is sample variance across MC energies; energy error is squared MC-mean error. '
             'Force uncertainty is the mean component variance over atoms and coordinates; force error is structure-level mean squared error. '
             'Spearman correlation measures rank association. It is descriptive on 111 validation structures; no significance or generalization claim is made.')
    code('''from ml.training.dongxiao_edge_experiment import uq_metrics
with np.load(run/'validation_targets.npz') as z:
    te, tf = z['energy'], z['force']
with np.load(run/'edge_gaussian_validation_samples.npz') as z:
    e, f = z['energy'], z['force']
checked = uq_metrics(e, f, te, tf, r['config']['uncertainty'])
for key in ['energy_mae','force_mae','energy_spearman','force_spearman']:
    assert np.isclose(checked[key], arms['edge_gaussian']['evaluation'][key])
print('Saved sample metrics independently recomputed and matched.')
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].scatter(e.var(0,ddof=1), (e.mean(0)-te)**2, s=15, alpha=.7)
axes[1].scatter(f.var(0,ddof=1).mean((1,2)), ((f.mean(0)-tf)**2).mean((1,2)), s=15, alpha=.7)
for ax, kind, unit in zip(axes,['energy','force'], ['(kcal/mol)^2','(kcal/mol/Angstrom)^2']):
    ax.set_xlabel('MC variance ('+unit+')'); ax.set_ylabel('Squared error ('+unit+')')
    ax.set_title(kind.title()+' | Spearman = '+format(checked[kind+'_spearman'], '.3f'))
fig.tight_layout(); fig.savefig(out/'03_uncertainty_error.png', dpi=160); plt.close(fig)
display_png(out/'03_uncertainty_error.png')
''')
    markdown('5. Selective prediction and Monte Carlo cost',
             'Structures with the highest force uncertainty are removed first. The random comparison repeats equally sized random subsets 100 times; '
             'its shaded 5th–95th percentile range is a random-selection reference, not a confidence interval on model performance. '
             'The 8/16/32-draw comparison uses nested prefixes of one MC stream. Agreement with 32 draws does not establish convergence.')
    code('''u = arms['edge_gaussian']
rows = u['evaluation']['rejection']; prefixes = u['mc_prefix']
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
x = [100*z['reject_fraction'] for z in rows]
axes[0].plot(x, [z['force_mae'] for z in rows], 'o-', label='Reject highest uncertainty')
axes[0].plot(x, [z['random_mean'] for z in rows], 'o--', label='Random rejection')
axes[0].fill_between(x, [z['random_p05'] for z in rows], [z['random_p95'] for z in rows], alpha=.2)
axes[0].set_xlabel('Structures rejected (%)'); axes[0].set_ylabel('Retained force MAE (kcal/mol/Angstrom)'); axes[0].legend()
axes[1].plot([z['samples'] for z in prefixes], [z['seconds'] for z in prefixes], 'o-')
axes[1].set_xlabel('MC samples'); axes[1].set_ylabel('Measured total prediction time (seconds)')
fig.tight_layout(); fig.savefig(out/'04_rejection_and_cost.png', dpi=160); plt.close(fig)
display_png(out/'04_rejection_and_cost.png')
for z in prefixes:
    print('MC', z['samples'], '| force MAE', round(z['force_mae'],4), '| force rho', round(z['force_spearman'],4),
          '| force mean shift vs 32', round(z['force_mean_shift_to_largest'],5), '| seconds', round(z['seconds'],3))
''')
    markdown('6. Raw energy interval diagnostic',
             'Central empirical quantiles of stochastic energies are compared with reference energies. '
             'These are uncalibrated model-sampling intervals, not guaranteed prediction intervals: no observation-noise model or coverage adjustment is fitted. '
             'The mean absolute coverage gap across the stated nominal levels is explicitly a regression interval diagnostic, not classification ECE. '
             'Only 32 samples are used, making tail quantiles coarse.')
    code('''coverage = checked['raw_energy_sample_interval_coverage']
fig, ax = plt.subplots(figsize=(5,4))
ax.plot([0,1],[0,1],'--',color='gray',label='Ideal coverage')
ax.plot([z['nominal'] for z in coverage],[z['observed'] for z in coverage],'o-',label='Raw MC energy intervals')
ax.set(xlabel='Nominal coverage', ylabel='Observed validation coverage', xlim=(0,1), ylim=(0,1))
ax.legend(); fig.tight_layout(); fig.savefig(out/'05_raw_interval_coverage.png',dpi=160); plt.close(fig)
display_png(out/'05_raw_interval_coverage.png')
print('Raw interval coverage gap:', checked['raw_interval_coverage_gap'])
''')
    d, s = report['arms']['deterministic']['evaluation'], report['arms']['edge_gaussian']['evaluation']
    conclusion = (f"Compared with matched deterministic fine-tuning, edge noise changed raw validation energy MAE from {d['energy_mae']:.3f} to {s['energy_mae']:.3f} kcal/mol "
                  f"and force-component MAE from {d['force_mae']:.3f} to {s['force_mae']:.3f} kcal/mol/Angstrom. "
                  f"Force uncertainty/error Spearman was {s['force_spearman']:.3f}. "
                  'Energy MAE improved at the selected checkpoints, while force MAE worsened; the joint objective is dominated by force MSE. '
                  'This does not demonstrate overall superiority. Removing the nominal highest-uncertainty 30% of structures '
                  f"reduced force MAE within the noisy model to {s['rejection'][-1]['force_mae']:.3f}; this selected subset is not directly comparable to the control on all structures. "
                  f"The nominal 50% raw energy interval covered {100*s['raw_energy_sample_interval_coverage'][0]['observed']:.1f}% of validation targets, suggesting over-wide intervals in this sample. "
                  'A completed experiment does not imply an accuracy improvement or calibrated uncertainty. '
                  'This is one molecule, one training seed and one fixed noise strength, with validation-based selection and no new test evaluation. '
                  'Previously inspected test results remain historical. Next steps are replication and a team-agreed shared-backbone comparison; '
                  'do not tune parameters on the historical test set.')
    markdown('7. Findings and limitations', conclusion)
    notebook = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(display_name='Python 3 (project .venv)', language='python', name='python3'),
                    language_info=dict(name='python', version='3.12')), cells=cells)
    for i, cell in enumerate(cells):
        cell['id'] = 'edge-noise-'+str(i)
    (OUT/'Edge_Noise_Lab.ipynb').write_text(json.dumps(notebook, indent=2), encoding='utf-8')
    style = 'body{font:17px/1.65 system-ui;color:#153047;background:#f1f5f9;max-width:1120px;margin:40px auto;padding:20px}section{background:white;border-radius:12px;padding:24px;margin:20px 0}img{max-width:100%}pre{font:13px/1.5 monospace;white-space:pre-wrap;overflow-wrap:anywhere}h1,h2{line-height:1.25}summary{cursor:pointer}'
    (OUT/'Edge_Noise_Report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>DimeNet++ Edge Noise Pilot</title><style>'+style+'</style><h1>DimeNet++ Edge Noise Pilot</h1>'+''.join(blocks)+'</html>', encoding='utf-8')
    card = '# Dongxiao: fixed edge-noise DimeNet++ pilot\n\n'+conclusion+'\n\n'
    card += '## Protocol\n\nSee `experiments/configs/dongxiao_edge_noise_20.yaml`. Gold only; saved 889/111 training/validation split. '
    card += 'Both arms start from the same 200-epoch checkpoint with fresh Adam state; 20 epochs each. '
    card += 'No node-noise model, learned noise scale, Bayesian objective, BLIP implementation, or team-wide benchmark is claimed.\n\n'
    for name, arm in report['arms'].items():
        path = RUN/(name+'_best.pt')
        card += f"## {name}\n\nSelected additional epoch: {arm['best_epoch']}. Noise standard deviation: {arm['noise_std']}.\n\nCheckpoint: `{path.relative_to(ROOT).as_posix()}` (ignored by Git).\n\nSHA256: `{hashlib.sha256(path.read_bytes()).hexdigest()}`.\n\n"
    card += '## Evaluation definitions\n\nForce MAE averages Cartesian components. Spearman uses one MSE and one mean component variance per structure. '
    card += 'Empirical energy-interval coverage is uncalibrated and descriptive. Validation is reused for selection and reporting; estimates are not an independent holdout. '
    card += 'All energy metrics in this experiment are raw, without the historical train-fitted additive correction.\n'
    (ROOT/'docs/model_cards/dongxiao_edge_noise_20.md').write_text(card, encoding='utf-8')
    print(conclusion)
    print('Report: '+str(OUT/'Edge_Noise_Report.html'))


if __name__ == '__main__':
    main()
