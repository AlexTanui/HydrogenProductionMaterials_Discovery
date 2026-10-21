"""Verify saved MC arrays and report all three fine-tuning seeds; no training."""
import base64
import csv
import hashlib
import html
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ml.training.dongxiao_edge_experiment import uq_metrics

OUT = Path(__file__).resolve().parent/'seed_replication'


def main():
    protocol = json.loads((OUT/'protocol.json').read_text())
    rows, uq, cards, verified = [], [], [], []
    reference = None
    for seed in protocol['seeds']:
        folder = ROOT/'experiments/results'/('dongxiao_edge_noise_20'+(f'_seed{seed}' if seed!=42 else ''))
        r = json.loads((folder/'metrics.json').read_text())
        assert r['status']=='completed' and not r['test_evaluated']
        fixed = {k:r[k] for k in ['gold_sha256','initial_checkpoint_sha256','selected_gold_indices','energy_mean','source_sha256']}
        if reference is None:
            reference = fixed
        assert fixed==reference, 'Data, source or initial checkpoint differs'
        normalized = json.loads(json.dumps(r['config']))
        normalized.pop('name'); normalized['data'].pop('seed'); normalized['train'].pop('output_dir')
        if seed==42:
            fixed_config = normalized
        assert normalized==fixed_config, 'Unexpected protocol change'
        with np.load(folder/'validation_targets.npz') as z:
            te, tf = z['energy'], z['force']
            assert z['indices'].tolist()==r['selected_gold_indices']['val']
        predictions = {}
        for name, arm in r['arms'].items():
            assert len(arm['history'])==20
            assert arm['best_epoch']==min(arm['history'],key=lambda h:h['selection_score'])['epoch']
            with np.load(folder/(name+'_validation_samples.npz')) as z:
                e, f = z['energy'], z['force']
            result = uq_metrics(e, f, te, tf, r['config']['uncertainty'])
            for key in ['energy_mae','force_mae','energy_mse','force_mse']:
                np.testing.assert_allclose(result[key],arm['evaluation'][key],atol=1e-12,rtol=1e-12)
            if name=='edge_gaussian':
                for key in ['energy_spearman','force_spearman','raw_interval_coverage_gap']:
                    np.testing.assert_allclose(result[key],arm['evaluation'][key],atol=1e-12,rtol=1e-12)
                uq.append(dict(seed=seed, **result))
            rows.append(dict(seed=seed, arm=name, selected_epoch=arm['best_epoch'],
                             energy_mae=result['energy_mae'],force_mae=result['force_mae'],
                             joint_mse=result['energy_mse']+100*result['force_mse']))
            predictions[name]=(e,f)
            ckpt=folder/(name+'_best.pt')
            cards.append(f"### Seed {seed}: {name}\n\nSelected additional epoch {arm['best_epoch']}. "
                         f"Checkpoint `{ckpt.relative_to(ROOT).as_posix()}` (Git ignored). "
                         f"SHA256 `{hashlib.sha256(ckpt.read_bytes()).hexdigest()}`.\n")
        # Compare both models on exactly the same structures retained by noisy-model ranking.
        _, noisy_f = predictions['edge_gaussian']
        uncertainty=noisy_f.var(0,ddof=1).mean((1,2))
        n=int(len(tf)*.7)
        retained=np.argsort(uncertainty,kind='stable')[:n]
        uq[-1]['same_retained_subset']={name:float(np.abs(f.mean(0)[retained]-tf[retained]).mean())
                                         for name, (_,f) in predictions.items()}
        uq[-1]['retained_count']=n
        verified.append(dict(seed=seed, metrics_sha256=hashlib.sha256((folder/'metrics.json').read_bytes()).hexdigest()))
    assert hashlib.sha256((ROOT/'data/gold/md17/ethanol_ccsd_t.npz').read_bytes()).hexdigest()==reference['gold_sha256']
    aggregate={}
    for name in ['deterministic','edge_gaussian']:
        rr=[r for r in rows if r['arm']==name]
        aggregate[name]={key:dict(mean=float(np.mean([r[key] for r in rr])),
                                  sample_sd=float(np.std([r[key] for r in rr],ddof=1)))
                         for key in ['energy_mae','force_mae','joint_mse']}
    pairs=[]
    for seed in protocol['seeds']:
        d=next(r for r in rows if r['seed']==seed and r['arm']=='deterministic')
        s=next(r for r in rows if r['seed']==seed and r['arm']=='edge_gaussian')
        pairs.append(dict(seed=seed, **{key:s[key]-d[key] for key in ['energy_mae','force_mae','joint_mse']}))
    result=dict(protocol=protocol,verification=verified,rows=rows,aggregate=aggregate,
                paired_differences_noise_minus_control=pairs,uncertainty=uq,test_evaluated=False)
    (OUT/'replication_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    with (OUT/'replication_metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,key,unit in zip(axes,['energy_mae','force_mae'],['kcal/mol','kcal/mol/Angstrom']):
        for seed,color in zip(protocol['seeds'],['#2563eb','#0d9488','#b45309']):
            rr=[next(r for r in rows if r['seed']==seed and r['arm']==name) for name in ['deterministic','edge_gaussian']]
            ax.plot([0,1],[r[key] for r in rr],'o-',label=f'Seed {seed}',color=color)
        ax.set_xticks([0,1],['Deterministic +20','Edge noise +20']);ax.set_ylabel(unit)
        ax.set_title('Validation '+key.replace('_',' '));ax.legend()
    fig.tight_layout();fig.savefig(OUT/'01_paired_seeds.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for q,color in zip(uq,['#2563eb','#0d9488','#b45309']):
        axes[0].scatter(q['seed'],q['force_spearman'],color=color,s=60)
        axes[0].annotate(f"{q['force_spearman']:.3f}",(q['seed'],q['force_spearman']),xytext=(0,8),textcoords='offset points',ha='center')
        rej=q['rejection'];axes[1].plot([100*x['reject_fraction'] for x in rej],[x['force_mae'] for x in rej],'o-',label=f"Seed {q['seed']}",color=color)
    axes[0].set(xticks=protocol['seeds'],xlabel='Fine-tuning seed',ylabel='Force uncertainty/error Spearman',ylim=(-.1,1))
    axes[0].axhline(0,color='gray',linestyle='--');axes[1].set(xlabel='Nominal rejected structures (%)',ylabel='Retained noisy-model force MAE (kcal/mol/Angstrom)');axes[1].legend()
    fig.tight_layout();fig.savefig(OUT/'02_uq_replication.png',dpi=160);plt.close(fig)
    e_wins=sum(p['energy_mae']<0 for p in pairs);f_wins=sum(p['force_mae']<0 for p in pairs)
    conclusion=(f'At validation-selected checkpoints, edge noise improved raw energy MAE in {e_wins}/3 fine-tuning seeds '
                f'and force MAE in {f_wins}/3 seeds. These are paired runs from one pretrained checkpoint, '
                'not three independent end-to-end training initializations. No test evaluation was performed.')
    lines=['# DimeNet++ edge-noise replication study','',conclusion,'',
           '## Protocol','',protocol['purpose'], '',protocol['fixed'], '',protocol['changes'],'',
           'Seed 42 is the earlier pilot; seeds 43 and 44 were specified before running the two replications. All runs are included. '
           'Fresh Adam optimizers and matched batch order within each pair. Same 889/111 Gold split. '
           'Checkpoint selection uses energy MSE + 100 times force-component MSE; no energy offset correction. '
           'Selection uses 8 noise samples; reporting uses 32 samples with a separate fixed random stream.','',
           '## Per-seed validation results','',
           '| Seed | Arm | Selected extra epoch | Energy MAE (kcal/mol) | Force MAE (kcal/mol/Angstrom) |',
           '|---|---|---|---|---|']
    for row in rows:
        lines.append(f"| {row['seed']} | {row['arm']} | {row['selected_epoch']} | {row['energy_mae']:.4f} | {row['force_mae']:.4f} |")
    lines+=['','## Mean and sample standard deviation across fine-tuning seeds','']
    for name,a in aggregate.items():
        lines.append(f"- {name}: energy MAE {a['energy_mae']['mean']:.4f} +/- {a['energy_mae']['sample_sd']:.4f}; force MAE {a['force_mae']['mean']:.4f} +/- {a['force_mae']['sample_sd']:.4f} (units as above).")
    lines+=['','Standard deviations describe these three runs; they are not confidence intervals or proof of significance.',
            '', '## Uncertainty and matched retained subsets','',
            'Force Spearman compares structure-level mean MC variance with structure-level mean squared error. '
            'Both models below are evaluated on the SAME 77 structures retained after ranking by noisy-model uncertainty. '
            'Nominal rejection is 30%; rounding retains 77 of 111 structures.','',
            '| Seed | Force Spearman | Retained control force MAE | Retained noisy force MAE | Observed coverage of nominal 50% energy interval |',
            '|---|---|---|---|---|']
    for q in uq:
        subset=q['same_retained_subset']
        lines.append(f"| {q['seed']} | {q['force_spearman']:.3f} | {subset['deterministic']:.4f} | {subset['edge_gaussian']:.4f} | {100*q['raw_energy_sample_interval_coverage'][0]['observed']:.1f}% |")
    lines+=['','## Limitations and next step','',
            'One molecule, one pretrained checkpoint, three fine-tuning seeds and one fixed noise strength. '
            'Validation was used for model selection and reporting. Fixed MC streams reduce evaluation randomness but do not measure independent MC-seed variability. '
            'Raw energy-sampling intervals are not calibrated predictive intervals. No Bayesian posterior, OOD robustness or overall superiority is established. '
            'Do not replace the deterministic baseline solely on these diagnostics. A smaller predeclared noise-strength experiment or a team-shared backbone evaluation is a possible next study; no such study has been run here.',
            '', '## Verification','',
            'All six selected-model errors were recomputed from saved MC arrays. Selection epochs, configs, source hashes, Gold checksum, '
            'saved split indices and common starting checkpoint were checked. Each arm completed exactly 20 epochs. Original results were preserved.']
    md='\n'.join(lines)+'\n'
    (OUT/'Replication_Report.md').write_text(md,encoding='utf-8')
    # Render the small report without adding a Markdown package dependency.
    body=[];in_table=False
    for line in lines:
        if line.startswith('|'):
            if line.startswith('|---'):continue
            if not in_table:body.append('<table>');in_table=True
            body.append('<tr>'+''.join('<td>'+html.escape(c.strip())+'</td>' for c in line.strip('|').split('|'))+'</tr>')
        else:
            if in_table:body.append('</table>');in_table=False
            if line.startswith('# '):body.append('<h1>'+html.escape(line[2:])+'</h1>')
            elif line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
            elif line:body.append('<p>'+html.escape(line)+'</p>')
    if in_table:body.append('</table>')
    for name in ['01_paired_seeds.png','02_uq_replication.png']:
        body.append('<img alt="Fine-tuning seed replication chart" src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'">')
    body.append('<details><summary>Executed analysis source</summary><pre>'+html.escape(Path(__file__).read_text())+'</pre></details>')
    style='body{font:17px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:24px;color:#153047;background:#f8fafc}table{border-collapse:collapse;width:100%;font-size:15px;background:white}td{padding:10px;border:1px solid #cbd5e1}tr:first-child{font-weight:bold;background:#e2e8f0}img{width:100%;margin:20px 0}pre{white-space:pre-wrap;font-size:12px}h2{margin-top:36px}'
    (OUT/'Replication_Report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Edge-noise seed replication</title><style>'+style+'</style>'+''.join(body)+'</html>',encoding='utf-8')
    (ROOT/'docs/model_cards/dongxiao_edge_noise_replication.md').write_text('# Edge-noise replication checkpoints\n\n'+conclusion+'\n\n'+
        'Protocol and validation-only limitations: `notebooks/dongxiao_dimenet/edge_noise/seed_replication/Replication_Report.md`.\n\n'+'\n'.join(cards),encoding='utf-8')
    print(conclusion)
    print(json.dumps(aggregate,indent=2))
    print('Verified all six selected checkpoints from saved predictions. Report: '+str(OUT/'Replication_Report.html'))


if __name__=='__main__':
    main()
