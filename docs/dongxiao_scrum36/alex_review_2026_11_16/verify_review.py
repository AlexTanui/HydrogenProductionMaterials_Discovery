import json,hashlib,csv
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'source_manifest.json').read_text())
for f in manifest['files']:
    assert hashlib.sha256((root/'sources'/f['path']).read_bytes()).hexdigest()==f['sha256'], f['path']
def load(p):return json.loads((root/'sources'/p).read_text())
runs=['phase2_blip_nll','phase3_graph_stochastic','phase3_graph_stochastic_nll']
rows=[]
for run in runs:
    record=load(f'experiments/runs/{run}/test_metrics.json')
    assert record['rescore']['max_relative_difference'] < record['rescore']['tolerance_relative']
    for method,path in [('CQR',f'reports/calibration/cqr/calibrate_cqr_{run}.json'),('Scalar',f'reports/calibration/calibrate_{run.removeprefix("phase2_").removeprefix("phase3_")}.json')]:
        d=load(path)
        assert d['alpha']==0.1 and d['n_calibration']==55509 and d['n_test']==5000
        for quantity in ['energy','force']:
            a=d['after'][quantity]
            rows.append(dict(run=run,method=method,quantity=quantity,coverage=a.get('coverage_90',a.get('coverage',{}).get('90%')),width=a.get('mean_width',''),ece=a.get('ece','')))
with (root/'calibration_comparison.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Verified',len(manifest['files']),'source hashes; 3 rescore records; 12 calibration rows.')
for quantity in ['energy','force']:
    a=next(r for r in rows if r['run']==runs[0] and r['method']=='CQR' and r['quantity']==quantity)
    b=next(r for r in rows if r['run']==runs[2] and r['method']=='CQR' and r['quantity']==quantity)
    print(quantity,'matched-pair CQR width increase %:',100*(b['width']/a['width']-1))
