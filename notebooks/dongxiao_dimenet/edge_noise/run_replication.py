"""Run the predeclared seed-43/44 replications, preserving completed experiments."""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent/'seed_replication'


def main():
    if (ROOT/'.git/HEAD').read_text().strip() != 'ref: refs/heads/dongxiao':
        raise RuntimeError('Only run on dongxiao')
    base_path = ROOT/'experiments/configs/dongxiao_edge_noise_20.yaml'
    base = yaml.safe_load(base_path.read_text())
    original = json.loads((ROOT/base['train']['output_dir']/'metrics.json').read_text())
    if original['status'] != 'completed':
        raise RuntimeError('Original experiment incomplete')
    for path, digest in original['source_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Original experiment source changed: '+path)
    OUT.mkdir(exist_ok=True)
    protocol = dict(date='2026-10-15', seeds=[42,43,44], new_seeds=[43,44],
        purpose='Fine-tuning seed sensitivity from ONE shared pretrained checkpoint; not independent pretraining seeds.',
        fixed='Gold splits, checkpoint, architecture, noise sigma 0.05, 20 epochs per arm, optimizer, validation MC streams and selection criterion.',
        changes='Training batch shuffle seed and training noise seed only. Evaluation streams remain fixed across seeds.',
        stopping_rule='Exactly two additional seed pairs. No hyperparameter search or test evaluation.',
        reporting='Include all three seeds, paired differences and sample standard deviation; no significance claim.')
    p = OUT/'protocol.json'
    if p.exists() and json.loads(p.read_text()) != protocol:
        raise RuntimeError('Preserve existing protocol')
    p.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    configs = []
    for seed in protocol['new_seeds']:
        cfg = copy.deepcopy(base)
        cfg['name'] = f'dongxiao_edge_noise_20_seed{seed}'
        cfg['data']['seed'] = seed
        cfg['train']['output_dir'] = 'experiments/results/'+cfg['name']
        path = ROOT/'experiments/configs'/f"{cfg['name']}.yaml"
        if path.exists() and yaml.safe_load(path.read_text()) != cfg:
            raise RuntimeError('Existing config differs: '+str(path))
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        configs.append((seed, cfg, path))
    for seed, cfg, path in configs:
        result_dir = ROOT/cfg['train']['output_dir']
        if result_dir.exists():
            saved = json.loads((result_dir/'metrics.json').read_text())
            if saved['status']=='completed' and saved['config']==cfg:
                print(f'Seed {seed}: completed run preserved.', flush=True)
                continue
            raise RuntimeError('Partial or conflicting result preserved: '+str(result_dir))
        print(f'Starting seed {seed}: 20 epochs per arm.', flush=True)
        with (OUT/f'seed{seed}_training.log').open('x', encoding='utf-8') as log:
            subprocess.run([sys.executable,'-B','-m','ml.training.dongxiao_edge_experiment','--config',str(path)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'Seed {seed}: both arms completed.', flush=True)
    subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('build_replication_report.py'))],
                   cwd=ROOT, check=True)
    print('Replication training and verified report complete.', flush=True)


if __name__ == '__main__':
    main()
