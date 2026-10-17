"""Evaluate the fixed adapter export on the previously inspected public IBC sample."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.special import softmax

from possum.checkpoints import save_json
from possum.data import read_jsonl
from possum.option_a import PARTS, digest, router_weights


def exported_predictions(export, raw, embeddings):
    """Apply saved calibration and heads without fitting or selecting anything."""
    import joblib
    import torch
    config = json.loads((export / 'config.json').read_text())
    names = config['experts']
    p = np.column_stack([softmax(raw[n] / config['temperatures'][n], axis=1)[:, 1] for n in names])
    own = joblib.load(export / 'self_head.joblib')
    own_raw = np.log(np.clip(own.predict_proba(embeddings), 1e-7, 1))
    own_p = softmax(own_raw / config['self_temperature'], axis=1)[:, 1]
    predictions = {n: p[:, j] for j, n in enumerate(names)}
    predictions.update(equal=p.mean(1), compact_equal=p[:, :2].mean(1), self_only=own_p)
    for mode in ('fixed', 'router', 'router_self'):
        for seed in (42, 43, 44):
            key = f'{mode}_{seed}'
            model = torch.nn.Linear(384, 5 if mode == 'router_self' else 4)
            model.load_state_dict(torch.load(export / f'{key}.pt', weights_only=True))
            model.eval()
            with torch.inference_mode():
                w = router_weights(model, torch.tensor(embeddings, dtype=torch.float32), mode).numpy()
            components = np.column_stack([p, own_p]) if mode == 'router_self' else p
            predictions[key] = (w * components).sum(1)
    return predictions, config['selected']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    meta = json.loads((root / 'run.json').read_text())
    if meta['status'] != 'complete':
        raise ValueError('Main training and evaluation must finish before diagnostic scoring')
    if time.time() >= meta['deadline']:
        raise TimeoutError('Run deadline reached')
    export = root / 'export'
    rows = read_jsonl(Path('data/assembled-v1/external_diagnostic.jsonl'))
    if len(rows) != 97 or any(r['source'] != 'ibc_official_sample' for r in rows):
        raise ValueError('Expected the established 97-example public IBC diagnostic')
    # Reject overlap with any model-development partition, without opening locked tests.
    fingerprint = lambda text: hashlib.sha256(' '.join(text.lower().split()).encode()).hexdigest()
    development = sum([read_jsonl(root / 'input' / f'{p}.jsonl') for p in PARTS], [])
    if {r['id'] for r in rows} & {r['id'] for r in development}:
        raise ValueError('Diagnostic IDs overlap development')
    if {fingerprint(r['text']) for r in rows} & {fingerprint(r['text']) for r in development}:
        raise ValueError('Diagnostic text overlaps development')
    out = root / 'ibc-diagnostic'
    out.mkdir(exist_ok=True)
    import fcntl
    with (out / 'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (out / 'input').mkdir(exist_ok=True)
        for part in PARTS:
            path = out / 'input' / f'{part}.jsonl'
            content = ''.join(json.dumps(r) + '\n' for r in rows) if part == 'validation' else ''
            if path.exists() and path.read_text() != content:
                raise ValueError('Diagnostic input changed')
            if not path.exists():
                path.write_text(content)
        for name in ('bert', 'mistral'):
            target = root / f'{name}_adapter'
            info = json.loads((target / 'training.json').read_text())
            filename = 'adapter_model.safetensors' if name == 'bert' else 'adapters.safetensors'
            if digest(target / filename) != info['adapter_sha256']:
                raise ValueError('Trained adapter hash mismatch')
            link = out / target.name
            if not link.exists():
                link.symlink_to(target, target_is_directory=True)
            if link.resolve() != target:
                raise ValueError('Diagnostic adapter link mismatch')
        save_json(out / 'control.json', dict(stage_deadline=meta['deadline'], require_ac=True))
        for name in ('bge', 'bert', 'gliner', 'mistral', 'qwen'):
            subprocess.run([sys.executable, '-u', '-m', 'possum.option_a', f'{name}_score', str(out)],
                           check=True, timeout=max(1, meta['deadline'] - time.time() + 30))
        raw = {}
        for name in ('bge', 'bert', 'gliner', 'mistral', 'qwen'):
            target = out / f'{name}_scores.npz'
            if not json.loads(target.with_suffix('.json').read_text())['complete']:
                raise ValueError('Incomplete diagnostic scores')
            with np.load(target, allow_pickle=False) as z:
                if z['ids'].tolist() != [r['id'] for r in rows]:
                    raise ValueError('Diagnostic score alignment mismatch')
                raw[name] = z['values']
        predictions, selected = exported_predictions(export, raw, raw.pop('bge'))
        from possum.learning import metrics
        y = np.array([r['label'] for r in rows])
        report = dict(scope='Previously inspected human-labeled public IBC diagnostic; not a fresh final test',
                      count=len(rows), selected_before_diagnostic=selected,
                      calibration_and_heads='Loaded unchanged from development export; no diagnostic fitting or selection',
                      metrics={k: metrics(y, v) for k, v in predictions.items()},
                      constant_half=metrics(y, np.full(len(y), .5)),
                      limitations=['Only 97 sentences; original book/author groups unavailable.',
                                   'This sample has already been inspected during earlier development.'])
        save_json(out / 'report.json', report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
