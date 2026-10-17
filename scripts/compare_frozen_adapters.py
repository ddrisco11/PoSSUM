"""Matched-development comparison of frozen BERT/Mistral baselines and adapters."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.special import softmax

from possum.checkpoints import save_json, save_scores
from possum.option_a import MODELS, PARTS, data, digest, guard, update


def cache(root, name):
    parts = data(root)
    rows = sum([parts[p] for p in PARTS if name == 'bert' or p != 'fit'], [])
    ids = [r['id'] for r in rows]
    target = root / f'{name}_frozen.npz'
    signature = {p: digest(root / 'input' / f'{p}.jsonl') for p in PARTS}
    values, elapsed = [], []
    if target.exists():
        if json.loads(target.with_suffix('.json').read_text())['input_sha256'] != signature:
            raise ValueError('Baseline inputs changed')
        with np.load(target, allow_pickle=False) as z:
            if z['ids'].tolist() != ids[:len(z['ids'])]:
                raise ValueError('Baseline ID mismatch')
            values, elapsed = z['values'].tolist(), z['chunk_seconds'].tolist()
    if len(values) < len(rows):
        save_json(target.with_suffix('.json'), dict(input_sha256=signature, complete=False))
        guard(root)
        if name == 'bert':
            from possum.components.bert import BERT
            model = BERT(device='mps')
            predict = model.encode
            size = 128
        else:
            from huggingface_hub import snapshot_download
            from possum.components.causal import MLXCausal
            repo, revision = MODELS['mistral']
            model = MLXCausal(snapshot_download(repo, revision=revision, local_files_only=True))
            model.model.eval()
            predict = model.score
            size = 10
        for start in range(len(values), len(rows), size):
            guard(root)
            t = time.perf_counter()
            result = predict([r['text'] for r in rows[start:start+size]])
            if not np.isfinite(result).all():
                raise ValueError('Nonfinite baseline output')
            elapsed.append(time.perf_counter() - t)
            values.extend(result.tolist())
            save_scores(target, values, ids[:len(values)], elapsed)
            update(root, f'{name}_frozen', completed=len(values), total=len(rows),
                   seconds=sum(elapsed), status='scoring')
    save_json(target.with_suffix('.json'), dict(input_sha256=signature, complete=True))
    update(root, f'{name}_frozen', completed=len(values), total=len(rows), seconds=sum(elapsed), status='complete')


def evaluate(root):
    from sklearn.linear_model import LogisticRegression
    from possum.learning import bootstrap_delta, metrics, temperature
    parts = data(root)
    nf, nc, ng, nv = [len(parts[p]) for p in PARTS]
    rows = sum([parts[p] for p in PARTS[1:]], [])
    ids = [r['id'] for r in rows]
    y = np.array([r['label'] for r in rows])
    validation = np.arange(nc+ng, nc+ng+nv)
    def load(name, expected):
        target = root / f'{name}.npz'
        if not json.loads(target.with_suffix('.json').read_text())['complete']:
            raise ValueError('Incomplete baseline cache')
        with np.load(target, allow_pickle=False) as z:
            if z['ids'].tolist() != expected:
                raise ValueError('Baseline alignment mismatch')
            return z['values']
    features = load('bert_frozen', [r['id'] for r in parts['fit']] + ids)
    probe = LogisticRegression(C=1., class_weight='balanced', max_iter=1000, random_state=42)
    probe.fit(features[:nf], [r['label'] for r in parts['fit']])
    raw = {'bert': np.log(np.clip(probe.predict_proba(features[nf:]), 1e-7, 1)),
           'mistral': load('mistral_frozen', ids)}
    temps = {n: temperature(z[:nc], y[:nc]) for n, z in raw.items()}
    pred = {f'{n}_frozen': softmax(z/temps[n], axis=1)[:, 1] for n, z in raw.items()}
    adapted = json.loads((root / 'report.json').read_text())
    for name in ('bert', 'gliner', 'mistral', 'qwen'):
        z = load(f'{name}_scores', ids)
        pred[name] = softmax(z/adapted['temperatures'][name], axis=1)[:, 1]
    pred['equal_frozen'] = np.column_stack([pred[n] for n in ('bert_frozen', 'gliner', 'mistral_frozen', 'qwen')]).mean(1)
    pred['equal_adapted'] = np.column_stack([pred[n] for n in ('bert', 'gliner', 'mistral', 'qwen')]).mean(1)
    pred = {k: p[validation] for k, p in pred.items()}
    labels = y[validation]
    groups = np.array([r['group'] for r in parts['validation']])
    report = dict(scope='Matched development partitions; no reserved-test scoring',
                  counts={p: len(parts[p]) for p in PARTS}, temperatures=temps,
                  metrics={k: metrics(labels, p) for k, p in pred.items()},
                  paired_group_bootstrap={name: bootstrap_delta(labels, pred[name], pred[reference], groups, repeats=2000)
                                          for name, reference in [('bert', 'bert_frozen'), ('mistral', 'mistral_frozen'), ('equal_adapted', 'equal_frozen')]},
                  by_source={},
                  limitations=['Frozen BERT uses a mean-pooled logistic probe; adapted BERT uses its trained sequence classifier, so this compares complete recipes.',
                               'Mistral uses identical prompts and base revision with versus without LoRA, each independently calibrated on the same calibration partition.',
                               'Group bootstrap is descriptive on 21 development groups, not an independent test or a multiple-selection correction.'])
    for source in sorted({r['source'] for r in parts['validation']}):
        mask = np.array([r['source'] == source for r in parts['validation']])
        report['by_source'][source] = {k: metrics(labels[mask], p[mask]) for k, p in pred.items()}
    save_json(root / 'frozen-adapter-comparison.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--worker', choices=['bert', 'mistral'])
    args = parser.parse_args()
    root = args.run.resolve()
    meta = json.loads((root / 'run.json').read_text())
    if meta['status'] != 'complete' or time.time() >= meta['deadline']:
        raise ValueError('Requires completed main evaluation and remaining original budget')
    if args.worker:
        cache(root, args.worker)
    else:
        import fcntl
        with (root / 'baseline-process.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for name in ('bert', 'mistral'):
                subprocess.run([sys.executable, '-u', __file__, '--run', str(root), '--worker', name],
                               check=True, timeout=max(1, meta['deadline'] - time.time() + 30))
            evaluate(root)
