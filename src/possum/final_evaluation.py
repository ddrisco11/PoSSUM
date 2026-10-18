"""Freeze verified candidates before a resumable, one-time topic holdout audit."""
import argparse
import datetime
import fcntl
import json
import time
from pathlib import Path

import numpy as np

from .checkpoints import save_json, save_scores
from .improvement import digest, report_predictions


def decision_summary(y, p, threshold=.5):
    """Fixed-threshold errors, with undefined absent-class recall kept explicit."""
    y, p = np.asarray(y), np.asarray(p)
    assert y.shape == p.shape and len(y) and set(np.unique(y)) <= {0, 1}
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    predicted = p >= threshold
    matrix = [[int(((y == c) & (predicted == d)).sum()) for d in [0, 1]] for c in [0, 1]]
    recalls = {}
    for c in [0, 1]:
        n, correct = sum(matrix[c]), matrix[c][c]
        if not n:
            recalls[str(c)] = {'n': 0, 'recall': None, 'wilson_95_interval': None}
            continue
        rate, z = correct / n, 1.959963984540054
        center = (rate + z * z / (2 * n)) / (1 + z * z / n)
        radius = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        recalls[str(c)] = {'n': n, 'recall': rate,
                           'wilson_95_interval': [max(0., float(center - radius)), min(1., float(center + radius))]}
    return {'threshold': threshold, 'confusion_matrix_true_rows_predicted_columns': matrix,
            'class_order': [0, 1], 'class_recalls': recalls,
            'balanced_accuracy': float(np.mean([recalls[str(c)]['recall'] for c in [0, 1]]))
                if all(sum(row) for row in matrix) else None}


def uncertainty(y, p, draws=2000, seed=42):
    """Descriptive fixed-class-count intervals; no holdout-dependent choices."""
    y, p = np.asarray(y), np.asarray(p)
    decision_summary(y, p)  # Validate probability range and binary labels.
    rng = np.random.default_rng(seed)
    indices = np.concatenate([rng.choice(np.flatnonzero(y == c), size=(draws, int((y == c).sum())), replace=True)
                              for c in [0, 1] if (y == c).any()], axis=1)
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    losses = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    samples = {'log_loss': losses[indices].mean(axis=1), 'brier': ((p - y) ** 2)[indices].mean(axis=1)}
    yy, pp = y[indices], p[indices] >= .5
    tp, tn = ((yy == 1) & pp).sum(axis=1), ((yy == 0) & ~pp).sum(axis=1)
    fp, fn = ((yy == 0) & pp).sum(axis=1), ((yy == 1) & ~pp).sum(axis=1)
    if set(np.unique(y)) == {0, 1}:
        samples['balanced_accuracy'] = (tp / (tp + fn) + tn / (tn + fp)) / 2
        samples['macro_f1'] = (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn)) / 2
    return {'percentile_95_intervals': {key: np.quantile(value, [.025, .975]).tolist() for key, value in samples.items()},
            'draws': draws, 'seed': seed, 'method': 'Class-stratified row bootstrap with fixed observed class counts.',
            'limitations': 'Conditional on these passages and class counts; does not capture topic/author dependence, label uncertainty or population shift. Only three topic groups cannot establish broad topic generalization.'}


def deadline(root):
    return datetime.datetime.fromisoformat(json.loads((root / 'state.json').read_text())['deadline']).timestamp()


def bundle_record(root, bundle):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / 'model.json').read_text())
    verification = json.loads((bundle / 'verification.json').read_text())
    signature = json.loads((root / 'dataset.json').read_text())['manifest_sha256']
    assert manifest['dataset']['manifest_sha256'] == signature, 'Different dataset'
    assert verification['passed'], 'Bundle must pass reload verification'
    for path, sha in manifest['artifact_hashes'].items():
        assert digest(bundle / path) == sha, f'Changed artifact: {path}'
    return {'path': str(bundle), 'manifest_sha256': digest(bundle / 'model.json'),
            'selected_name': manifest['selected_name'], 'validation': manifest['validation'],
            'decision_rules': manifest.get('decision_rules')}


def freeze(root, compact_bundle=None, allow_incomplete=False, reason=None):
    lock = (root / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    marker = root / 'selection-freeze.json'
    if marker.exists():
        raise ValueError('Selection is already frozen; resume evaluation without changing candidates')
    if time.time() >= deadline(root):
        raise TimeoutError('Development deadline reached')
    dependencies = ['modernbert-finished.json', 'replication-finished.json', 'distillation-finished.json',
                    'experts/qwen_chat.json', 'experts/mistral_human_lora_e2.json']
    if (root / 'deberta-plan.json').exists():
        dependencies.append('deberta-finished.json')
    unfinished = [p for p in dependencies if not (root / p).exists() or not json.loads((root / p).read_text()).get('complete')]
    if unfinished and (not allow_incomplete or not reason):
        raise ValueError('Planned comparisons incomplete; a documented partial-run freeze is required')
    selected = json.loads((root / 'export.json').read_text())
    records = {'selected': bundle_record(root, selected['path'])}
    if compact_bundle is not None:
        record = bundle_record(root, compact_bundle)
        manifest = json.loads((Path(record['path']) / 'model.json').read_text())
        assert len(manifest['components']) == 1 and next(iter(manifest['components'].values()))['kind'] == 'compact', 'Expected one compact model'
        records['compact'] = record
    dataset = json.loads((root / 'dataset.json').read_text())
    manifest_path = Path(dataset['path']) / 'manifest.json'
    assert digest(manifest_path) == dataset['manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    plan = {'dataset': dataset, 'holdout_sha256': manifest['split_hashes']['topic_holdout'], 'bundles': records,
            'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'unfinished': unfinished,
            'reason': reason or 'Planned comparisons complete; candidates selected using development validation only.',
            'rules': ['No further fitting, calibration, routing or candidate selection on this run.',
                      'Evaluate the fixed bundles once; interrupted predictions resume with the same hashes.',
                      'Topic stress is not author-disjoint and topics may overlap synthetic policy themes.']}
    save_json(marker, plan)
    return plan


def evaluate(root, device='mps'):
    lock = (root / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    marker = root / 'selection-freeze.json'
    if not marker.exists():
        raise ValueError('Freeze candidate selection before opening the topic holdout')
    plan = json.loads(marker.read_text())
    signature = digest(marker)
    if time.time() >= deadline(root):
        raise TimeoutError('No new evaluation after the deadline')
    assert json.loads((root / 'dataset.json').read_text()) == plan['dataset']
    for record in plan['bundles'].values():
        assert digest(Path(record['path']) / 'model.json') == record['manifest_sha256'], 'Frozen bundle changed'
        bundle_record(root, record['path'])
    out = root / 'final-evaluation'
    out.mkdir(exist_ok=True)
    completed = out / 'report.json'
    if completed.exists():
        report = json.loads(completed.read_text())
        assert report['selection_sha256'] == signature
        return report
    exposure = out / 'exposure.json'
    if exposure.exists():
        assert json.loads(exposure.read_text())['selection_sha256'] == signature
    else:
        # Conservative exposure marker is durable before reading any holdout text.
        save_json(exposure, {'selection_sha256': signature, 'opened_at': time.time()})
    holdout = Path(plan['dataset']['path']) / 'topic_holdout.jsonl'
    assert digest(holdout) == plan['holdout_sha256'], 'Holdout changed'
    from .data import read_jsonl
    rows = read_jsonl(holdout)
    ids = [r['id'] for r in rows]
    from .improvement_inference import predict
    reports = {}
    for name, record in plan['bundles'].items():
        target = out / f'{name}_scores.npz'
        meta = target.with_suffix('.json')
        provenance = {'selection_sha256': signature, 'bundle_sha256': record['manifest_sha256']}
        values, seconds = [], []
        if target.exists():
            assert json.loads(meta.read_text()) == provenance
            cache = np.load(target, allow_pickle=False)
            assert cache['ids'].tolist() == ids[:len(cache['ids'])]
            values, seconds = cache['values'].tolist(), cache['chunk_seconds'].tolist()
        save_json(meta, provenance)
        for start in range(len(values), len(rows), 10):
            if time.time() >= deadline(root):
                return None
            began = time.perf_counter()
            predictions = predict(record['path'], [r['text'] for r in rows[start:start + 10]], device=device)
            scores = [r['probability_right'] for r in predictions]
            assert len(scores) == min(10, len(rows) - start) and np.isfinite(scores).all()
            values.extend(scores)
            seconds.append(time.perf_counter() - began)
            save_scores(target, values, ids[:len(values)], seconds)
            save_json(out / 'progress.json', {'bundle': name, 'completed': len(values), 'total': len(rows), 'updated_at': time.time()})
        p = np.asarray(values)
        y = np.array([r['label'] for r in rows])
        report = report_predictions(rows, p)
        report['default_decision'] = decision_summary(y, p)
        report['uncertainty'] = uncertainty(y, p)
        for field, key in [('source', 'sources'), ('topic', 'topics')]:
            for group, summary in report[key].items():
                mask = np.array([r[field] == group for r in rows])
                summary['default_decision'] = decision_summary(y[mask], p[mask])
        report['class_counts'] = {str(c): sum(r['label'] == c for r in rows) for c in [0, 1]}
        report['end_to_end_seconds'] = sum(seconds)
        report['timing_scope'] = 'Sequential batches of ten with model loading and integrity checks; not warm-server latency.'
        if record['decision_rules']:
            threshold = record['decision_rules']['gate_balanced_threshold']
            report['gate_fixed_balanced_decision'] = decision_summary(y, p, threshold)
        reports[name] = report
    result = {'selection_sha256': signature, 'dataset_sha256': plan['dataset']['manifest_sha256'],
              'holdout_sha256': plan['holdout_sha256'], 'results': reports, 'rows': len(rows),
              'completed_at': time.time(), 'scope': 'Sealed MITweet topic stress evaluation after model selection.',
              'limitations': ['Only three held-out topic groups; not a broad US political-text benchmark.',
                             'Human ideology facets mapped to binary labels; not direct overall-US human gold.',
                             'No author-disjoint guarantee; synthetic training can cover related policy themes.',
                             'Results cannot be used to select or retune these candidates.']}
    save_json(completed, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    freezing = sub.add_parser('freeze')
    freezing.add_argument('run', type=Path)
    freezing.add_argument('--compact-bundle', type=Path)
    freezing.add_argument('--allow-incomplete', action='store_true')
    freezing.add_argument('--reason')
    scoring = sub.add_parser('evaluate')
    scoring.add_argument('run', type=Path)
    scoring.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.run, args.compact_bundle, args.allow_incomplete, args.reason)
    else:
        evaluate(args.run, args.device)
