"""End-to-end inference timing and batch parity on existing calibration passages."""
import argparse
import datetime
import fcntl
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .checkpoints import save_json
from .improvement import load_data


def sample_rows(rows):
    cells = defaultdict(list)
    for row in rows:
        cells[(row['source'], row['label'])].append(row)
    selected = []
    for key in sorted(cells):
        ordered = sorted(cells[key], key=lambda r: (len(r['text'].split()), r['id']))
        for index in sorted({0, len(ordered) // 2, len(ordered) - 1}):
            selected.append(ordered[index])
    assert len({r['id'] for r in selected}) == len(selected), 'Duplicate calibration identities'
    return selected


def time_requests(bundle, rows, device, deadline, predictor):
    texts = [r['text'] for r in rows]
    single_scores, seconds = [], []
    for text in texts:
        if time.time() >= deadline:
            raise TimeoutError('Inference benchmark deadline reached')
        began = time.perf_counter()
        prediction = predictor(bundle, [text], device=device)
        seconds.append(time.perf_counter() - began)
        assert len(prediction) == 1
        single_scores.append(prediction[0]['probability_right'])
    if time.time() >= deadline:
        raise TimeoutError('Inference benchmark deadline reached')
    began = time.perf_counter()
    batch = predictor(bundle, texts, device=device)
    batch_seconds = time.perf_counter() - began
    batch_scores = np.array([r['probability_right'] for r in batch])
    assert batch_scores.shape == (len(rows),) and np.isfinite(batch_scores).all()
    assert ((batch_scores >= 0) & (batch_scores <= 1)).all()
    error = float(np.max(np.abs(batch_scores - single_scores)))
    assert error < 2e-4, f'Batch/single prediction mismatch: {error}'
    repeated = np.array(seconds[1:] or seconds)
    return {'examples': len(rows), 'first_single_request_seconds': seconds[0],
            'subsequent_single_request_seconds': {'median': float(np.median(repeated)),
                'p90': float(np.quantile(repeated, .9)), 'min': float(repeated.min()), 'max': float(repeated.max())},
            'all_single_requests_seconds': seconds, 'batch_seconds': batch_seconds,
            'batch_seconds_per_passage': batch_seconds / len(rows),
            'batch_single_max_absolute_error': error, 'batch_single_parity_passed': True}


def run(root, bundles, device='mps'):
    lock = (root / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = datetime.datetime.fromisoformat(json.loads((root / 'state.json').read_text())['deadline']).timestamp()
    if time.time() >= deadline:
        raise TimeoutError('No new benchmark after the deadline')
    rows = sample_rows(load_data(root)['calibration'])
    signature = json.loads((root / 'dataset.json').read_text())['manifest_sha256']
    plan = {'dataset_sha256': signature, 'sample_ids_sha256': hashlib.sha256(json.dumps([r['id'] for r in rows]).encode()).hexdigest(),
            'sampling': 'Shortest, median-length and longest calibration passage per source/class cell.',
            'device': device, 'word_counts': {'min': min(len(r['text'].split()) for r in rows),
                                           'max': max(len(r['text'].split()) for r in rows)},
            'topic_holdout_scored': False}
    from .improvement_inference import predict
    from .final_evaluation import bundle_record
    results = []
    out = root / 'inference-benchmarks'
    out.mkdir(exist_ok=True)
    for bundle in bundles:
        bundle = Path(bundle)
        record = bundle_record(root, bundle)
        manifest = json.loads((bundle / 'model.json').read_text())
        bundle_sha = record['manifest_sha256']
        result_path = out / f'{bundle_sha[:16]}-{device}.json'
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            assert previous['plan'] == plan and previous['bundle_manifest_sha256'] == bundle_sha
            results.append(previous)
            continue
        measurements = time_requests(bundle, rows, device, deadline, predict)
        result = {'plan': plan, 'bundle_manifest_sha256': bundle_sha, 'bundle_name': bundle.name,
                  'selected_name': manifest['selected_name'], 'validation': manifest['validation']['sources'],
                  'measurements': measurements,
                  'scope': ['Each request uses the public reloadable predictor, including artifact hashing, model loading and cleanup.',
                            'First call includes lazy imports; later calls may benefit from filesystem/runtime caches.',
                            'Models are reloaded for each request; this is not a persistent warm-server benchmark.',
                            'Calibration examples only; latency measurements are not new accuracy evaluation.']}
        save_json(result_path, result)
        results.append(result)
        print(json.dumps({'bundle': bundle.name, 'measurements': measurements}), flush=True)
    save_json(out / 'report.json', {'results': results, 'completed_at': time.time()})
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--bundle', type=Path, action='append', required=True)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    args = parser.parse_args()
    run(args.run, args.bundle, args.device)
