"""Verify the frozen compact export on CPU using existing calibration inputs."""
import argparse
import datetime
import fcntl
import json
import sys
import time
from pathlib import Path

import numpy as np

from possum.checkpoints import save_json
from possum.export_validation import compare_scores
from possum.improvement import digest, load_data
from possum.improvement_inference import predict
from possum.predict_development import resolve_bundle


def verify(root):
    lock = (root / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = datetime.datetime.fromisoformat(json.loads((root / 'state.json').read_text())['deadline']).timestamp()
    if time.time() >= deadline:
        raise TimeoutError('Verification deadline reached')
    assert (root / 'selection-freeze.json').exists(), 'Requires a frozen candidate'
    bundle = resolve_bundle(root, 'compact')
    manifest = json.loads((bundle / 'model.json').read_text())
    assert len(manifest['components']) == 1 and next(iter(manifest['components'].values()))['kind'] == 'compact'
    rows = load_data(root)['calibration']
    selected = [rows[0], rows[-1], max(rows, key=lambda r: len(r['text']))]
    comparison = root / manifest['source_comparison']['path']
    assert digest(comparison / 'comparison.json') == manifest['source_comparison']['sha256']
    scores = np.load(comparison / f"{manifest['selected_name']}_scores.npz", allow_pickle=False)
    lookup = dict(zip(scores['ids'].tolist(), scores['values']))
    started = time.perf_counter()
    actual = [p['probability_right'] for p in predict(bundle, [r['text'] for r in selected], device='cpu')]
    seconds = time.perf_counter() - started
    error = compare_scores(actual, [lookup[r['id']] for r in selected])
    assert 'mlx.core' not in sys.modules, 'CPU compact check unexpectedly initialized MLX'
    return {'variant': 'compact', 'selected_name': manifest['selected_name'], 'manifest_sha256': digest(bundle / 'model.json'),
            'selection_sha256': digest(root / 'selection-freeze.json'),
            'dataset_sha256': manifest['dataset']['manifest_sha256'], 'device': 'cpu', 'examples': len(selected),
            'max_absolute_error_against_cached_mps_scores': error, 'passed': True, 'mlx_initialized': False,
            'seconds_including_loading_and_integrity_checks': seconds,
            'scope': 'Existing calibration passages on the same Mac and installed runtime. No holdout predictions or accuracy reevaluation.',
            'limitations': 'Confirms CPU inference for the compact encoder only; the quantized Mistral/Qwen exports still use MLX on Apple Silicon.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    save_json(args.output, result)
    print(json.dumps(result, indent=2))
