"""Check frozen bundle relocation in fresh processes using calibration inputs."""
import argparse
import datetime
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .checkpoints import save_json
from .final_evaluation import bundle_record
from .improvement import digest, load_data

CHILD = '''import json,sys
from possum.improvement_inference import predict
result=predict(sys.argv[1],json.load(sys.stdin),device=sys.argv[2])
print(json.dumps([r['probability_right'] for r in result]))
'''


def compare_scores(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not len(actual):
        raise ValueError('Relocated prediction count differs')
    if any(not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all() for values in [actual, expected]):
        raise ValueError('Invalid relocated probability')
    error = float(np.max(np.abs(actual - expected)))
    if error >= 2e-4:
        raise ValueError(f'Relocated predictions differ: {error}')
    return error


def run(root, device='mps'):
    root = Path(root).absolute()
    lock = (root / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path = root / 'selection-freeze.json'
    plan = json.loads(plan_path.read_text())
    cutoff = datetime.datetime.fromisoformat(json.loads((root / 'state.json').read_text())['deadline']).timestamp()
    if time.time() >= cutoff:
        raise TimeoutError('Export validation deadline reached')
    parts = load_data(root)
    rows = [parts['calibration'][0], parts['calibration'][-1], max(parts['calibration'], key=lambda r: len(r['text']))]
    records = {}
    for name, record in plan['bundles'].items():
        if time.time() >= cutoff:
            raise TimeoutError('Export validation deadline reached')
        original = Path(record['path'])
        verified = bundle_record(root, original)
        assert verified['manifest_sha256'] == record['manifest_sha256']
        manifest = json.loads((original / 'model.json').read_text())
        comparison = root / manifest['source_comparison']['path']
        assert digest(comparison / 'comparison.json') == manifest['source_comparison']['sha256']
        scores = np.load(comparison / f"{manifest['selected_name']}_scores.npz", allow_pickle=False)
        lookup = dict(zip(scores['ids'].tolist(), scores['values']))
        expected = [float(lookup[r['id']]) for r in rows]
        with tempfile.TemporaryDirectory(prefix='relocated-', dir=root) as temporary:
            relocated = Path(temporary) / 'bundle'
            shutil.copytree(original, relocated)
            assert bundle_record(root, relocated)['manifest_sha256'] == record['manifest_sha256']
            env = dict(os.environ, HF_HUB_OFFLINE='1')
            if 'HF_HOME' in env:
                env['HF_HOME'] = str(Path(env['HF_HOME']).absolute())
            began = time.perf_counter()
            child = subprocess.run([sys.executable, '-c', CHILD, str(relocated), device],
                input=json.dumps([r['text'] for r in rows]), text=True, capture_output=True,
                cwd=temporary, env=env, timeout=max(1, cutoff-time.time()), check=True)
            error = compare_scores(json.loads(child.stdout), expected)
            records[name] = {'manifest_sha256': record['manifest_sha256'], 'examples': len(rows),
                             'max_absolute_error': error, 'passed': True,
                             'fresh_process_seconds': time.perf_counter()-began}
        # The originals must remain byte-identical after relocation checks.
        assert bundle_record(root, original)['manifest_sha256'] == record['manifest_sha256']
        print(json.dumps({'variant': name, **records[name]}), flush=True)
    result = {'selection_sha256': digest(plan_path), 'dataset_sha256': plan['dataset']['manifest_sha256'],
              'results': records, 'device': device,
              'scope': 'Copied bundle, fresh Python process, different working directory; three existing calibration inputs per variant. No new accuracy evaluation or holdout predictions.',
              'limitations': ['Uses the existing installed editable package and runtime.',
                             'Causal models require the separately cached pinned base weights; these were not copied.',
                             'This does not establish a clean-machine installation or cross-platform compatibility.'],
              'completed_at': time.time()}
    save_json(root / 'export-portability.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    args = parser.parse_args()
    run(args.run, args.device)
