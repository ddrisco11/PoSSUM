"""Resumable, frozen expert comparison on development partitions only."""
import argparse
import datetime
import fcntl
import json
import time
from pathlib import Path

import joblib
import numpy as np

from .checkpoints import atomic_write, save_json, save_scores
from .improvement import PARTS, calibration, load_data, report_predictions, digest
from .option_a import MODELS
from .revisions import resolve, local_snapshot


EXPERT_NAMES = ('gliner', 'mistral', 'qwen', 'qwen_chat', 'mistral_human_lora_e1', 'mistral_human_lora_e2')


def expert_settings(name):
    if name in ['mistral_human_lora_e1', 'mistral_human_lora_e2']:
        return {'backbone': list(MODELS['mistral']), 'engine': 'mlx', 'prompt_format': 'plain',
                'adapter': f'adapters/mistral_human/epoch-{name[-1]}'}
    base = 'qwen' if name == 'qwen_chat' else name
    if base in MODELS:
        return {'backbone': list(MODELS[base]), 'engine': 'mlx',
                'prompt_format': 'chat_nonthinking' if name == 'qwen_chat' else 'plain'}
    if name != 'gliner':
        raise ValueError(f'Unknown expert: {name}')
    return {'backbone': ['fastino/gliner2.5-base-v1', resolve('fastino/gliner2.5-base-v1')], 'engine': 'gliner'}


def run(root, name):
    from .improvement import require_development_open
    require_development_open(root)
    lock = (root / 'expert-process.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parts = load_data(root)
    rows = sum([parts[p] for p in PARTS[1:]], [])
    ids = [r['id'] for r in rows]
    end = datetime.datetime.fromisoformat(json.loads((root / 'state.json').read_text())['deadline']).timestamp()
    out = root / 'experts'
    out.mkdir(exist_ok=True)
    target = out / f'{name}_raw.npz'
    meta = out / f'{name}.json'
    settings = expert_settings(name)
    backbone = settings['backbone']
    signature = {'manifest_sha256': json.loads((root / 'dataset.json').read_text())['manifest_sha256'],
                 'backbone': list(backbone), 'frozen': 'adapter' not in settings, 'prompt_version': 'qwen-chat-nonthinking-v1' if name == 'qwen_chat' else 'causal-v1' if settings['engine'] == 'mlx' else 'gliner-v1'}
    if 'adapter' in settings:
        adapter = root / settings['adapter']
        training = json.loads((adapter / 'training.json').read_text())
        assert training['complete'] and training['signature']['manifest_sha256'] == signature['manifest_sha256']
        assert digest(adapter / 'adapters.safetensors') == training['adapter_sha256']
        signature['adapter_sha256'] = training['adapter_sha256']
        signature['training_sha256'] = digest(adapter / 'training.json')
    values, elapsed = [], []
    if meta.exists():
        assert json.loads(meta.read_text())['signature'] == signature
    if target.exists():
        assert meta.exists(), 'Missing cache provenance'
        z = np.load(target, allow_pickle=False)
        assert z['ids'].tolist() == ids[:len(z['ids'])]
        values, elapsed = z['values'].tolist(), z['chunk_seconds'].tolist()
    save_json(meta, {'signature': signature, 'complete': False})
    if len(values) < len(rows):
        if time.time() >= end:
            return
        if settings['engine'] == 'mlx':
            from .components.causal import MLXCausal
            model = MLXCausal(local_snapshot(backbone[0], backbone[1]), prompt_format=settings['prompt_format'],
                              adapter_path=str(root / settings['adapter']) if 'adapter' in settings else None)
            model.model.eval()
        else:
            import torch
            from .components.gliner import GLiNER
            torch.set_num_threads(4)
            model = GLiNER()
        for start in range(len(values), len(rows), 10):
            if time.time() >= end:
                return
            began = time.perf_counter()
            predicted = model.score([r['text'] for r in rows[start:start + 10]])
            assert predicted.shape == (min(10, len(rows) - start), 2)
            assert np.isfinite(predicted).all()
            values.extend(predicted.tolist())
            elapsed.append(time.perf_counter() - began)
            save_scores(target, values, ids[:len(values)], elapsed)
            progress = {'expert': name, 'completed': len(values), 'total': len(rows),
                        'seconds': sum(elapsed), 'seconds_per_excerpt': sum(elapsed) / len(values), 'updated_at': time.time()}
            save_json(root / 'expert-progress.json', progress)
            if len(values) % 100 == 0 or len(values) == len(rows):
                print(json.dumps(progress), flush=True)
    values = np.asarray(values)
    raw = values[:, 1] - values[:, 0]
    cal = calibration(raw[:len(parts['calibration'])], parts['calibration'])
    probabilities = cal.predict_proba(raw[:, None])[:, 1]
    validation = report_predictions(parts['validation'], probabilities[-len(parts['validation']):])
    atomic_write(out / f'{name}_calibrator.joblib', lambda f: joblib.dump(cal, f))
    save_scores(out / f'{name}_scores.npz', probabilities, ids, elapsed)
    save_json(meta, {'signature': signature, 'complete': True, 'validation': validation,
                     'seconds': sum(elapsed), 'rows': len(rows)})
    print(name, json.dumps(validation['sources']['MITweet']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('expert', choices=EXPERT_NAMES)
    args = parser.parse_args()
    run(args.run, args.expert)
