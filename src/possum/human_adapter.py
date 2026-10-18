"""Human-fit-only Mistral LoRA with durable optimizer recovery and epoch exports."""
import argparse
import datetime
import fcntl
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from .checkpoints import atomic_write, save_json
from .improvement import load_data, sample_weights, digest
from .option_a import MODELS, durable_file

RECIPE = {'model': 'mistral', 'regime': 'MITweet fit only', 'epochs': 2, 'seed': 42,
          'learning_rate': 1e-5, 'accumulation': 4, 'clip_norm': 1., 'layers': 4,
          'lora': {'rank': 8, 'scale': 16., 'dropout': 0., 'keys': ['self_attn.q_proj', 'self_attn.v_proj']},
          'loss': 'class-balanced binary left/right next-token cross entropy'}


def fit_rows(root):
    rows = [r for r in load_data(root)['fit'] if r['source'] == 'MITweet']
    assert rows and set(r['label'] for r in rows) == {0, 1}
    return rows


def update_indices(count, step, recipe=RECIPE):
    per_epoch = math.ceil(count / recipe['accumulation'])
    epoch, offset = divmod(step, per_epoch)
    order = np.random.default_rng(recipe['seed'] + epoch).permutation(count)
    return order[offset * recipe['accumulation']:(offset + 1) * recipe['accumulation']]


def train(root, target_epoch):
    from .improvement import require_development_open
    require_development_open(root)
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten, tree_map
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.trainer import grad_checkpoint
    from .components.causal import prompt, candidate_ids
    from .revisions import local_snapshot

    lock = (root / 'human-adapter-process.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = fit_rows(root)
    signature = {'manifest_sha256': json.loads((root / 'dataset.json').read_text())['manifest_sha256'],
                 'fit_ids_sha256': __import__('hashlib').sha256(json.dumps([r['id'] for r in rows]).encode()).hexdigest(),
                 'recipe': RECIPE, 'backbone': list(MODELS['mistral'])}
    out = root / 'adapters/mistral_human'
    out.mkdir(parents=True, exist_ok=True)
    plan = out / 'plan.json'
    if plan.exists():
        assert json.loads(plan.read_text()) == signature, 'Adapter plan changed'
    else:
        save_json(plan, signature)
    exported = out / f'epoch-{target_epoch}'
    if (exported / 'training.json').exists():
        meta = json.loads((exported / 'training.json').read_text())
        assert meta['signature'] == signature and meta['adapter_sha256'] == digest(exported / 'adapters.safetensors')
        return
    deadline = datetime.datetime.fromisoformat(json.loads((root / 'state.json').read_text())['deadline']).timestamp()
    if time.time() >= deadline:
        return
    mx.random.seed(RECIPE['seed'])
    mx.set_cache_limit(256 * 1024 ** 2)
    model, tokenizer = load(local_snapshot(*MODELS['mistral']))
    model.freeze()
    linear_to_lora_layers(model, RECIPE['layers'], RECIPE['lora'])
    grad_checkpoint(model.layers[-1])
    model.train()
    optimizer = optim.Adam(learning_rate=RECIPE['learning_rate'])
    weights = sample_weights(rows)
    encoded = []
    for row in rows:
        pairs = [candidate_ids(tokenizer, prompt(row['text']), c) for c in [' left', ' right']]
        assert all(len(b) == len(a) + 1 for a, b in pairs), 'Training requires single-token choices'
        assert len(pairs[0][0]) <= 256, 'No silent truncation'
        encoded.append((pairs[0][0], [b[-1] for _, b in pairs], row['label']))
    def loss(m, tokens, choices, label, weight):
        return nn.losses.cross_entropy(m(tokens)[:, -1, choices].astype(mx.float32), label).mean() * weight
    value_grad = nn.value_and_grad(model, loss)
    def step_once(step):
        indices = update_indices(len(rows), step)
        accumulated = None
        values = []
        for index in indices:
            tokens, choices, label = encoded[int(index)]
            value, gradients = value_grad(model, mx.array([tokens]), mx.array(choices), mx.array([label]), float(weights[index]))
            mx.eval(value, gradients)
            scalar = float(value)
            if not np.isfinite(scalar):
                raise ValueError('Nonfinite adapter loss')
            values.append(scalar)
            accumulated = gradients if accumulated is None else tree_map(lambda a, b: a + b, accumulated, gradients)
        gradients = tree_map(lambda x: x / len(indices), accumulated)
        gradients, norm = optim.clip_grad_norm(gradients, RECIPE['clip_norm'])
        mx.eval(gradients, norm)
        if not np.isfinite(float(norm)):
            raise ValueError('Nonfinite adapter gradient')
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state)
        mx.clear_cache()
        return float(np.mean(values))
    def save(step, seconds):
        folder = out / 'recovery' / f'step-{step:06d}'
        folder.mkdir(parents=True, exist_ok=True)
        durable_file(folder / 'adapters.safetensors', lambda p: mx.save_safetensors(str(p), dict(tree_flatten(model.trainable_parameters()))))
        durable_file(folder / 'optimizer.safetensors', lambda p: mx.save_safetensors(str(p), {k: mx.array(v) for k, v in tree_flatten(optimizer.state)}))
        save_json(folder / 'state.json', {'signature': signature, 'step': step, 'seconds': seconds})
        save_json(out / 'latest.json', {'folder': str(folder.relative_to(out))})
        for old in sorted((out / 'recovery').glob('step-*'))[:-2]:
            shutil.rmtree(old)
        return folder
    def restore(folder):
        state = json.loads((folder / 'state.json').read_text())
        assert state['signature'] == signature
        model.load_weights(str(folder / 'adapters.safetensors'), strict=False)
        optimizer.state = tree_unflatten(list(mx.load(str(folder / 'optimizer.safetensors')).items()))
        return state
    step, seconds = 0, 0.
    if (out / 'latest.json').exists():
        state = restore(out / json.loads((out / 'latest.json').read_text())['folder'])
        step, seconds = state['step'], state['seconds']
    target_steps = target_epoch * math.ceil(len(rows) / RECIPE['accumulation'])
    assert step <= target_steps, 'Cannot recreate an earlier epoch from a later checkpoint'
    # Verify optimizer reload on a real update before sustained training.
    if step == 0:
        began = time.perf_counter()
        step_once(0)
        seconds += time.perf_counter() - began
        step = 1
    if not (out / 'resume-check.json').exists():
        folder = save(step, seconds)
        step_once(step)
        expected = {k: np.array(v) for k, v in tree_flatten(model.trainable_parameters())}
        restore(folder)
        step_once(step)
        error = max(float(np.abs(np.asarray(v) - expected[k]).max()) for k, v in tree_flatten(model.trainable_parameters()))
        assert error < 1e-6, f'Adapter optimizer reload mismatch: {error}'
        restore(folder)
        save_json(out / 'resume-check.json', {'passed': True, 'max_parameter_difference': error,
                  'scope': 'Same next accumulated update after adapter and optimizer reload; deterministic order and zero dropout.'})
    assert json.loads((out / 'resume-check.json').read_text())['passed']
    losses = []
    try:
        while step < target_steps and time.time() < deadline:
            began = time.perf_counter()
            losses.append(step_once(step))
            seconds += time.perf_counter() - began
            step += 1
            if step % 25 == 0 or step == target_steps:
                save(step, seconds)
                progress = {'target_epoch': target_epoch, 'updates': step, 'target_updates': target_steps,
                            'fit_rows': len(rows), 'seconds': seconds, 'loss': float(np.mean(losses[-25:])),
                            'peak_mlx_gib': mx.get_peak_memory() / 1024 ** 3, 'updated_at': time.time()}
                save_json(root / 'human-adapter-progress.json', progress)
                print(json.dumps(progress), flush=True)
    finally:
        final = save(step, seconds)
    if step < target_steps:
        return
    exported.mkdir(exist_ok=True)
    atomic_write(exported / 'adapters.safetensors', lambda f: f.write((final / 'adapters.safetensors').read_bytes()))
    save_json(exported / 'adapter_config.json', {'model': MODELS['mistral'][0], 'fine_tune_type': 'lora',
              'num_layers': RECIPE['layers'], 'lora_parameters': RECIPE['lora']})
    save_json(exported / 'training.json', {'signature': signature, 'epoch': target_epoch, 'updates': step,
              'fit_rows': len(rows), 'seconds': seconds, 'adapter_sha256': digest(exported / 'adapters.safetensors'),
              'trainable_parameters': sum(v.size for _, v in tree_flatten(model.trainable_parameters())), 'complete': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--epoch', type=int, choices=[1, 2], required=True)
    args = parser.parse_args()
    train(args.run, args.epoch)
