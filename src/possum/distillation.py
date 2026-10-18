"""Fit-only frozen-teacher targets and reproducible compact distillation plans."""
import argparse
import datetime
import fcntl
import json
import time
from pathlib import Path

import numpy as np

from .checkpoints import save_json, save_scores
from .improvement import digest, load_data
from .option_a import MODELS


def plan(root):
    signature = json.loads((root / 'dataset.json').read_text())['manifest_sha256']
    path = root / 'distillation-plan.json'
    if path.exists():
        result = json.loads(path.read_text())
        assert result['input_sha256'] == signature
        return result
    experts = {name: json.loads((root / 'experts' / f'{name}.json').read_text()) for name in ['mistral', 'qwen']}
    assert all(item['complete'] and item['signature']['manifest_sha256'] == signature for item in experts.values())
    teacher = min(experts, key=lambda n: experts[n]['validation']['sources']['MITweet']['log_loss'])
    compact = json.loads((root / 'replication-plan.json').read_text())
    assert compact['input_sha256'] == signature
    student = compact['base_trial'].get('model_id', 'google-bert/bert-base-uncased')
    result = {'input_sha256': signature, 'teacher': teacher, 'teacher_signature': experts[teacher]['signature'],
              'selection': 'Teacher by human validation log loss among frozen Mistral/Qwen; student backbone from initial compact recipe comparison.',
              'target_role': 'Model-generated soft logits on the fit partition only, never new human annotations.',
              'trials': [dict(name=f'distilled_{teacher}_alpha{alpha:g}', model_id=student, mode='full', regime='mixed',
                             lr=2e-5, epochs=4, seed=42, distillation_alpha=alpha, distillation_temperature=2.,
                             teacher=teacher) for alpha in [.25, .5]]}
    save_json(path, result)
    return result


def load_targets(root, rows, expected_teacher):
    metadata = json.loads((root / 'distillation/teacher.json').read_text())
    signature = json.loads((root / 'dataset.json').read_text())['manifest_sha256']
    assert metadata['complete'] and metadata['input_sha256'] == signature
    assert metadata['teacher'] == expected_teacher and metadata['role'] == 'fit'
    target = root / 'distillation/teacher_raw.npz'
    assert digest(target) == metadata['scores_sha256'], 'Teacher cache changed'
    z = np.load(target, allow_pickle=False)
    assert z['ids'].tolist() == [r['id'] for r in rows], 'Teacher targets must match fit rows exactly'
    values = z['values']
    assert values.shape == (len(rows), 2) and np.isfinite(values).all()
    return values.astype(np.float32), metadata['scores_sha256']


def training_loss(logits, labels, weights, teacher_logits=None, alpha=0., temperature=2.):
    import torch
    if not 0 <= alpha <= 1 or temperature <= 0:
        raise ValueError('Invalid distillation hyperparameters')
    hard = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    if teacher_logits is None:
        if alpha:
            raise ValueError('Teacher logits required')
        return (hard * weights).mean()
    if teacher_logits.shape != logits.shape:
        raise ValueError('Teacher/student shape mismatch')
    target = (teacher_logits.detach() / temperature).softmax(dim=-1)
    soft = torch.nn.functional.kl_div((logits / temperature).log_softmax(dim=-1), target,
                                     reduction='none').sum(dim=-1) * temperature ** 2
    return (((1 - alpha) * hard + alpha * soft) * weights).mean()


def score_fit(root):
    from .improvement import require_development_open
    require_development_open(root)
    lock = (root / 'teacher-process.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = plan(root)
    rows = load_data(root)['fit']
    ids = [r['id'] for r in rows]
    out = root / 'distillation'
    out.mkdir(exist_ok=True)
    meta = out / 'teacher.json'
    target = out / 'teacher_raw.npz'
    signature = {'input_sha256': spec['input_sha256'], 'teacher': spec['teacher'],
                 'teacher_signature': spec['teacher_signature'], 'role': 'fit'}
    if meta.exists():
        previous = json.loads(meta.read_text())
        assert all(previous[k] == v for k, v in signature.items())
        if previous.get('complete'):
            load_targets(root, rows, spec['teacher'])
            return
    values, elapsed = [], []
    if target.exists():
        assert meta.exists(), 'Missing teacher provenance'
        z = np.load(target, allow_pickle=False)
        assert z['ids'].tolist() == ids[:len(z['ids'])]
        values, elapsed = z['values'].tolist(), z['chunk_seconds'].tolist()
    save_json(meta, dict(signature, complete=False))
    deadline = datetime.datetime.fromisoformat(json.loads((root / 'state.json').read_text())['deadline']).timestamp()
    if time.time() >= deadline:
        return
    from .components.causal import MLXCausal
    from .revisions import local_snapshot
    assert list(MODELS[spec['teacher']]) == spec['teacher_signature']['backbone'], 'Teacher revision changed'
    model = MLXCausal(local_snapshot(*MODELS[spec['teacher']]))
    model.model.eval()
    for start in range(len(values), len(rows), 10):
        if time.time() >= deadline:
            return
        began = time.perf_counter()
        predicted = model.score([r['text'] for r in rows[start:start + 10]])
        assert predicted.shape == (min(10, len(rows) - start), 2) and np.isfinite(predicted).all()
        values.extend(predicted.tolist())
        elapsed.append(time.perf_counter() - began)
        save_scores(target, values, ids[:len(values)], elapsed)
        progress = {'teacher': spec['teacher'], 'completed': len(values), 'total': len(rows),
                    'seconds': sum(elapsed), 'updated_at': time.time()}
        save_json(out / 'progress.json', progress)
        if len(values) % 100 == 0 or len(values) == len(rows):
            print(json.dumps(progress), flush=True)
    save_json(meta, dict(signature, complete=True, scores_sha256=digest(target), rows=len(rows), seconds=sum(elapsed)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    score_fit(args.run)
