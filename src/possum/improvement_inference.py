"""Portable local development bundles and sequential, reloadable prediction."""
import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.special import softmax

from .checkpoints import save_json
from .improvement import digest, load_data
from .option_a import MODELS
from .revisions import resolve


def release_memory():
    gc.collect()
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except ImportError:
        pass


def encoder_features(name, texts, device):
    from .components.bge import BGE
    from .components.bert import BERT
    encoder = (BGE if name == 'bge' else BERT)(device=device)
    lengths = [len(encoder.tokenizer.encode(t, add_special_tokens=True)) for t in texts]
    if max(lengths) > 256:
        raise ValueError('Input exceeds the validated 256-token encoder limit; segment it explicitly.')
    features = encoder.encode(texts)
    del encoder
    release_memory()
    return features


def component(bundle, spec, texts, device, embeddings):
    if spec['kind'] == 'baseline':
        saved = joblib.load(bundle / spec['model'])
        feature = saved['features']
        if feature == 'lexical':
            x = saved['vectorizer'].transform(texts)
        else:
            if feature not in embeddings:
                embeddings[feature] = encoder_features(feature, texts, device)
            x = embeddings[feature]
        raw = saved['head'].decision_function(x)
        return saved['calibrator'].predict_proba(raw[:, None])[:, 1]
    if spec['kind'] == 'compact':
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import PeftModel
        folder = bundle / spec['model']
        trial = spec['trial']
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        if trial['mode'] == 'lora':
            name = trial.get('model_id', 'google-bert/bert-base-uncased')
            base = AutoModelForSequenceClassification.from_pretrained(name, revision=resolve(name), num_labels=2,
                    attn_implementation='eager', local_files_only=True)
            model = PeftModel.from_pretrained(base, folder)
            del base
        else:
            model = AutoModelForSequenceClassification.from_pretrained(folder, attn_implementation='eager', local_files_only=True)
        model.to(device).eval(); model.requires_grad_(False)
        raw = []
        with torch.inference_mode():
            for start in range(0, len(texts), 8):
                batch = tokenizer(texts[start:start + 8], padding=True, truncation=False, return_tensors='pt').to(device)
                if batch['input_ids'].shape[1] > 256:
                    raise ValueError('Input exceeds the validated 256-token encoder limit; segment it explicitly.')
                z = model(**batch).logits.detach().cpu().numpy()
                raw.extend((z[:, 1] - z[:, 0]).tolist())
        del model
        release_memory()
        return joblib.load(folder / 'calibrator.joblib').predict_proba(np.asarray(raw)[:, None])[:, 1]
    name = spec['model']
    if name in MODELS:
        from huggingface_hub import snapshot_download
        from .components.causal import MLXCausal
        backbone = spec['backbone']
        model = MLXCausal(snapshot_download(backbone[0], revision=backbone[1], local_files_only=True))
        model.model.eval()
    else:
        from .components.gliner import GLiNER
        assert resolve(spec['backbone'][0]) == spec['backbone'][1], 'GLiNER revision differs from export'
        model = GLiNER()
    raw = model.score(texts)
    del model
    release_memory()
    return joblib.load(bundle / spec['calibrator']).predict_proba((raw[:, 1] - raw[:, 0])[:, None])[:, 1]


def predict(bundle, texts, device='mps'):
    bundle = Path(bundle)
    manifest = json.loads((bundle / 'model.json').read_text())
    for model_id, revision in manifest.get('encoder_revisions', {}).items():
        assert resolve(model_id) == revision, f'Encoder revision differs from export: {model_id}'
    for path, expected in manifest['artifact_hashes'].items():
        assert digest(bundle / path) == expected, f'Changed artifact: {path}'
    texts = [' '.join(t.replace('\ufeff', '').split()) for t in texts]
    if not texts or any(not t for t in texts):
        raise ValueError('Provide nonempty passages.')
    import torch
    torch.set_num_threads(4)
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; use --device cpu for compatible compact models.')
    embeddings = {}
    names = list(manifest['components'])
    probabilities = np.stack([component(bundle, spec, texts, device, embeddings) for spec in manifest['components'].values()], axis=1)
    selected = manifest['selection']
    if selected['kind'] == 'single':
        weights = np.ones((len(texts), 1))
    elif selected['kind'] == 'fixed':
        weights = np.broadcast_to(np.asarray(selected['weights']), probabilities.shape)
    else:
        saved = joblib.load(bundle / selected['model'])
        if 'bge' not in embeddings:
            embeddings['bge'] = encoder_features('bge', texts, device)
        x = (embeddings['bge'] - saved['mean']) / saved['std']
        state = saved['state']
        hidden = np.tanh(x @ state['0.weight'].T + state['0.bias'])
        weights = softmax(hidden @ state['2.weight'].T + state['2.bias'], axis=1)
    p = (weights * probabilities).sum(axis=1)
    return [{'probability_right': float(p[i]), 'probability_left': float(1 - p[i]),
             'label': int(p[i] >= .5), 'word_count': len(text.split()),
             'within_development_word_range': 12 <= len(text.split()) <= 150,
             'components': {name: {'probability_right': float(probabilities[i, j]), 'weight': float(weights[i, j])} for j, name in enumerate(names)},
             'interpretation': 'Binary class probability under the calibration distribution; not ideological intensity or a neutrality test.'}
            for i, text in enumerate(texts)]


def export(root):
    comparison = json.loads((root / 'mixtures/comparison.json').read_text())
    selected_name = comparison['selected']
    selected = dict(comparison['results'][selected_name]['spec'])
    entries = comparison['candidates']
    if selected['kind'] == 'single':
        name = list(entries)[selected['index']]
        entries = {name: entries[name]}
        selected = {'kind': 'single'}
    elif selected['kind'] == 'fixed':
        active = [(name, spec, weight) for (name, spec), weight in zip(entries.items(), selected['weights']) if weight > 1e-8]
        entries = {name: spec for name, spec, _ in active}
        weights = np.array([weight for _, _, weight in active]); weights /= weights.sum()
        selected['weights'] = weights.tolist()
    # Versions are immutable so interrupted copies cannot replace a verified bundle.
    bundle = root / 'exports' / f'{selected_name}-{time.time_ns()}'
    bundle.mkdir(parents=True)
    components = {}
    for name, original in entries.items():
        spec = dict(original)
        spec.pop('scores', None)
        if spec['kind'] == 'expert':
            source = root / 'experts' / f'{name}_calibrator.joblib'
            destination = bundle / 'components' / f'{name}_calibrator.joblib'
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(source, destination)
            spec['calibrator'] = str(destination.relative_to(bundle))
            spec['backbone'] = list(MODELS[name]) if name in MODELS else ['fastino/gliner2.5-base-v1', resolve('fastino/gliner2.5-base-v1')]
        else:
            source = root / spec['model']
            destination = bundle / 'components' / name
            destination.parent.mkdir(exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination, ignore=shutil.ignore_patterns('scores.npz', 'evaluation.json'))
            else:
                shutil.copy2(source, destination)
            spec['model'] = str(destination.relative_to(bundle))
        components[name] = spec
    if selected['kind'] == 'router':
        shutil.copy2(root / selected['model'], bundle / 'router.joblib')
        selected['model'] = 'router.joblib'
    hashes = {str(p.relative_to(bundle)): digest(p) for p in bundle.rglob('*') if p.is_file()}
    manifest = {'format_version': 1, 'status': 'development candidate; final topic holdout pending', 'selected_name': selected_name,
                'selection': selected, 'components': components, 'artifact_hashes': hashes, 'dataset': comparison['dataset'],
                'encoder_revisions': {name: resolve(name) for name in ['google-bert/bert-base-uncased', 'BAAI/bge-small-en-v1.5']},
                'validation': comparison['results'][selected_name]['validation'], 'preprocessing': 'remove BOM; collapse whitespace',
                'limitations': comparison['limitations']}
    save_json(bundle / 'model.json', manifest)
    parts = load_data(root)
    # Calibration examples are already used; parity checks never touch the topic holdout.
    checks = [parts['calibration'][0], parts['calibration'][-1], max(parts['calibration'], key=lambda r: len(r['text']))]
    actual = np.array([r['probability_right'] for r in predict(bundle, [r['text'] for r in checks])])
    z = np.load(root / 'mixtures' / f'{selected_name}_scores.npz', allow_pickle=False)
    lookup = {k: float(v) for k, v in zip(z['ids'].tolist(), z['values'])}
    expected = np.array([lookup[r['id']] for r in checks])
    error = float(np.max(np.abs(actual - expected)))
    assert error < 2e-4, f'Reloaded prediction mismatch: {error}'
    save_json(bundle / 'verification.json', {'passed': True, 'examples': len(checks), 'max_absolute_error': error,
                                           'evaluation_role': 'calibration parity only'})
    save_json(root / 'export.json', {'path': str(bundle.resolve()), 'selected': selected_name, 'verified': True})
    print(json.dumps({'export': str(bundle), 'max_absolute_error': error}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    make = sub.add_parser('export'); make.add_argument('run', type=Path)
    infer = sub.add_parser('predict'); infer.add_argument('bundle', type=Path); infer.add_argument('--text', required=True)
    infer.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    args = parser.parse_args()
    if args.command == 'export':
        export(args.run)
    else:
        print(json.dumps(predict(args.bundle, [args.text], args.device), indent=2))
