"""Reload an adapted ensemble and classify one passage, sequentially on Apple Silicon."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax


def predict(export, text):
    import torch
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_lm import load
    from possum.option_a import bert_model, router_weights
    from possum.components.causal import MLXCausal
    from possum.components.gliner import GLiNER
    from possum.components.bge import BGE
    import joblib

    config = json.loads((export / 'config.json').read_text())
    selected = config['selected']
    names = config['experts'][:2] if selected == 'compact_equal' else config['experts']
    probabilities = []
    for name in names:
        if name == 'bert':
            model, tok = bert_model(export / config['bert_adapter'])
            model.eval()
            encoded = tok(text, return_tensors='pt').to('mps')
            if encoded['input_ids'].shape[1] > 256:
                raise ValueError('Passage exceeds the BERT token limit')
            with torch.inference_mode():
                scores = model(**encoded).logits.cpu().numpy()
            del encoded, tok
        elif name == 'gliner':
            torch.set_num_threads(4)
            model = GLiNER()
            scores = model.score([text])
        else:
            repo, revision = config['backbones'][name]
            model = MLXCausal.__new__(MLXCausal)
            adapter = str(export / config['mistral_adapter']) if name == 'mistral' else None
            model.model, model.tokenizer = load(snapshot_download(repo, revision=revision, local_files_only=True), adapter_path=adapter)
            model.model.eval()
            scores = model.score([text])
        probabilities.append(float(softmax(scores / config['temperatures'][name], axis=1)[0, 1]))
        del model
        gc.collect()
        torch.mps.empty_cache()
        mx.clear_cache()
    weights = np.ones(len(names)) / len(names)
    if config['router_mode']:
        mode = config['router_mode']
        encoder = BGE(device='mps')
        embedding = encoder.encode([text])
        del encoder
        router = torch.nn.Linear(384, 5 if mode == 'router_self' else 4)
        router.load_state_dict(torch.load(export / f'{selected}.pt', weights_only=True))
        router.eval()
        with torch.inference_mode():
            weights = router_weights(router, torch.tensor(embedding, dtype=torch.float32), mode).numpy()[0]
        if mode == 'router_self':
            own = joblib.load(export / 'self_head.joblib')
            raw = np.log(np.clip(own.predict_proba(embedding), 1e-7, 1))
            probabilities.append(float(softmax(raw / config['self_temperature'], axis=1)[0, 1]))
            names = [*names, 'self']
    right = float(np.dot(weights, probabilities))
    return dict(probabilities={'left': 1-right, 'right': right}, label=int(right >= .5), selected=selected,
                component_probabilities=dict(zip(names, probabilities)), weights=dict(zip(names, weights.tolist())),
                status=config['status'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--text', required=True)
    args = parser.parse_args()
    if not 20 <= len(args.text.split()) <= 150:
        parser.error('Expected approximately 20–150 words of English political text')
    print(json.dumps(predict(args.export.resolve(), args.text), indent=2))
