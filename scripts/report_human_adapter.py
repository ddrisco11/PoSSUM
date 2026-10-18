"""Summarize both human-only adapter epochs and the matched mixture comparison."""
import argparse
import json
from pathlib import Path

import numpy as np

from possum.checkpoints import save_json
from possum.improvement import digest, load_data
from possum.improvement_audit import paired_uncertainty


def report(root):
    parts = load_data(root)
    comparison_path = root / 'mixtures/comparison.json'
    comparison = json.loads(comparison_path.read_text())
    selected = comparison['selected']
    rows = parts['calibration'] + parts['gate'] + parts['validation']
    mask = np.array([r['source'] == 'MITweet' for r in parts['validation']])
    labels = np.array([r['label'] for r in parts['validation']])[mask]

    def scores(name):
        values = np.load(root / 'mixtures' / f'{name}_scores.npz', allow_pickle=False)
        assert values['ids'].tolist() == [r['id'] for r in rows]
        return values['values'][-len(parts['validation']):][mask]

    epochs = {}
    for epoch in [1, 2]:
        name = f'mistral_human_lora_e{epoch}'
        training = json.loads((root / f'adapters/mistral_human/epoch-{epoch}/training.json').read_text())
        assert training['adapter_sha256'] == digest(root / f'adapters/mistral_human/epoch-{epoch}/adapters.safetensors')
        epochs[str(epoch)] = {'training': training, 'evaluation': comparison['results'][name]}
    pointer = json.loads((root / 'export.json').read_text())
    bundle = Path(pointer['path'])
    manifest = json.loads((bundle / 'model.json').read_text())
    assert manifest['selected_name'] == selected
    weights = dict(zip(comparison['candidates'], comparison['results'][selected]['spec']['weights']))
    result = {
        'dataset_sha256': comparison['dataset']['manifest_sha256'],
        'comparison_sha256': digest(comparison_path),
        'completed_epochs': 2, 'planned_epochs': 2,
        'epochs': epochs,
        'frozen_mistral': comparison['results']['mistral'],
        'selected': selected,
        'selected_evaluation': comparison['results'][selected],
        'selected_weights_above_numerical_tolerance': {k: v for k, v in weights.items() if v > 1e-8},
        'selection_metric': comparison['selection_metric'],
        'paired_human_validation': {
            'epoch2_minus_frozen': paired_uncertainty(labels, scores('mistral_human_lora_e2'), scores('mistral')),
            'epoch2_minus_epoch1': paired_uncertainty(labels, scores('mistral_human_lora_e2'), scores('mistral_human_lora_e1')),
            'mixture_minus_epoch2': paired_uncertainty(labels, scores(selected), scores('mistral_human_lora_e2')),
        },
        'export': {'manifest_sha256': digest(bundle / 'model.json'),
                   'verification': json.loads((bundle / 'verification.json').read_text())},
        'topic_holdout_scored': False,
        'limitations': ['Exploratory development selection on 408 human validation examples; no selection correction.',
                       'Human validation contains 366 left and 42 right examples; 83.8% concern two topics.',
                       'MITweet human facets are mapped to binary ideology, not fresh adjudicated overall U.S. ideology labels.',
                       'Synthetic loss is reported separately; better human loss does not imply universal transfer.',
                       'Calibration and mixture fitting use disjoint calibration and gate partitions, respectively.'],
    }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = report(args.run)
    save_json(args.output, result)
    print(json.dumps(result['paired_human_validation'], indent=2))
