"""Publish aggregate DeBERTa results only after both planned recipes complete."""
import argparse
import json
from pathlib import Path

import numpy as np

from possum.checkpoints import save_json
from possum.compact_training import DEBERTA_TRIALS
from possum.improvement import load_data
from possum.improvement_audit import paired_uncertainty


def report(root):
    parts = load_data(root)
    signature = json.loads((root / 'dataset.json').read_text())['manifest_sha256']
    registry = json.loads((root / 'compact/report.json').read_text())
    ids = [r['id'] for part in ['calibration', 'gate', 'validation'] for r in parts[part]]
    human = np.array([r['source'] == 'MITweet' for r in parts['validation']])
    y = np.array([r['label'] for r in parts['validation']])[human]

    def selected(name):
        folder = root / 'compact' / name
        complete = json.loads((folder / 'complete.json').read_text())
        best = json.loads((folder / 'best.json').read_text())
        assert complete['input_sha256'] == best['input_sha256'] == signature
        evaluation = json.loads((folder / best['folder'] / 'evaluation.json').read_text())
        assert evaluation['input_sha256'] == signature
        scores = np.load(folder / best['folder'] / 'scores.npz', allow_pickle=False)
        assert scores['ids'].tolist() == ids
        return best, evaluation, scores['values'][-len(parts['validation']):][human]

    baseline_best, baseline, baseline_scores = selected('bert_full_mixed')
    trials = {}
    for trial in DEBERTA_TRIALS:
        name = trial['name']
        best, evaluation, scores = selected(name)
        epochs = [registry[f'{name}_epoch{epoch}'] for epoch in range(1, trial['epochs'] + 1)]
        assert all(r['input_sha256'] == signature and r['trial'] == trial for r in epochs)
        assert evaluation['epoch'] == best['epoch'] == min(epochs, key=lambda r: r['validation']['sources']['MITweet']['log_loss'])['epoch']
        trials[name] = {'recipe': trial, 'selected_epoch': best['epoch'], 'selected_validation': evaluation['validation'],
                        'all_epochs': epochs, 'paired_vs_bert_mixed': paired_uncertainty(y, scores, baseline_scores)}
    return {'dataset_sha256': signature, 'model_revision': json.loads((root / 'deberta-plan.json').read_text())['model_revision'],
            'completed_recipes': len(trials), 'trials': trials,
            'bert_reference': {'selected_epoch': baseline_best['epoch'], 'validation': baseline['validation']},
            'selection_metric': 'MITweet validation log loss, same criterion for all epochs and recipes.',
            'training_precision': 'float32 parameters and AdamW moments; nonfinite gradient checks enabled.',
            'failed_attempt': 'An initial half-precision update produced nonfinite parameters; no completed epoch or evaluation. Restarted from the same pretrained revision and seed in float32.',
            'topic_holdout_scored': False,
            'limitations': ['Development validation reused for checkpoint and architecture selection; no selection-corrected inference.',
                           'One seed per DeBERTa recipe; BERT reference is seed 42.',
                           'Human validation is left-skewed and dominated by two topics; synthetic results are separate.',
                           'Human ideology facets are mapped labels, not fresh overall-US ideology adjudication.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = report(args.run)
    save_json(args.output, result)
    print(json.dumps({name: trial['selected_validation']['sources']['MITweet'] for name, trial in result['trials'].items()}, indent=2))
