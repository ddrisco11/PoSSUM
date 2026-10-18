"""Class-prior sensitivity and paired uncertainty for development comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

from .checkpoints import save_json
from .improvement import load_data, report_predictions
from .improvement_mixtures import binary_loss, diagnostics


def threshold_balanced_accuracy(y, p, threshold):
    return float(np.mean([((p[y == c] >= threshold) == c).mean() for c in [0, 1]]))


def choose_threshold(y, p):
    # Fixed grid and gate labels only; ties favor the conventional threshold.
    grid = np.linspace(.01, .99, 99)
    return float(min(grid, key=lambda t: (-threshold_balanced_accuracy(y, p, t), abs(t - .5))))


def shift_prior(p, old_prior, new_prior=.5):
    if not 0 < old_prior < 1 or not 0 < new_prior < 1:
        raise ValueError('Both priors must be strictly between zero and one.')
    return expit(logit(np.clip(p, 1e-12, 1 - 1e-12)) + logit(new_prior) - logit(old_prior))


def paired_uncertainty(y, candidate, baseline, draws=2000, seed=42):
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Both classes are required.')
    # Fixed-class-count bootstrap avoids replicates with no minority examples.
    rng = np.random.default_rng(seed)
    indices = np.concatenate([rng.choice(np.flatnonzero(y == c), size=(draws, int((y == c).sum())), replace=True) for c in [0, 1]], axis=1)
    loss_delta = binary_loss(y, candidate) - binary_loss(y, baseline)
    samples = loss_delta[indices].mean(axis=1)
    return {'candidate_minus_baseline_log_loss': float(loss_delta.mean()),
            'percentile_95_interval': np.quantile(samples, [.025, .975]).tolist(),
            'draws': draws, 'method': 'paired class-stratified row bootstrap',
            'limitations': 'Descriptive development interval, without selection correction or author clustering; not a significance test.'}


def run(root):
    parts = load_data(root)
    comparison = json.loads((root / 'mixtures/comparison.json').read_text())
    rows = parts['calibration'] + parts['gate'] + parts['validation']
    ids = [r['id'] for r in rows]
    ncal, ngate = len(parts['calibration']), len(parts['gate'])
    cal_human = [r for r in parts['calibration'] if r['source'] == 'MITweet']
    old_prior = np.mean([r['label'] for r in cal_human])
    gate_mask = np.array([r['source'] == 'MITweet' for r in parts['gate']])
    val_mask = np.array([r['source'] == 'MITweet' for r in parts['validation']])
    gy = np.array([r['label'] for r in parts['gate']])[gate_mask]
    vy = np.array([r['label'] for r in parts['validation']])[val_mask]
    all_scores = {}
    reports = {}
    for name in comparison['results']:
        z = np.load(root / 'mixtures' / f'{name}_scores.npz', allow_pickle=False)
        assert z['ids'].tolist() == ids
        p = z['values']; all_scores[name] = p[-len(parts['validation']):][val_mask]
        threshold = choose_threshold(gy, p[ncal:ncal + ngate][gate_mask])
        balanced = shift_prior(p[-len(parts['validation']):], old_prior)
        reports[name] = {
            'observed_prior': {'validation': comparison['results'][name]['validation']['sources']['MITweet'],
                               'diagnostics': diagnostics(parts['validation'], p[-len(parts['validation']):])},
            'equal_prior_sensitivity': {'validation': report_predictions(parts['validation'], balanced)['sources']['MITweet'],
                                        'diagnostics': diagnostics(parts['validation'], balanced)},
            'gate_selected_balanced_threshold': {'threshold': threshold,
                'validation_balanced_accuracy': threshold_balanced_accuracy(vy, all_scores[name], threshold),
                'validation_left_recall': float((all_scores[name][vy == 0] < threshold).mean()),
                'validation_right_recall': float((all_scores[name][vy == 1] >= threshold).mean())}}
    selected = comparison['selected']
    lexical = next(n for n in comparison['candidates'] if n.startswith('lexical_'))
    result = {'selected': selected, 'lexical_reference': lexical,
        'human_calibration_prior_right': float(old_prior), 'human_validation_counts': {'left': int((vy == 0).sum()), 'right': int((vy == 1).sum())},
        'models': reports, 'paired_uncertainty': paired_uncertainty(vy, all_scores[selected], all_scores[lexical]),
        'topic_holdout_scored': False, 'selection_changed': False,
        'interpretation': ['Equal-prior adjustment assumes class-conditional distributions remain fixed while prevalence changes.',
                           'This is a sensitivity analysis, not evidence that deployment prevalence is 50/50.',
                           'Gate thresholds use only human gate labels; they do not change calibrated probabilities.',
                           'Probability is not ideological intensity or a neutral/nonpolitical detector.']}
    save_json(root / 'mixtures/audit.json', result)
    print(json.dumps({'selected': selected, 'paired_uncertainty': result['paired_uncertainty']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run(args.run)
