"""Compare calibrated experts and small mixtures without accessing the sealed holdout."""
import argparse
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

from .checkpoints import atomic_write, save_json, save_scores
from .improvement import PARTS, load_data, report_predictions


def binary_loss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def diagnostics(rows, probabilities):
    mask = np.array([r['source'] == 'MITweet' for r in rows])
    y = np.array([r['label'] for r in rows])[mask]
    p = np.asarray(probabilities)[mask]
    losses = binary_loss(y, p)
    return {'human_roc_auc': float(roc_auc_score(y, p)),
            'human_class_balanced_log_loss': float(np.mean([losses[y == c].mean() for c in [0, 1]])),
            'human_left_recall': float((p[y == 0] < .5).mean()),
            'human_right_recall': float((p[y == 1] >= .5).mean()),
            'human_counts': {'left': int((y == 0).sum()), 'right': int((y == 1).sum())}}


def predictor_groups(entries):
    """Calibration variants share one underlying predictor's penalty mass."""
    groups = {}
    for index, spec in enumerate(entries.values()):
        key = (spec['kind'], spec.get('model', 'constant'))
        groups.setdefault(key, []).append(index)
    matrix = np.zeros((len(groups), len(entries)))
    for row, indices in enumerate(groups.values()):
        matrix[row, indices] = 1.
    return matrix


def fit_fixed(gate, labels, penalty, groups=None):
    if groups is None:
        groups = np.eye(gate.shape[1])
    target = np.full(len(groups), 1 / len(groups))
    initial = (groups / groups.sum(axis=1, keepdims=True)).T @ target
    def objective(weights):
        return binary_loss(labels, gate @ weights).mean() + penalty * np.square(groups @ weights - target).sum()
    result = minimize(objective, initial, method='SLSQP', bounds=[(0., 1.)] * gate.shape[1],
                      constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
                      options={'maxiter': 1000, 'ftol': 1e-10})
    assert result.success, result.message
    weights = np.maximum(0, result.x)
    return weights / weights.sum()


def candidates(root, ids, calibration_prior):
    entries = {}
    baselines = json.loads((root / 'baselines/report.json').read_text())
    for family in ['lexical', 'bge', 'bert']:
        name = min([n for n in baselines if n.startswith(family + '_')],
                   key=lambda n: baselines[n]['validation']['sources']['MITweet']['log_loss'])
        entries[name] = {'kind': 'baseline', 'model': f'baselines/{name}.joblib', 'scores': f'baselines/{name}_scores.npz'}
    for best_path in sorted((root / 'compact').glob('*/best.json')):
        best = json.loads(best_path.read_text())
        folder = best_path.parent / best['folder']
        entries[best['trial']['name']] = {'kind': 'compact', 'model': str(folder.relative_to(root)),
                                         'scores': str((folder / 'scores.npz').relative_to(root)), 'trial': best['trial']}
    from .improvement_experts import EXPERT_NAMES
    for name in EXPERT_NAMES:
        meta_path = root / 'experts' / f'{name}.json'
        if meta_path.exists() and json.loads(meta_path.read_text()).get('complete'):
            entries[name] = {'kind': 'expert', 'model': name, 'scores': f'experts/{name}_scores.npz'}
    entries['constant_calibration_prior'] = {'kind': 'constant', 'probability': float(calibration_prior)}
    columns = []
    for spec in entries.values():
        if spec['kind'] == 'constant':
            columns.append(np.full(len(ids), spec['probability']))
            continue
        z = np.load(root / spec['scores'], allow_pickle=False)
        assert z['ids'].tolist() == ids, 'Prediction order differs'
        p = z['values']
        assert p.shape == (len(ids),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        columns.append(p)
    return entries, np.stack(columns, axis=1)


def run(root):
    from .improvement import require_development_open
    require_development_open(root)
    import torch
    torch.set_num_threads(4)
    parts = load_data(root)
    # Preserve each completed comparison before regenerating current selections.
    previous = root / 'mixtures'
    if (previous / 'comparison.json').exists():
        history = root / 'mixture-history' / str(time.time_ns())
        shutil.copytree(previous, history)
    rows = sum([parts[p] for p in PARTS[1:]], [])
    ids = [r['id'] for r in rows]
    calibration_prior = np.mean([r['label'] for r in parts['calibration'] if r['source'] == 'MITweet'])
    entries, probabilities = candidates(root, ids, calibration_prior)
    from .calibration_cv import augment
    entries, probabilities = augment(root, parts, entries, probabilities)
    names = list(entries)
    ncal, ngate = len(parts['calibration']), len(parts['gate'])
    gs = slice(ncal, ncal + ngate)
    vs = slice(ncal + ngate, len(rows))
    human = np.array([r['source'] == 'MITweet' for r in parts['gate']])
    y = np.array([r['label'] for r in parts['gate']])[human]
    gate = probabilities[gs][human]
    out = root / 'mixtures'
    out.mkdir(exist_ok=True)
    registry = {}

    def record(name, predicted, spec):
        report = report_predictions(parts['validation'], predicted[vs])
        registry[name] = {'validation': report, 'diagnostics': diagnostics(parts['validation'], predicted[vs]), 'spec': spec}
        save_scores(out / f'{name}_scores.npz', predicted, ids, [])
        print(name, json.dumps(report['sources']['MITweet']), flush=True)

    for i, name in enumerate(names):
        record(name, probabilities[:, i], {'kind': 'single', 'index': i})
    record('equal_average', probabilities.mean(axis=1), {'kind': 'fixed', 'weights': [1 / len(names)] * len(names)})
    for penalty in [0., .01, .1]:
        weights = fit_fixed(gate, y, penalty)
        record(f'fixed_{penalty:g}', probabilities @ weights, {'kind': 'fixed', 'weights': weights.tolist(), 'penalty': penalty})
    groups = predictor_groups(entries)
    for penalty in [.01, .1]:
        weights = fit_fixed(gate, y, penalty, groups)
        record(f'fixed_grouped_{penalty:g}', probabilities @ weights,
               {'kind': 'fixed', 'weights': weights.tolist(), 'penalty': penalty,
                'regularization': 'Equal total mass per underlying predictor; calibration variants share a group.',
                'groups': [[names[i] for i in np.flatnonzero(row)] for row in groups]})
    # A small, predefined family of two-predictor mixtures checks whether the
    # full ensemble's deployment cost is necessary. Within each family the
    # original-calibration representative is chosen using gate labels only.
    from itertools import combinations
    pool = []
    for family in ['baseline', 'compact']:
        eligible = [i for i, name in enumerate(names) if entries[name]['kind'] == family
                    and 'calibration_override' not in entries[name]
                    and (family != 'baseline' or name.startswith('lexical_'))]
        if eligible:
            pool.append(min(eligible, key=lambda i: binary_loss(y, gate[:, i]).mean()))
    pool += [names.index(name) for name in ['mistral', 'qwen', 'qwen_chat'] if name in entries]
    for left, right in combinations(pool, 2):
        active = [left, right]
        weights = np.zeros(len(names))
        weights[active] = fit_fixed(gate[:, active], y, 0.)
        record(f'pair_{names[left]}__{names[right]}', probabilities @ weights,
               {'kind': 'fixed', 'weights': weights.tolist(), 'penalty': 0.,
                'family': 'two-predictor cost comparison',
                'representative_selection': 'Original calibrators; lexical and compact representatives by human gate log loss.'})
    z = np.load(root / 'bge_features.npz', allow_pickle=False)
    lookup = {k: i for i, k in enumerate(z['ids'].tolist())}
    x = z['values'][[lookup[i] for i in ids]].astype(np.float32)
    mean, std = x[gs][human].mean(axis=0), x[gs][human].std(axis=0).clip(.02)
    xx = (x - mean) / std
    for penalty in [.01, .1]:
        torch.manual_seed(42)
        router = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 8), torch.nn.Tanh(), torch.nn.Linear(8, len(names)))
        optimizer = torch.optim.AdamW(router.parameters(), lr=.005, weight_decay=.01)
        xt = torch.tensor(xx[gs][human]); pt = torch.tensor(gate, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
        for _ in range(200):
            optimizer.zero_grad()
            logits = router(xt)
            pred = (logits.softmax(dim=1) * pt).sum(dim=1).clamp(1e-7, 1 - 1e-7)
            loss = torch.nn.functional.binary_cross_entropy(pred, yt) + penalty * logits.square().mean()
            loss.backward(); optimizer.step()
        with torch.no_grad():
            weights = router(torch.tensor(xx)).softmax(dim=1).numpy()
        payload = {'layers': [x.shape[1], 8, len(names)], 'state': {k: v.detach().numpy() for k, v in router.state_dict().items()}, 'mean': mean, 'std': std}
        name = f'router_{penalty:g}'
        atomic_write(out / f'{name}.joblib', lambda f: joblib.dump(payload, f))
        record(name, (weights * probabilities).sum(axis=1), {'kind': 'router', 'model': f'mixtures/{name}.joblib',
                'penalty': penalty, 'validation_mean_weights': weights[vs].mean(axis=0).tolist(),
                'self_branch': 'BGE logistic classifier fitted only on fit partition'})
    selected = min(registry, key=lambda n: registry[n]['validation']['sources']['MITweet']['log_loss'])
    signature = json.loads((root / 'dataset.json').read_text())
    report = {'candidates': entries, 'results': registry, 'selected': selected, 'selection_metric': 'MITweet validation log_loss',
              'dataset': signature, 'topic_holdout_scored': False, 'fit_roles': {'calibrators': 'human calibration',
              'mixture_weights': 'human gate', 'selection': 'human validation'},
              'limitations': ['Development selection is not a final test.', 'Human labels map multiple ideology facets to one binary axis.',
                              'Human validation is strongly left-skewed.', 'Author-disjointness is unavailable.']}
    save_json(out / 'comparison.json', report)
    from .improvement_audit import run as audit
    audit(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run(args.run)
