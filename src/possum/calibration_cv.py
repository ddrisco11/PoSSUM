"""Choose calibration within calibration data, preserving gate and validation roles."""
from dataclasses import dataclass
import hashlib
import json

import joblib
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .checkpoints import atomic_write, save_json, save_scores
from .improvement import digest, report_predictions

METHODS = ('platt', 'monotone_platt', 'temperature', 'prior_temperature')


@dataclass
class AffineCalibration:
    slope: float
    intercept: float
    method: str

    def predict_proba(self, raw):
        values = np.asarray(raw, dtype=float).reshape(-1)
        p = expit(values * self.slope + self.intercept)
        return np.column_stack([1 - p, p])


def fit(method, raw, labels):
    raw = np.asarray(raw, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=float)
    assert set(y) == {0, 1} and np.isfinite(raw).all()
    if method == 'platt':
        return LogisticRegression(C=1, max_iter=500, random_state=42).fit(raw[:, None], y)
    prior = float(logit(y.mean()))
    if method == 'monotone_platt':
        def objective(parameters):
            a, b = parameters
            z = a * raw + b
            residual = expit(z) - y
            value = np.mean(np.logaddexp(0, z) - y * z) + a * a / (2 * len(y))
            gradient = np.array([np.mean(residual * raw) + a / len(y), residual.mean()])
            return value, gradient
        fitted = minimize(objective, [1., prior], jac=True, method='SLSQP', bounds=[(0, None), (None, None)],
                          options={'maxiter': 1000, 'ftol': 1e-12})
        assert fitted.success, fitted.message
        return AffineCalibration(float(fitted.x[0]), float(fitted.x[1]), method)
    if method not in ('temperature', 'prior_temperature'):
        raise ValueError(method)
    intercept = prior if method == 'prior_temperature' else 0.
    def objective(log_temperature):
        z = raw * np.exp(-log_temperature) + intercept
        return np.mean(np.logaddexp(0, z) - y * z)
    fitted = minimize_scalar(objective, bounds=(-5., 5.), method='bounded', options={'xatol': 1e-9})
    assert fitted.success, fitted.message
    return AffineCalibration(float(np.exp(-fitted.x)), intercept, method)


def select(raw, labels):
    raw, y = np.asarray(raw), np.asarray(labels)
    assert set(y) == {0, 1} and min(np.bincount(y.astype(int))) >= 5
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=20261018).split(raw, y))
    results = {}
    for method in METHODS:
        oof = np.empty(len(y))
        for train, held in folds:
            model = fit(method, raw[train], y[train])
            oof[held] = model.predict_proba(raw[held, None])[:, 1]
        p = oof.clip(1e-7, 1 - 1e-7)
        results[method] = float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p)))
    # Ignore numerical optimizer differences; prefer the existing recipe on ties.
    best = min(results.values())
    method = next(m for m in METHODS if results[m] <= best + 1e-5)
    return fit(method, raw, y), method, results


def reconstruct_raw(probabilities, calibrator):
    p = np.asarray(probabilities, dtype=float)
    a, b = float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])
    if abs(a) < 1e-10 or ((p <= 0) | (p >= 1)).any():
        raise ValueError('Cannot safely invert a saturated or constant calibration cache')
    raw = (logit(p) - b) / a
    assert np.max(np.abs(calibrator.predict_proba(raw[:, None])[:, 1] - p)) < 1e-10
    return raw


def augment(root, parts, entries, probabilities):
    """Append only CV-selected alternatives; original calibrated candidates stay intact."""
    out = root / 'calibration-cv'
    out.mkdir(exist_ok=True)
    ncal = len(parts['calibration'])
    cal_mask = np.array([r['source'] == 'MITweet' for r in parts['calibration']])
    labels = np.array([r['label'] for r in parts['calibration']])[cal_mask]
    ids = [r['id'] for p in ['calibration', 'gate', 'validation'] for r in parts[p]]
    additions, columns, report = {}, [], {}
    for index, (name, spec) in enumerate(entries.items()):
        if spec['kind'] == 'constant':
            continue
        signature = {'dataset': json.loads((root / 'dataset.json').read_text())['manifest_sha256'],
                     'score_sha256': digest(root / spec['scores']), 'methods': list(METHODS), 'version': 2, 'tie_tolerance': 1e-5}
        version = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
        artifacts = out / name / version
        artifacts.mkdir(parents=True, exist_ok=True)
        metadata = out / f'{name}.json'
        cached = json.loads(metadata.read_text()) if metadata.exists() else None
        if cached and cached['signature'] == signature:
            result = cached
        else:
            if spec['kind'] == 'expert':
                z = np.load(root / 'experts' / f'{spec["model"]}_raw.npz', allow_pickle=False)
                assert z['ids'].tolist() == ids
                raw = z['values'][:, 1] - z['values'][:, 0]
                derivation = 'stored raw class-score difference'
            else:
                calibrator = joblib.load(root / spec['model'])['calibrator'] if spec['kind'] == 'baseline' else joblib.load(root / spec['model'] / 'calibrator.joblib')
                try:
                    raw = reconstruct_raw(probabilities[:, index], calibrator)
                except ValueError as error:
                    report[name] = {'skipped': str(error)}
                    continue
                derivation = 'exact inverse of stored affine calibration, with round-trip check'
            model, selected, losses = select(raw[:ncal][cal_mask], labels)
            p = model.predict_proba(raw[:, None])[:, 1]
            atomic_write(artifacts / 'calibrator.joblib', lambda f: joblib.dump(model, f))
            save_scores(artifacts / 'scores.npz', p, ids, [])
            result = {'signature': signature, 'selected': selected, 'calibration_oof_log_loss': losses,
                      'validation': report_predictions(parts['validation'], p[-len(parts['validation']):]),
                      'raw_derivation': derivation, 'selection_role': 'five-fold stratified CV within human calibration only',
                      'calibrator': str((artifacts / 'calibrator.joblib').relative_to(root)), 'scores': str((artifacts / 'scores.npz').relative_to(root))}
            save_json(metadata, result)
            save_json(artifacts / 'metadata.json', result)
        report[name] = result
        if result['selected'] == 'platt':
            continue
        alternative = f'{name}_cvcal'
        additions[alternative] = dict(spec, scores=result['scores'], calibration_override=result['calibrator'], calibration_method=result['selected'])
        z = np.load(root / additions[alternative]['scores'], allow_pickle=False)
        assert z['ids'].tolist() == ids
        columns.append(z['values'])
    save_json(out / 'report.json', {'models': report, 'topic_holdout_scored': False,
                                    'selection': 'Calibration method chosen by calibration-only out-of-fold log loss; validation is reported, not used for this choice.'})
    return {**entries, **additions}, np.column_stack([probabilities, *columns]) if columns else probabilities
