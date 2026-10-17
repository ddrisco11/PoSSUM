"""Exploratory equal-weight subset audit using completed caches; no fitting or export changes."""
import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import softmax

from evaluate_adapter_diagnostic import exported_predictions
from possum.checkpoints import save_json
from possum.data import read_jsonl
from possum.learning import bootstrap_delta, metrics


def audit(root):
    meta = json.loads((root / 'run.json').read_text())
    if meta['status'] != 'complete' or time.time() >= meta['deadline']:
        raise ValueError('Requires completed main run and remaining original budget')
    config = json.loads((root / 'export/config.json').read_text())
    names = config['experts']
    times = {}
    for name in names:
        p = json.loads((root / f'{name}_score_progress.json').read_text())
        times[name] = p['seconds'] / p['completed']
    report = dict(scope='Post-hoc equal-weight subset audit; original selected export unchanged',
                  fitting='None; reuse original calibration temperatures',
                  timing_scope='Sum of measured per-example scoring compute; excludes model loading, swapping and checkpoint writes',
                  component_compute_seconds=times, datasets={})
    for scope, folder in [('development_validation', root), ('previously_seen_ibc', root / 'ibc-diagnostic')]:
        rows = read_jsonl(folder / 'input/validation.jsonl')
        n = len(rows)
        ids = [r['id'] for r in rows]
        y = np.array([r['label'] for r in rows])
        raw = {}
        for name in names:
            path = folder / f'{name}_scores.npz'
            if not json.loads(path.with_suffix('.json').read_text())['complete']:
                raise ValueError('Incomplete score cache')
            with np.load(path, allow_pickle=False) as z:
                if z['ids'].tolist()[-n:] != ids:
                    raise ValueError('Subset score alignment mismatch')
                raw[name] = z['values'][-n:]
        probabilities = {k: softmax(v / config['temperatures'][k], axis=1)[:, 1] for k, v in raw.items()}
        results = {}
        for count in range(1, len(names)+1):
            for subset in itertools.combinations(names, count):
                p = np.column_stack([probabilities[k] for k in subset]).mean(1)
                result = dict(experts=list(subset), metrics=metrics(y, p),
                              estimated_compute_seconds=sum(times[k] for k in subset), by_source={})
                for source in sorted({r['source'] for r in rows}):
                    mask = np.array([r['source'] == source for r in rows])
                    result['by_source'][source] = metrics(y[mask], p[mask])
                results['+'.join(subset)] = result
        report['datasets'][scope] = dict(count=n, results=results)
        if scope == 'development_validation':
            with np.load(folder / 'bge_scores.npz', allow_pickle=False) as z:
                if z['ids'].tolist()[-n:] != ids:
                    raise ValueError('Subset embedding alignment mismatch')
                embeddings = z['values'][-n:]
            predictions, selected = exported_predictions(root / 'export', raw, embeddings)
            pair = (probabilities['bert'] + probabilities['mistral']) / 2
            report['pair_versus_original_export'] = bootstrap_delta(
                y, pair, predictions[selected], np.array([r['group'] for r in rows]), repeats=2000)
    report['limitations'] = [
        'Subset analysis is exploratory and follows inspection of development and IBC results.',
        'No choice here is validated on fresh human data; IBC has only 97 previously inspected examples.',
        'The paired group-bootstrap interval is descriptive and not adjusted for model selection.',
        'Removing an expert can affect domains differently; aggregate ranking alone is insufficient.']
    save_json(root / 'expert-subset-audit.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    audit(args.run.resolve())
