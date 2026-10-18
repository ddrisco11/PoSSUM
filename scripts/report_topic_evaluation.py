"""Validate and publish aggregate results from an already completed frozen audit."""
import argparse
import json
from pathlib import Path

import numpy as np

from possum.checkpoints import save_json
from possum.final_evaluation import bundle_record
from possum.improvement import digest


def report(root):
    plan_path = root / 'selection-freeze.json'
    plan = json.loads(plan_path.read_text())
    result = json.loads((root / 'final-evaluation/report.json').read_text())
    exposure = json.loads((root / 'final-evaluation/exposure.json').read_text())
    signature = digest(plan_path)
    assert result['selection_sha256'] == exposure['selection_sha256'] == signature
    assert result['holdout_sha256'] == plan['holdout_sha256']
    assert result['dataset_sha256'] == plan['dataset']['manifest_sha256']
    assert set(result['results']) == set(plan['bundles'])
    records = {}
    score_ids = None
    for name, record in plan['bundles'].items():
        verified = bundle_record(root, record['path'])
        assert verified['manifest_sha256'] == record['manifest_sha256']
        scores_path = root / 'final-evaluation' / f'{name}_scores.npz'
        metadata = json.loads(scores_path.with_suffix('.json').read_text())
        assert metadata == {'selection_sha256': signature, 'bundle_sha256': record['manifest_sha256']}
        scores = np.load(scores_path, allow_pickle=False)
        assert len(scores['values']) == len(scores['ids']) == result['rows']
        assert np.isfinite(scores['values']).all() and ((scores['values'] >= 0) & (scores['values'] <= 1)).all()
        ids = scores['ids'].tolist()
        assert len(set(ids)) == len(ids)
        assert score_ids is None or score_ids == ids
        score_ids = ids
        assert result['results'][name]['all']['n'] == sum(result['results'][name]['class_counts'].values()) == result['rows']
        records[name] = {key: record[key] for key in ['manifest_sha256', 'selected_name', 'decision_rules', 'validation']}
    return dict(result, frozen_candidates=records, audit_checks={
        'all_bundles_and_prediction_provenance_verified': True,
        'matched_complete_prediction_identities': True,
        'new_predictions_or_model_selection_performed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    save_json(args.output, report(args.run))
