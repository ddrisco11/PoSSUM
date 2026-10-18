import json
from pathlib import Path

import numpy as np
import pytest

from possum.improvement import PARTS, calibration, digest, load_data, sample_weights


def test_source_class_weights_give_equal_cell_mass():
    rows = [{'source': source, 'label': label} for source, label, n in
            [('human', 0, 7), ('human', 1, 2), ('synthetic', 0, 3), ('synthetic', 1, 10)] for _ in range(n)]
    weights = sample_weights(rows)
    assert weights.mean() == pytest.approx(1)
    for source, label in {(r['source'], r['label']) for r in rows}:
        assert sum(w for r, w in zip(rows, weights) if (r['source'], r['label']) == (source, label)) == pytest.approx(len(rows) / 4)


def test_calibration_ignores_synthetic_labels():
    rows = [{'source': 'MITweet', 'label': y} for y in [0, 0, 1, 1]]
    raw = np.array([-2., -1., 1., 2.])
    a = calibration(raw, rows)
    b = calibration(np.r_[raw, 100., -100.], rows + [{'source': 'synthetic', 'label': 0}, {'source': 'synthetic', 'label': 1}])
    np.testing.assert_allclose(a.predict_proba(raw[:, None]), b.predict_proba(raw[:, None]))


def test_loader_rejects_modified_partition_without_reading_topic_holdout(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    for part in PARTS:
        (data / f'{part}.jsonl').write_text(json.dumps({'id': part, 'text': 'example', 'label': 0}) + '\n')
    (data / 'topic_holdout.jsonl').write_text('not valid json: must never be opened by development loader')
    manifest = data / 'manifest.json'
    manifest.write_text(json.dumps({'split_hashes': {part: digest(data / f'{part}.jsonl') for part in PARTS}}))
    (tmp_path / 'dataset.json').write_text(json.dumps({'path': str(data), 'manifest_sha256': digest(manifest)}))
    assert set(load_data(tmp_path)) == set(PARTS)
    (data / 'fit.jsonl').write_text('{}\n')
    with pytest.raises(AssertionError):
        load_data(tmp_path)


def test_reloadable_lexical_prediction_and_artifact_integrity(tmp_path):
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from possum.improvement_inference import predict
    texts = ['support public services', 'expand social welfare', 'reduce income taxes', 'protect private markets']
    vectorizer = TfidfVectorizer().fit(texts)
    head = LogisticRegression().fit(vectorizer.transform(texts), [0, 0, 1, 1])
    raw = head.decision_function(vectorizer.transform(texts))
    cal = calibration(raw, [{'source': 'MITweet', 'label': y} for y in [0, 0, 1, 1]])
    artifact = tmp_path / 'classifier.joblib'
    joblib.dump({'features': 'lexical', 'vectorizer': vectorizer, 'head': head, 'calibrator': cal}, artifact)
    manifest = {'artifact_hashes': {'classifier.joblib': digest(artifact)}, 'selection': {'kind': 'single'},
                'components': {'lexical': {'kind': 'baseline', 'model': 'classifier.joblib'}}}
    (tmp_path / 'model.json').write_text(json.dumps(manifest))
    actual = predict(tmp_path, ['\ufeffsupport  public\nservices', texts[-1]], device='cpu')
    expected = cal.predict_proba(head.decision_function(vectorizer.transform([texts[0], texts[-1]]))[:, None])[:, 1]
    np.testing.assert_allclose([r['probability_right'] for r in actual], expected)
    assert all(r['components']['lexical']['weight'] == 1 for r in actual)
    artifact.write_bytes(artifact.read_bytes() + b'changed')
    with pytest.raises(AssertionError, match='Changed artifact'):
        predict(tmp_path, [texts[0]], device='cpu')
