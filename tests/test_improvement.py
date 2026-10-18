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
                'components': {'lexical': {'kind': 'baseline', 'model': 'classifier.joblib'}},
                'decision_rules': {'human_calibration_prior_right': .5, 'gate_balanced_threshold': .4}}
    (tmp_path / 'model.json').write_text(json.dumps(manifest))
    actual = predict(tmp_path, ['\ufeffsupport  public\nservices', texts[-1]], device='cpu')
    expected = cal.predict_proba(head.decision_function(vectorizer.transform([texts[0], texts[-1]]))[:, None])[:, 1]
    np.testing.assert_allclose([r['probability_right'] for r in actual], expected)
    np.testing.assert_allclose([r['score_right_equal_prior'] for r in actual], expected)
    assert [r['balanced_decision_label'] for r in actual] == [int(p >= .4) for p in expected]
    assert all(r['components']['lexical']['weight'] == 1 for r in actual)
    from possum.calibration_cv import AffineCalibration
    replacement = tmp_path / 'override.joblib'
    joblib.dump(AffineCalibration(0., float(np.log(3)), 'monotone_platt'), replacement)
    manifest['components']['lexical']['calibration_override'] = 'override.joblib'
    manifest['artifact_hashes']['override.joblib'] = digest(replacement)
    (tmp_path / 'model.json').write_text(json.dumps(manifest))
    assert predict(tmp_path, [texts[0]], device='cpu')[0]['probability_right'] == pytest.approx(.75)
    artifact.write_bytes(artifact.read_bytes() + b'changed')
    with pytest.raises(AssertionError, match='Changed artifact'):
        predict(tmp_path, [texts[0]], device='cpu')


def test_prior_shift_preserves_order_and_maps_prior_to_new_prior():
    from possum.improvement_audit import shift_prior
    probabilities = np.array([.01, .1, .2, .8])
    changed = shift_prior(probabilities, .1, .5)
    assert changed[1] == pytest.approx(.5)
    assert np.all(np.diff(changed) > 0)
    np.testing.assert_allclose(shift_prior(changed, .5, .1), probabilities)
    with pytest.raises(ValueError):
        shift_prior(probabilities, 0)


def test_paired_bootstrap_identical_predictions_have_zero_difference():
    from possum.improvement_audit import paired_uncertainty
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([.1, .2, .3, .7, .9])
    result = paired_uncertainty(y, p, p, draws=30)
    assert result['candidate_minus_baseline_log_loss'] == 0
    assert result['percentile_95_interval'] == [0, 0]


def test_replication_recipe_is_frozen_after_initial_selection(tmp_path):
    from possum.compact_training import TRIALS, MODERN_TRIALS, replication_plan
    (tmp_path / 'dataset.json').write_text(json.dumps({'manifest_sha256': 'frozen'}))
    for i, trial in enumerate(TRIALS + MODERN_TRIALS):
        folder = tmp_path / 'compact' / trial['name']
        folder.mkdir(parents=True)
        (folder / 'complete.json').write_text(json.dumps({'input_sha256': 'frozen'}))
        (folder / 'best.json').write_text(json.dumps({'input_sha256': 'frozen', 'trial': trial, 'log_loss': 1 / (i + 1)}))
    plan = replication_plan(tmp_path)
    assert plan['base_trial'] == MODERN_TRIALS[-1]
    assert [t['seed'] for t in plan['trials']] == [43, 44]
    assert all(t['epochs'] == 4 for t in plan['trials'])
    assert replication_plan(tmp_path) == plan
    (tmp_path / 'dataset.json').write_text(json.dumps({'manifest_sha256': 'changed'}))
    with pytest.raises(AssertionError):
        replication_plan(tmp_path)


def test_cpu_cleanup_does_not_initialize_mlx(monkeypatch):
    import builtins
    import sys
    from possum.improvement_inference import release_memory
    monkeypatch.delitem(sys.modules, 'mlx.core', raising=False)
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith('mlx'):
            raise AssertionError('CPU cleanup must not initialize MLX')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    release_memory()


def test_local_snapshot_falls_back_without_network_and_checks_all_shards(tmp_path, monkeypatch):
    import huggingface_hub
    import huggingface_hub.constants
    from possum.revisions import local_snapshot
    configured, standard = tmp_path / 'configured', tmp_path / 'home/.cache/huggingface/hub'
    monkeypatch.setattr(huggingface_hub.constants, 'HF_HUB_CACHE', str(configured))
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path / 'home'))
    for directory in [configured, standard]:
        directory.mkdir(parents=True)
        (directory / 'config.json').write_text('{}')
        (directory / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'part1.safetensors', 'b': 'part2.safetensors'}}))
        (directory / 'part1.safetensors').touch()
    (standard / 'part2.safetensors').touch()
    def cached(repo, *, revision, cache_dir, local_files_only):
        assert repo == 'test/model' and revision == 'pinned' and local_files_only is True
        return cache_dir
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', cached)
    assert local_snapshot('test/model', 'pinned') == str(standard)
    (standard / 'part2.safetensors').unlink()
    with pytest.raises(FileNotFoundError, match='Complete local snapshot unavailable'):
        local_snapshot('test/model', 'pinned')


def test_constant_prior_bundle_needs_no_model_weights(tmp_path):
    from possum.improvement_inference import predict
    manifest = {'artifact_hashes': {}, 'selection': {'kind': 'single'},
                'components': {'prior': {'kind': 'constant', 'probability': .2}},
                'decision_rules': {'human_calibration_prior_right': .2, 'gate_balanced_threshold': .15}}
    (tmp_path / 'model.json').write_text(json.dumps(manifest))
    output = predict(tmp_path, ['A short statement.', 'A different short statement.'], device='cpu')
    assert all(row['probability_right'] == .2 and row['label'] == 0 for row in output)
    assert all(row['balanced_decision_label'] == 1 for row in output)
    assert all(row['score_right_equal_prior'] == pytest.approx(.5) for row in output)


def test_monotone_calibration_cannot_reverse_class_order():
    from possum.calibration_cv import fit
    raw = np.linspace(-3, 3, 20)
    # Deliberately inverted labels would give unconstrained Platt a negative slope.
    labels = (raw < 0).astype(int)
    model = fit('monotone_platt', raw, labels)
    assert model.slope >= 0
    assert np.all(np.diff(model.predict_proba(raw[:, None])[:, 1]) >= -1e-12)


def test_cached_logits_reconstruct_without_reversing_original_model():
    from possum.calibration_cv import fit, reconstruct_raw
    raw = np.linspace(-3, 3, 20)
    model = fit('platt', raw, (raw < 0).astype(int))
    np.testing.assert_allclose(reconstruct_raw(model.predict_proba(raw[:, None])[:, 1], model), raw, atol=1e-12)
    with pytest.raises(ValueError, match='saturated'):
        reconstruct_raw(np.array([0., 1.]), model)


def test_calibration_selection_is_reproducible_with_calibration_arrays_only():
    from possum.calibration_cv import select, METHODS
    raw = np.linspace(-3, 3, 30)
    labels = (raw > 0).astype(int)
    first, method, losses = select(raw, labels)
    second, again, repeated = select(raw, labels)
    assert method == again and set(losses) == set(METHODS)
    assert losses == repeated
    np.testing.assert_allclose(first.predict_proba(raw[:, None]), second.predict_proba(raw[:, None]))


def test_export_selection_keeps_better_incumbent_and_matches_dataset(tmp_path):
    from possum.improvement_inference import select_verified_export
    (tmp_path / 'dataset.json').write_text(json.dumps({'manifest_sha256': 'same'}))
    for name, score, signature, passed in [('older', .2, 'same', True), ('newer', .3, 'same', True),
                                          ('different', .1, 'other', True), ('unverified', .05, 'same', False)]:
        folder = tmp_path / 'exports' / name
        folder.mkdir(parents=True)
        (folder / 'verification.json').write_text(json.dumps({'passed': passed}))
        (folder / 'model.json').write_text(json.dumps({'dataset': {'manifest_sha256': signature}, 'selected_name': name,
                  'validation': {'sources': {'MITweet': {'log_loss': score}}}}))
    best = select_verified_export(tmp_path)
    assert best['selected'] == 'older'
    assert len(json.loads((tmp_path / 'export-ledger.json').read_text())['candidates']) == 2
