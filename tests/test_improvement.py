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


def test_distillation_loss_preserves_hard_loss_and_detaches_teacher():
    import torch
    from possum.distillation import training_loss
    logits = torch.tensor([[1., -1.], [-.2, .4]], requires_grad=True)
    teacher = torch.tensor([[2., -2.], [-1., 1.]], requires_grad=True)
    labels = torch.tensor([0, 1])
    weights = torch.tensor([.5, 1.5])
    hard = (torch.nn.functional.cross_entropy(logits, labels, reduction='none') * weights).mean()
    torch.testing.assert_close(training_loss(logits, labels, weights), hard)
    torch.testing.assert_close(training_loss(logits, labels, weights, teacher, alpha=0), hard)
    loss = training_loss(logits, labels, weights, teacher, alpha=.5)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and teacher.grad is None
    with pytest.raises(ValueError, match='required'):
        training_loss(logits, labels, weights, alpha=.5)


def test_distillation_targets_reject_role_order_and_integrity_changes(tmp_path):
    from possum.distillation import load_targets
    from possum.checkpoints import save_scores
    from possum.improvement import digest
    rows = [{'id': 'fit-a'}, {'id': 'fit-b'}]
    (tmp_path / 'dataset.json').write_text(json.dumps({'manifest_sha256': 'dataset'}))
    directory = tmp_path / 'distillation'
    directory.mkdir()
    path = directory / 'teacher_raw.npz'
    save_scores(path, [[-1., -2.], [-3., -1.]], ['fit-a', 'fit-b'], [])
    meta = dict(complete=True, input_sha256='dataset', teacher='mistral', role='fit', scores_sha256=digest(path))
    (directory / 'teacher.json').write_text(json.dumps(meta))
    scores, _ = load_targets(tmp_path, rows, 'mistral')
    assert scores.shape == (2, 2)
    with pytest.raises(AssertionError, match='fit rows'):
        load_targets(tmp_path, rows[::-1], 'mistral')
    meta['role'] = 'validation'
    (directory / 'teacher.json').write_text(json.dumps(meta))
    with pytest.raises(AssertionError):
        load_targets(tmp_path, rows, 'mistral')
    meta['role'] = 'fit'
    (directory / 'teacher.json').write_text(json.dumps(meta))
    save_scores(path, [[-2., -1.], [-3., -1.]], ['fit-a', 'fit-b'], [])
    with pytest.raises(AssertionError, match='cache changed'):
        load_targets(tmp_path, rows, 'mistral')


def test_grouped_mixture_penalty_is_invariant_to_duplicate_calibration_candidates():
    from possum.improvement_mixtures import fit_fixed, predictor_groups
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([[.1, .2], [.3, .1], [.2, .4], [.7, .1], [.8, .6], [.9, .5], [.4, .8], [.6, .9]])
    entries = {'a': {'kind': 'expert', 'model': 'a'}, 'b': {'kind': 'expert', 'model': 'b'}}
    first = fit_fixed(scores, labels, .1, predictor_groups(entries))
    duplicate = dict(entries, a_calibrated={'kind': 'expert', 'model': 'a', 'calibration_override': 'other'})
    expanded = np.column_stack([scores, scores[:, 0]])
    second = fit_fixed(expanded, labels, .1, predictor_groups(duplicate))
    np.testing.assert_allclose(scores @ first, expanded @ second, atol=2e-5)
    assert second[0] + second[2] == pytest.approx(first[0], abs=3e-5)


def test_inference_reuses_raw_model_scores_across_calibrators_only_within_batch(tmp_path, monkeypatch):
    import joblib
    from possum import improvement_inference as module
    from possum.calibration_cv import AffineCalibration
    calls = []
    def fake_raw(bundle, spec, texts, device, embeddings):
        calls.append((spec['model'], list(texts)))
        return np.arange(len(texts), dtype=float), AffineCalibration(1., 0., 'temperature')
    monkeypatch.setattr(module, 'raw_component', fake_raw)
    joblib.dump(AffineCalibration(0., np.log(3), 'monotone_platt'), tmp_path / 'override.joblib')
    shared = {'kind': 'expert', 'model': 'mistral', 'backbone': ['model', 'revision'], 'calibrator': 'shared.joblib'}
    manifest = {'artifact_hashes': {}, 'components': {'original': shared, 'alternative': dict(shared, calibration_override='override.joblib')},
                'selection': {'kind': 'fixed', 'weights': [.5, .5]}}
    (tmp_path / 'model.json').write_text(json.dumps(manifest))
    result = module.predict(tmp_path, ['first passage', 'second passage'], device='cpu')
    assert len(calls) == 1
    assert result[0]['probability_right'] == pytest.approx(.625)
    assert result[1]['components']['alternative']['probability_right'] == pytest.approx(.75)
    module.predict(tmp_path, ['new passage'], device='cpu')
    assert len(calls) == 2 and calls[-1][1] == ['new passage']


def test_export_shares_predictor_artifacts_and_preserves_calibrated_parity(tmp_path, monkeypatch):
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from possum import improvement_inference as module, improvement_audit
    from possum.calibration_cv import AffineCalibration
    from possum.checkpoints import save_scores
    texts = ['support public services', 'expand social welfare', 'reduce income taxes', 'protect private markets']
    vectorizer = TfidfVectorizer().fit(texts)
    head = LogisticRegression().fit(vectorizer.transform(texts), [0, 0, 1, 1])
    calibrator = AffineCalibration(1., 0., 'temperature')
    alternate = AffineCalibration(.8, -.2, 'platt')
    joblib.dump(dict(features='lexical', vectorizer=vectorizer, head=head, calibrator=calibrator), tmp_path / 'classifier.joblib')
    joblib.dump(alternate, tmp_path / 'alternate.joblib')
    ids = [str(i) for i in range(4)]
    raw = head.decision_function(vectorizer.transform(texts))[:, None]
    expected = .4 * calibrator.predict_proba(raw)[:, 1] + .6 * alternate.predict_proba(raw)[:, 1]
    mixture = tmp_path / 'mixtures'
    mixture.mkdir()
    save_scores(mixture / 'fixed_scores.npz', expected, ids, [])
    spec = {'kind': 'baseline', 'model': 'classifier.joblib'}
    comparison = {'selected': 'fixed', 'candidates': {'original': spec, 'alternative': dict(spec, calibration_override='alternate.joblib')},
                  'results': {'fixed': {'spec': {'kind': 'fixed', 'weights': [.4, .6]},
                                        'validation': {'sources': {'MITweet': {'log_loss': .2}}}}},
                  'dataset': {'manifest_sha256': 'test'}, 'limitations': []}
    (mixture / 'comparison.json').write_text(json.dumps(comparison))
    (tmp_path / 'dataset.json').write_text(json.dumps(comparison['dataset']))
    (mixture / 'audit.json').write_text(json.dumps({'human_calibration_prior_right': .5,
        'models': {'fixed': {'gate_selected_balanced_threshold': {'threshold': .5}}}}))
    monkeypatch.setattr(improvement_audit, 'run', lambda root, comparison_dir=None, write=True: json.loads((Path(comparison_dir) / 'audit.json').read_text()))
    monkeypatch.setattr(module, 'resolve', lambda name: 'pinned')
    monkeypatch.setattr(module, 'load_data', lambda root: {'calibration': [{'id': i, 'text': t} for i, t in zip(ids, texts)]})
    module.export(tmp_path, device='cpu')
    pointer = json.loads((tmp_path / 'export.json').read_text())
    bundle = Path(pointer['path'])
    manifest = json.loads((bundle / 'model.json').read_text())
    assert manifest['components']['original']['model'] == manifest['components']['alternative']['model']
    assert len(list((bundle / 'components').iterdir())) == 2  # one predictor, one alternative calibrator
    np.testing.assert_allclose([r['probability_right'] for r in module.predict(bundle, texts, device='cpu')], expected)
    assert json.loads((bundle / 'verification.json').read_text())['passed']
    # Exporting a cheaper named candidate must not replace a better incumbent.
    comparison['results']['original'] = {'spec': {'kind': 'single', 'index': 0},
                                       'validation': {'sources': {'MITweet': {'log_loss': .3}}}}
    (mixture / 'comparison.json').write_text(json.dumps(comparison))
    save_scores(mixture / 'original_scores.npz', calibrator.predict_proba(raw)[:, 1], ids, [])
    (mixture / 'audit.json').write_text(json.dumps({'human_calibration_prior_right': .5,
        'models': {name: {'gate_selected_balanced_threshold': {'threshold': .5}} for name in ['fixed', 'original']}}))
    module.export(tmp_path, device='cpu', candidate='original')
    assert json.loads((tmp_path / 'export.json').read_text())['selected'] == 'fixed'
    assert len(json.loads((tmp_path / 'export-ledger.json').read_text())['candidates']) == 2


    # A prior comparison stays exportable even after the current score file changes.
    import shutil
    archived = tmp_path / 'mixture-history' / 'earlier'
    shutil.copytree(mixture, archived)
    comparison['selected'] = 'original'
    (mixture / 'comparison.json').write_text(json.dumps(comparison))
    save_scores(mixture / 'fixed_scores.npz', np.full(4, .99), ids, [])
    assert module.best_comparison(tmp_path) == archived
    module.export(tmp_path, device='cpu')
    latest = sorted((tmp_path / 'exports').iterdir(), key=lambda p: p.stat().st_mtime_ns)[-1]
    assert json.loads((latest / 'model.json').read_text())['source_comparison']['path'] == 'mixture-history/earlier'
    assert json.loads((latest / 'verification.json').read_text())['passed']


def test_chat_scoring_uses_nonreasoning_assistant_completion_and_preserves_plain_prompt():
    from possum.components.causal import scoring_prompt, prompt
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{'role': 'user', 'content': prompt('Some passage')}]
            assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
            return 'assistant-prefix\n'
    tokenizer = Tokenizer()
    assert scoring_prompt(tokenizer, 'Some passage') == prompt('Some passage')
    assert scoring_prompt(tokenizer, 'Some passage', 'chat_nonthinking') == 'assistant-prefix\nThis statement reflects a'
    with pytest.raises(ValueError, match='Unknown'):
        scoring_prompt(tokenizer, 'Some passage', 'typo')
    from possum.improvement_experts import expert_settings
    raw, chat = expert_settings('qwen'), expert_settings('qwen_chat')
    assert raw['backbone'] == chat['backbone'] and chat['engine'] == 'mlx'
    assert raw['prompt_format'] == 'plain' and chat['prompt_format'] == 'chat_nonthinking'


def test_best_comparison_excludes_different_dataset_histories(tmp_path):
    from possum.improvement_inference import best_comparison
    (tmp_path / 'dataset.json').write_text(json.dumps({'manifest_sha256': 'active'}))
    for directory, signature, loss in [('mixtures', 'active', .3), ('mixture-history/valid', 'active', .2),
                                       ('mixture-history/other', 'other', .1)]:
        folder = tmp_path / directory
        folder.mkdir(parents=True)
        (folder / 'comparison.json').write_text(json.dumps({'dataset': {'manifest_sha256': signature},
            'selected': 'model', 'results': {'model': {'validation': {'sources': {'MITweet': {'log_loss': loss}}}}}}))
    assert best_comparison(tmp_path) == tmp_path / 'mixture-history/valid'
