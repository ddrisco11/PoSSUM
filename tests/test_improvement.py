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


def test_human_adapter_uses_only_fit_human_rows_and_covers_each_epoch(monkeypatch):
    from possum import human_adapter
    rows = [{'id': 'h0', 'source': 'MITweet', 'label': 0}, {'id': 'h1', 'source': 'MITweet', 'label': 1},
            {'id': 'synthetic', 'source': 'synthetic', 'label': 1}]
    monkeypatch.setattr(human_adapter, 'load_data', lambda root: {'fit': rows, 'validation': [{'id': 'protected'}]})
    assert [r['id'] for r in human_adapter.fit_rows(Path('.'))] == ['h0', 'h1']
    first = np.concatenate([human_adapter.update_indices(11, step) for step in range(3)])
    second = np.concatenate([human_adapter.update_indices(11, step) for step in range(3, 6)])
    assert sorted(first.tolist()) == sorted(second.tolist()) == list(range(11))
    assert first.tolist() != second.tolist()
    np.testing.assert_array_equal(human_adapter.update_indices(11, 4), second[4:8])
    from possum.improvement_experts import expert_settings
    assert expert_settings('mistral_human_lora_e1')['adapter'] == 'adapters/mistral_human/epoch-1'
    assert expert_settings('mistral_human_lora_e2')['adapter'] == 'adapters/mistral_human/epoch-2'


def test_adapter_scoring_rejects_changed_weights_before_model_load(tmp_path, monkeypatch):
    from possum import improvement_experts
    from possum.improvement import PARTS
    monkeypatch.setattr(improvement_experts, 'load_data', lambda root: {p: [] for p in PARTS})
    (tmp_path / 'dataset.json').write_text(json.dumps({'manifest_sha256': 'frozen'}))
    (tmp_path / 'state.json').write_text(json.dumps({'deadline': '2026-10-18T14:00:00+00:00'}))
    adapter = tmp_path / 'adapters/mistral_human/epoch-1'
    adapter.mkdir(parents=True)
    (adapter / 'adapters.safetensors').write_bytes(b'modified')
    (adapter / 'training.json').write_text(json.dumps({'complete': True, 'signature': {'manifest_sha256': 'frozen'}, 'adapter_sha256': 'wrong'}))
    with pytest.raises(AssertionError):
        improvement_experts.run(tmp_path, 'mistral_human_lora_e1')


def final_fixture(tmp_path, holdout_text):
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'topic_holdout.jsonl').write_text(holdout_text)
    (data / 'manifest.json').write_text(json.dumps({'split_hashes': {'topic_holdout': digest(data / 'topic_holdout.jsonl')}}))
    dataset = {'path': str(data), 'manifest_sha256': digest(data / 'manifest.json')}
    (tmp_path / 'dataset.json').write_text(json.dumps(dataset))
    (tmp_path / 'state.json').write_text(json.dumps({'deadline': '2099-01-01T00:00:00+00:00'}))
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'model.json').write_text(json.dumps({'dataset': dataset, 'artifact_hashes': {}, 'selected_name': 'constant',
        'selection': {'kind': 'single'}, 'components': {'prior': {'kind': 'constant', 'probability': .25}},
        'validation': {'sources': {'MITweet': {'log_loss': .5}}},
        'decision_rules': {'human_calibration_prior_right': .25, 'gate_balanced_threshold': .5}}))
    (bundle / 'verification.json').write_text(json.dumps({'passed': True}))
    (tmp_path / 'export.json').write_text(json.dumps({'path': str(bundle)}))
    return data, bundle


def test_selection_freeze_does_not_open_holdout_and_blocks_fitting(tmp_path):
    from possum.final_evaluation import freeze
    from possum.improvement import require_development_open
    data, _ = final_fixture(tmp_path, 'INVALID JSON: selection freeze must not parse this')
    with pytest.raises(ValueError, match='incomplete'):
        freeze(tmp_path)
    freeze(tmp_path, allow_incomplete=True, reason='Fixture with incomplete training')
    assert not (tmp_path / 'final-evaluation/exposure.json').exists()
    with pytest.raises(RuntimeError, match='frozen'):
        require_development_open(tmp_path)
    with pytest.raises(ValueError, match='already frozen'):
        freeze(tmp_path, allow_incomplete=True, reason='Must not change frozen candidates')


def test_final_evaluation_requires_freeze_rejects_tampering_and_runs_once(tmp_path, monkeypatch):
    from possum.final_evaluation import freeze, evaluate
    from possum import improvement_inference
    rows = [{'id': str(i), 'text': f'Example passage number {i}', 'label': i % 2, 'source': 'MITweet', 'topic': 'topic'} for i in range(4)]
    data, bundle = final_fixture(tmp_path, ''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError, match='Freeze'):
        evaluate(tmp_path, 'cpu')
    freeze(tmp_path, allow_incomplete=True, reason='Fixture')
    original = (bundle / 'model.json').read_text()
    (bundle / 'model.json').write_text(original + '\n')
    with pytest.raises(AssertionError, match='bundle changed'):
        evaluate(tmp_path, 'cpu')
    assert not (tmp_path / 'final-evaluation/exposure.json').exists()
    (bundle / 'model.json').write_text(original)
    report = evaluate(tmp_path, 'cpu')
    assert report['rows'] == 4 and report['results']['selected']['class_counts'] == {'0': 2, '1': 2}
    selected = report['results']['selected']
    assert selected['default_decision']['confusion_matrix_true_rows_predicted_columns'] == [[2, 0], [2, 0]]
    assert selected['topics']['topic']['default_decision']['class_recalls']['1']['recall'] == 0
    assert selected['uncertainty']['draws'] == 2000
    sensitivity = selected['equal_prior_sensitivity']
    assert sensitivity['calibration_prior_right'] == .25 and sensitivity['assumed_prior_right'] == .5
    assert sensitivity['metrics']['all']['log_loss'] == pytest.approx(np.log(2))
    assert sensitivity['decision']['confusion_matrix_true_rows_predicted_columns'] == [[0, 2], [0, 2]]
    assert sensitivity['replaces_primary_scores'] is False
    assert (tmp_path / 'final-evaluation/exposure.json').exists()
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed holdout must not run predictions again')
    monkeypatch.setattr(improvement_inference, 'predict', forbidden)
    assert evaluate(tmp_path, 'cpu') == report


def test_inference_benchmark_sampling_and_batch_parity_use_supplied_rows_only():
    from possum.inference_benchmark import sample_rows, time_requests
    rows = [{'id': f'{source}-{label}-{i}', 'source': source, 'label': label, 'text': 'word ' * (i + 1)}
            for source in ['human', 'synthetic'] for label in [0, 1] for i in range(5)]
    selected = sample_rows(rows)
    assert len(selected) == 12 and {len(r['text'].split()) for r in selected} == {1, 3, 5}
    calls = []
    def predictor(bundle, texts, device):
        calls.append(texts)
        return [{'probability_right': len(t.split()) / 10} for t in texts]
    measured = time_requests('fixture', selected, 'cpu', float('inf'), predictor)
    assert len(calls) == 13 and len(calls[-1]) == 12
    assert measured['batch_single_max_absolute_error'] == 0
    assert measured['examples'] == 12
    def inconsistent(bundle, texts, device):
        return [{'probability_right': .2 if len(texts) == 1 else .8} for t in texts]
    with pytest.raises(AssertionError, match='mismatch'):
        time_requests('fixture', selected, 'cpu', float('inf'), inconsistent)
    with pytest.raises(TimeoutError):
        time_requests('fixture', selected, 'cpu', 0, predictor)


def test_final_decisions_expose_class_errors_and_undefined_absent_recall():
    from possum.final_evaluation import decision_summary, uncertainty
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([.1, .2, .8, .3, .9])
    summary = decision_summary(y, p)
    assert summary['confusion_matrix_true_rows_predicted_columns'] == [[2, 1], [1, 1]]
    assert summary['balanced_accuracy'] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert summary['class_recalls']['0']['n'] == 3
    assert summary['class_recalls']['1']['wilson_95_interval'] == pytest.approx([.0945312057, .9054687943])
    changed = decision_summary(y, p, .25)
    assert changed['confusion_matrix_true_rows_predicted_columns'] == [[2, 1], [0, 2]]
    one_class = decision_summary(y[:3], p[:3])
    assert one_class['balanced_accuracy'] is None
    assert one_class['class_recalls']['1']['recall'] is None
    assert one_class['class_recalls']['1']['wilson_95_interval'] is None
    intervals = uncertainty(y, p, draws=100)
    assert uncertainty(y, p, draws=100) == intervals
    assert set(intervals['percentile_95_intervals']) == {'log_loss', 'brier', 'balanced_accuracy', 'macro_f1'}
    assert all(np.isfinite(v).all() and v[0] <= v[1] for v in intervals['percentile_95_intervals'].values())
    perfect = uncertainty(y, y.astype(float), draws=100)
    assert perfect['percentile_95_intervals']['macro_f1'] == [1., 1.]
    assert perfect['percentile_95_intervals']['brier'] == [0., 0.]
    assert set(uncertainty(y[:3], p[:3], draws=100)['percentile_95_intervals']) == {'log_loss', 'brier'}
    with pytest.raises(AssertionError):
        uncertainty(y, p * 2)


def test_best_compact_selection_searches_history_and_excludes_ensembles(tmp_path):
    from possum.improvement_inference import best_compact_comparison, export
    (tmp_path / 'dataset.json').write_text(json.dumps({'manifest_sha256': 'active'}))
    with pytest.raises(ValueError, match='No single compact'):
        best_compact_comparison(tmp_path)
    def comparison(folder, signature, compact_loss):
        folder.mkdir(parents=True)
        payload = {'dataset': {'manifest_sha256': signature}, 'selected': 'mixture',
          'candidates': {'bert': {'kind': 'compact'}, 'mistral': {'kind': 'expert'}},
          'results': {
            'bert': {'spec': {'kind': 'single', 'index': 0}, 'validation': {'sources': {'MITweet': {'log_loss': compact_loss}}}},
            'mistral': {'spec': {'kind': 'single', 'index': 1}, 'validation': {'sources': {'MITweet': {'log_loss': .1}}}},
            'mixture': {'spec': {'kind': 'fixed', 'weights': [.5, .5]}, 'validation': {'sources': {'MITweet': {'log_loss': .05}}}}}}
        (folder / 'comparison.json').write_text(json.dumps(payload))
    comparison(tmp_path / 'mixtures', 'active', .3)
    comparison(tmp_path / 'mixture-history/earlier', 'active', .2)
    comparison(tmp_path / 'mixture-history/unrelated', 'different', .01)
    result = best_compact_comparison(tmp_path)
    assert result['candidate'] == 'bert' and result['human_validation_log_loss'] == .2
    assert Path(result['comparison_dir']).name == 'earlier'
    with pytest.raises(ValueError, match='cannot be combined'):
        export(tmp_path, candidate='bert', best_compact=True)
    with pytest.raises(ValueError, match='cannot be combined'):
        export(tmp_path, comparison_dir=tmp_path / 'mixtures', best_compact=True)


def test_pruning_uses_total_predictor_mass_and_keeps_one_predictor():
    from possum.improvement_mixtures import predictor_groups, prune_predictors
    entries = {'a': {'kind': 'expert', 'model': 'a'}, 'a_cv': {'kind': 'expert', 'model': 'a'},
               'b': {'kind': 'expert', 'model': 'b'}, 'c': {'kind': 'compact', 'model': 'c'}}
    groups = predictor_groups(entries)
    weights = np.array([.006, .006, .005, .983])
    pruned = prune_predictors(weights, groups)
    np.testing.assert_allclose(pruned, np.array([.006, .006, 0., .983]) / .995)
    assert pruned.sum() == pytest.approx(1.) and (pruned >= 0).all()
    # Two sub-threshold calibrators survive because their shared predictor exceeds 1%.
    assert pruned[0] > 0 and pruned[1] > 0 and pruned[2] == 0
    reduced = prune_predictors(np.array([.012, .005, .983]), np.eye(3))
    assert pruned[0] + pruned[1] == pytest.approx(reduced[0])
    np.testing.assert_allclose(prune_predictors([.5, .5], np.eye(2), .9), [1., 0.])
    np.testing.assert_allclose(prune_predictors([.01, .99], np.eye(2)), [.01, .99])
    with pytest.raises(AssertionError):
        prune_predictors([-.1, 1.1], np.eye(2))


def test_deberta_plan_pins_data_recipe_revision_and_reserves_final_hour(tmp_path):
    from possum.compact_training import deberta_plan
    import datetime
    (tmp_path / 'dataset.json').write_text(json.dumps({'manifest_sha256': 'fixed'}))
    (tmp_path / 'state.json').write_text(json.dumps({'deadline': '2099-01-01T14:00:00+00:00'}))
    plan = deberta_plan(tmp_path)
    assert [r['regime'] for r in plan['trials']] == ['mixed', 'human']
    assert all(r['epochs'] == 4 for r in plan['trials'])
    assert plan['model_revision'] == '8ccc9b6f36199bec6961081d44eb72fb3f7353f3'
    assert plan['compute_until'] == datetime.datetime.fromisoformat('2099-01-01T13:00:00+00:00').timestamp()
    assert deberta_plan(tmp_path) == plan
    (tmp_path / 'dataset.json').write_text(json.dumps({'manifest_sha256': 'changed'}))
    with pytest.raises(AssertionError, match='plan changed'):
        deberta_plan(tmp_path)


def test_expert_export_limit_is_forwarded_and_memory_released_on_error(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from possum import improvement_inference as inference
    from possum.components import causal
    captured, released = {}, []
    class RejectingScorer:
        model = SimpleNamespace(eval=lambda: None)
        def __init__(self, path, **kwargs):
            captured.update(kwargs)
        def score(self, texts):
            raise ValueError('Input exceeds the validated limit')
    monkeypatch.setattr(causal, 'MLXCausal', RejectingScorer)
    monkeypatch.setattr(inference, 'local_snapshot', lambda *args: 'fixture')
    monkeypatch.setattr(inference, 'release_memory', lambda: released.append(True))
    spec = {'kind': 'expert', 'model': 'mistral', 'engine': 'mlx',
            'backbone': ['fixture', 'revision'], 'max_input_tokens': 256}
    with pytest.raises(ValueError, match='validated limit'):
        inference.raw_component(tmp_path, spec, ['example'], 'cpu', {})
    assert captured['max_input_tokens'] == 256 and released == [True]


def test_compact_resume_rejects_nonfinite_parameters_and_moments():
    import torch
    import pytest
    from possum.compact_training import validate_resume_tensors
    state = {'model': {'weight': torch.ones(2)},
             'optimizer': {'state': {0: {'exp_avg': torch.zeros(2), 'step': torch.tensor(1.)}}}}
    validate_resume_tensors(state)
    state['model']['weight'][0] = float('nan')
    with pytest.raises(ValueError, match='checkpoint parameter'):
        validate_resume_tensors(state)
    state['model']['weight'][0] = 1.
    state['optimizer']['state'][0]['exp_avg'][1] = float('inf')
    with pytest.raises(ValueError, match='optimizer tensor'):
        validate_resume_tensors(state)


def test_single_expert_reference_is_frozen_before_holdout_and_cannot_change(tmp_path, monkeypatch):
    import shutil
    from possum.final_evaluation import freeze, evaluate
    from possum import improvement_inference
    rows = [{'id': str(i), 'text': 'Fixture text', 'label': i % 2,
             'source': 'MITweet', 'topic': 'fixture'} for i in range(4)]
    _, bundle = final_fixture(tmp_path, ''.join(json.dumps(r) + '\n' for r in rows))
    reference = tmp_path / 'single-expert'
    shutil.copytree(bundle, reference)
    manifest = json.loads((reference / 'model.json').read_text())
    manifest.update(selected_name='reference', components={'expert': {'kind': 'expert', 'model': 'fixture'}})
    manifest['selection'] = {'kind': 'fixed', 'weights': [1.]}
    (reference / 'model.json').write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match='Expected one expert'):
        freeze(tmp_path, allow_incomplete=True, reason='Fixture', expert_bundle=reference)
    assert not (tmp_path / 'selection-freeze.json').exists()
    manifest['selection'] = {'kind': 'single'}
    (reference / 'model.json').write_text(json.dumps(manifest))
    plan = freeze(tmp_path, allow_incomplete=True, reason='Fixture', expert_bundle=reference)
    assert set(plan['bundles']) == {'selected', 'single_expert'}
    assert not (tmp_path / 'final-evaluation/exposure.json').exists()
    original = (reference / 'model.json').read_text()
    (reference / 'model.json').write_text(original + '\n')
    with pytest.raises(AssertionError, match='Frozen bundle changed'):
        evaluate(tmp_path, 'cpu')
    assert not (tmp_path / 'final-evaluation/exposure.json').exists()
    (reference / 'model.json').write_text(original)
    calls = []
    def predict(path, texts, device):
        calls.append(Path(path).name)
        return [{'probability_right': .4} for _ in texts]
    monkeypatch.setattr(improvement_inference, 'predict', predict)
    result = evaluate(tmp_path, 'cpu')
    assert set(result['results']) == {'selected', 'single_expert'}
    assert calls == ['bundle', 'single-expert']
    assert evaluate(tmp_path, 'cpu') == result
    assert len(calls) == 2
