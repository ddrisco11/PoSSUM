"""Join frozen model identity with completed evaluation and deployment evidence."""
import argparse
import json
from pathlib import Path

from .checkpoints import save_json
from .final_evaluation import bundle_record
from .improvement import digest


def build(root, reports):
    root, reports = Path(root), Path(reports)
    plan_path = root / 'selection-freeze.json'
    plan = json.loads(plan_path.read_text())
    topic_path = reports / 'human-topic-stress.json'
    topic = json.loads(topic_path.read_text())
    local_topic = json.loads((root / 'final-evaluation/report.json').read_text())
    assert local_topic == {key: topic[key] for key in local_topic}, 'Published topic results differ'
    portability_path = reports / 'human-export-portability.json'
    portability = json.loads(portability_path.read_text())
    latency_path = reports / 'human-inference-latency.json'
    latency = json.loads(latency_path.read_text())
    signature = digest(plan_path)
    assert topic['selection_sha256'] == portability['selection_sha256'] == signature
    assert topic['dataset_sha256'] == portability['dataset_sha256'] == plan['dataset']['manifest_sha256']
    assert set(topic['results']) == set(portability['results']) == set(plan['bundles'])
    timings = {result['bundle_manifest_sha256']: result for result in latency['results']}
    models = {}
    for variant, record in plan['bundles'].items():
        bundle = Path(record['path'])
        verified = bundle_record(root, bundle)
        sha = record['manifest_sha256']
        assert verified['manifest_sha256'] == sha
        moved = portability['results'][variant]
        assert moved['manifest_sha256'] == sha and moved['passed']
        timing = timings[sha]
        assert timing['plan']['dataset_sha256'] == plan['dataset']['manifest_sha256']
        assert timing['measurements']['batch_single_parity_passed']
        manifest = json.loads((bundle / 'model.json').read_text())
        models[variant] = {
            'selected_name': manifest['selected_name'], 'manifest_sha256': sha,
            'component_names': list(manifest['components']),
            'combination': {key: value for key, value in manifest['selection'].items() if key in {'kind', 'weights', 'penalty'}},
            'decision_rules': manifest['decision_rules'],
            'human_development_validation': manifest['validation']['sources']['MITweet'],
            'topic_stress': topic['results'][variant]['all'],
            'inference_timing': timing['measurements'], 'relocation_verification': moved,
            'local_bundle_bytes': sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file()),
            'base_models_cached_separately': [spec['backbone'] for spec in manifest['components'].values() if spec.get('engine') == 'mlx'],
        }
    evidence = [topic_path, portability_path, latency_path]
    cpu_path = reports / 'human-compact-cpu.json'
    if cpu_path.exists():
        cpu = json.loads(cpu_path.read_text())
        assert cpu['selection_sha256'] == signature and cpu['dataset_sha256'] == plan['dataset']['manifest_sha256']
        assert cpu['manifest_sha256'] == models['compact']['manifest_sha256'] and cpu['passed'] and cpu['device'] == 'cpu'
        models['compact']['cpu_verification'] = cpu
        evidence.append(cpu_path)
    return {'release_id': 'PoSSUM-2026-10-18-research', 'status': 'Research candidate; completed frozen topic stress audit.',
            'primary_variant': 'selected', 'selection_sha256': signature,
            'dataset_sha256': plan['dataset']['manifest_sha256'], 'topic_holdout_sha256': topic['holdout_sha256'],
            'models': models,
            'evidence_sha256': {p.name: digest(p) for p in evidence},
            'scoring_contract': {'labels': {'0': 'Left', '1': 'Right'}, 'default_threshold': .5,
                                 'development_word_range': [12, 150], 'maximum_model_input_tokens': 256,
                                 'meaning': 'Binary class probability under the calibration distribution, not ideological intensity or neutrality.'},
            'limitations': ['Weights and base-model caches are local and are not included in the Git repository.',
                           'Only 159 passages across three topics were held out; no author-disjoint guarantee.',
                           'Labels derive from mapped human ideology facets; not fresh overall-U.S. adjudication.',
                           'Neutrality, abstention, broader independent evaluation and clean-machine installation remain unvalidated.',
                           'No candidate was reselected using the topic test.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--reports', type=Path, default=Path('reports'))
    args = parser.parse_args()
    public = build(args.run, args.reports)
    save_json(args.reports / 'human-model-release.json', public)
    plan = json.loads((args.run / 'selection-freeze.json').read_text())
    save_json(args.run / 'release.json', dict(public, local_bundles={name: record['path'] for name, record in plan['bundles'].items()}))
    print(json.dumps({'release_id': public['release_id'], 'variants': list(public['models'])}))
