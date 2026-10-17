"""Independently verify assembled split/provenance invariants, without test scoring."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from possum.data import normalize, read_jsonl


def check(root):
    rows = read_jsonl(root / 'candidates.jsonl')
    split = json.loads((root / 'split.json').read_text())
    split_sets = {part: set(ids) for part, ids in split.items()}
    protected = json.loads((root / 'protected.json').read_text())
    report = json.loads((root / 'manifest.json').read_text())
    failures = []

    def require(condition, message):
        if not condition:
            failures.append(message)

    by_id = {row['id']: row for row in rows}
    require(len(by_id) == len(rows), 'Candidate IDs are not unique')
    declared = [key for ids in split.values() for key in ids]
    require(len(set(declared)) == len(declared), 'IDs cross split partitions')
    require(set(declared) == set(by_id), 'Split membership does not cover candidates exactly')
    reserved = set(protected['ids'])
    reserved_groups = set(protected['group_ids'])
    require(set(protected['original_reserved_ids']) <= reserved, 'Original test IDs missing from closure')
    groups = collections.defaultdict(set)
    components = collections.defaultdict(set)
    diagnostic_ids = set()
    for row in read_jsonl(Path('data/external/external.jsonl')):
        diagnostic_ids.add(hashlib.sha256(normalize(row['text']).encode()).hexdigest())
    for row in rows:
        key = row['id']
        require(key == hashlib.sha256(normalize(row['text']).encode()).hexdigest(), 'Content ID mismatch')
        require(20 <= len(row['text'].split()) <= 150, 'Out-of-range word count')
        require(row.get('label') in (0, 1), 'Nonbinary candidate')
        require(bool(row.get('source')) and bool(row.get('group_ids')), 'Missing provenance')
        require(row.get('label_kind') in {'human', 'weak', 'synthetic'}, 'Missing supervision tier')
        require(row['backbone_fit_admitted'] == (row['split'] == 'fit'), 'Backbone fit flag includes held-out data')
        require(row['development_admitted'] == (row['split'] in {'fit', 'calibration', 'gate', 'validation'}),
                'Development flag disagrees with partition')
        require(key in split_sets.get(row['split'], set()), 'Row split disagrees with membership')
        origins = row.get('provenance', [])
        require(bool(origins), 'Original provenance missing')
        require(any(p.get('source') == row['source'] and p.get('label_kind') == row['label_kind']
                    and p.get('label') == row['label'] for p in origins), 'Selected provenance changed')
        if len({p.get('label') for p in origins}) > 1:
            require(not row['development_admitted'], 'Exact label conflict admitted')
            require('exact_duplicate_label_conflict' in row['quarantine_reasons'], 'Conflict reason missing')
        if row['development_admitted']:
            require(row['split'] in {'fit', 'calibration', 'gate', 'validation'}, 'Bad development partition')
            require(not row['quarantine_reasons'], 'Quarantined row admitted')
            require(key not in reserved and not set(row['group_ids']) & reserved_groups,
                    'Protected content or source group admitted')
            require(key not in diagnostic_ids, 'Existing diagnostic text admitted')
            require(row['source'] not in {'ibc_official_sample', 'mlburnham/dem_rep_party_platform_topics'},
                    'Diagnostic/quarantined source admitted')
            for group in row['group_ids']:
                groups[group].add(row['split'])
            components[row['group']].add(row['split'])
    require(all(len(parts) == 1 for parts in groups.values()), 'Provenance groups cross development partitions')
    require(all(len(parts) == 1 for parts in components.values()), 'Duplicate components cross development partitions')
    dev_ids = {r['id'] for r in rows if r['development_admitted']}
    require({r['id'] for r in read_jsonl(root / 'development.jsonl')} == dev_ids, 'Development file mismatch')
    for part, ids in split.items():
        actual = read_jsonl(root / f'{part}.jsonl')
        require(len(actual) == len(ids) and {r['id'] for r in actual} == set(ids), f'{part} file mismatch')
        require(len(ids) == report['partitions'][part], f'{part} report mismatch')
    require(len(dev_ids) == report['admitted_development'], 'Development count mismatch')
    original = set(protected['original_reserved_ids'])
    require(original <= set(by_id), 'Original reserved examples missing from candidates')
    require(all(not by_id[key]['development_admitted'] for key in original), 'Original final-test example admitted')
    inventory_roles = {
        'original-test-eligible': {key for key in original if by_id[key]['split'] == 'protected_test'},
        'original-test-quarantined': {key for key in original if by_id[key]['split'] != 'protected_test'},
        'new-weak-holdout': {row['id'] for row in rows if row['split'] == 'protected_test'
                             and row['source'] == 'Jacobvs/PoliticalTweets'},
        'test-closure-buffer': {row['id'] for row in rows if row['split'] == 'protected_test'
                               and row['id'] not in original and row['source'] != 'Jacobvs/PoliticalTweets'},
    }
    for name, expected in inventory_roles.items():
        actual = read_jsonl(root / f'{name}.jsonl')
        require(len(actual) == len(expected) and {r['id'] for r in actual} == expected,
                f'{name} inventory role mismatch')
    return {'passed': not failures, 'failures': sorted(set(failures)),
            'candidates_checked': len(rows), 'development_checked': len(dev_ids),
            'development_provenance_groups': len(groups), 'development_components': len(components),
            'original_reserved_ids': len(original),
            'original_reserved_by_current_partition': dict(collections.Counter(by_id[key]['split'] for key in original)),
            'protected_closure_ids': len(reserved),
            'test_inventory_roles': {name: len(ids) for name, ids in inventory_roles.items()},
            'protected_test_file_by_source': dict(collections.Counter(row['source'] for row in rows if row['split'] == 'protected_test')),
            'scope': 'Structural/provenance invariants only; no reserved-test scores, label accuracy claim, or independent duplicate-index recall guarantee.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check(args.directory)
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    raise SystemExit(0 if result['passed'] else 1)
