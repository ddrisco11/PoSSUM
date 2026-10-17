"""Derive new development inputs without policy-mapped Manifesto supervision.

Preserve all historical partition assignments and protected test identities.
No model training or test scoring is performed.
"""
import argparse
import collections
import hashlib
import json
import shutil
from pathlib import Path

from check_assembled_corpus import check
from possum.data import read_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('data/assembled-v1'))
    p.add_argument('--output', type=Path, default=Path('data/assembled-v2-no-manifesto'))
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Choose a new output directory; existing datasets are immutable')
    assert check(a.source)['passed'], 'Parent assembly failed verification'
    rows = read_jsonl(a.source / 'candidates.jsonl')
    removed = collections.Counter()
    for r in rows:
        if r['development_admitted'] and any('manifesto' in s.lower() for s in r['sources']):
            removed[r['split']] += 1
            r['quarantine_reasons'].append('policy_category_not_standalone_viewpoint')
            r.update(split='quarantine', development_admitted=False, backbone_fit_admitted=False)
    a.output.mkdir(parents=True)
    # Keep every non-development inventory and protection record intact.
    for f in a.source.glob('*.json*'):
        shutil.copyfile(f, a.output / f.name)
    parts = collections.defaultdict(list)
    for r in rows:
        parts[r['split']].append(r)
    def write(name, records):
        with (a.output / name).open('w') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    write('candidates.jsonl', rows)
    for part, records in parts.items():
        write(part + '.jsonl', records)
    dev = [r for r in rows if r['development_admitted']]
    write('development.jsonl', dev)
    write('core-development.jsonl', [r for r in dev if r['tier'] == 'synthetic'])
    split = {part: [r['id'] for r in records] for part, records in parts.items()}
    (a.output / 'split.json').write_text(json.dumps(split, indent=2) + '\n')
    report = json.loads((a.source / 'manifest.json').read_text())
    report.update(version='assembled-v2-no-manifesto', parent=str(a.source),
                  parent_manifest_sha256=hashlib.sha256((a.source / 'manifest.json').read_bytes()).hexdigest(),
                  excluded_from_development=dict(removed),
                  partitions={k: len(v) for k, v in parts.items()}, admitted_development=len(dev),
                  development_by_tier=dict(collections.Counter(r['tier'] for r in dev)),
                  development_by_partition_and_label={part: dict(collections.Counter(str(r['label']) for r in records))
                      for part, records in parts.items() if part in ('fit', 'calibration', 'gate', 'validation')},
                  quarantine_reasons=dict(collections.Counter(reason for r in rows for reason in r['quarantine_reasons'])))
    for source, summary in report['source_summary'].items():
        sr = [r for r in rows if r['source'] == source]
        summary.update(splits=dict(collections.Counter(r['split'] for r in sr)),
                       development_groups=len({r['group'] for r in sr if r['development_admitted']}))
    report['interpretation'].append('Manifesto policy mappings are excluded from all new development roles because isolated passages may not express a self-contained ideological position. Historical metrics retain their original inputs; no causal attribution is implied.')
    (a.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    verification = check(a.output)
    assert verification['passed'], verification
    for name in ('protected.json', 'protected_test.jsonl', 'original-test-eligible.jsonl',
                 'original-test-quarantined.jsonl', 'new-weak-holdout.jsonl', 'test-closure-buffer.jsonl'):
        assert (a.source / name).read_bytes() == (a.output / name).read_bytes(), name
    summary = dict(version=report['version'], excluded_from_development=dict(removed),
                   development_partitions={k: len(parts[k]) for k in ('fit', 'calibration', 'gate', 'validation')},
                   admitted_development=len(dev), protected_inventories_byte_identical=True,
                   verification=verification, training_started=False)
    Path('reports/manifesto-exclusion.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
