"""Assemble provenance-preserving research splits without releasing source text.

Existing test/document closure is immutable. New lexical duplicate discovery uses
MinHash LSH followed by exact word-trigram Jaccard, not an all-pairs matrix.
"""
import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from possum.data import normalize, prepare, read_jsonl


def content_id(row):
    return hashlib.sha256(normalize(row['text']).encode()).hexdigest()


class Union:
    def __init__(self, n):
        self.parent = list(range(n))

    def root(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, i, j):
        a, b = self.root(i), self.root(j)
        self.parent[max(a, b)] = min(a, b)


def write_rows(path, rows):
    with path.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def legacy_owners():
    owners, test_ids, hashes = {}, set(), {}
    for input_name, split_name in [
        ('data/frozen-four-pilot.jsonl', 'runs/frozen-four-42/split.json'),
        ('data/synthetic.jsonl', 'runs/synthetic-42/split.json'),
    ]:
        membership = json.loads(Path(split_name).read_text())['full']
        by_id = {key: part for part, ids in membership.items() for key in ids}
        found = set()
        for row in read_jsonl(input_name):
            key = content_id(row)
            if key not in by_id:
                continue
            found.add(key)
            for group in row['group_ids']:
                previous = owners.setdefault(group, by_id[key])
                if previous != by_id[key]:
                    raise ValueError('Legacy source group crosses partitions')
        if set(by_id) - found:
            raise ValueError('Missing original split IDs')
        test_ids.update(membership['test'])
        for name in [input_name, split_name]:
            hashes[name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    return owners, test_ids, hashes


def shingles(text):
    tokens = normalize(text).split()
    return np.unique(np.array([
        int.from_bytes(hashlib.blake2s(' '.join(tokens[i:i+3]).encode(), digest_size=4).digest(), 'little')
        for i in range(len(tokens)-2)
    ], dtype=np.uint64))


def near_edges(rows, union, reasons, threshold=.85, bucket_limit=200):
    rng = np.random.default_rng(42)
    prime = np.uint64(4294967311)
    aa = rng.integers(1, 2**32, size=64, dtype=np.uint64)
    bb = rng.integers(0, 2**32, size=64, dtype=np.uint64)
    buckets, previous = {}, []
    checks = edges = conflicts = crowded = 0
    for i, row in enumerate(rows):
        sh = shingles(row['text'])
        signature = ((aa[:, None] * sh[None, :] + bb[:, None]) % prime).min(axis=1)
        keys = [(band, signature[band*4:band*4+4].tobytes()) for band in range(16)]
        candidates = set()
        for key in keys:
            bucket = buckets.get(key, [])
            if len(bucket) >= bucket_limit:
                crowded += 1
                reasons[i].add('crowded_lsh_bucket_manual_review')
                for j in bucket:
                    reasons[j].add('crowded_lsh_bucket_manual_review')
                continue
            candidates.update(bucket)
        for j in candidates:
            other = previous[j]
            if min(len(sh), len(other)) / max(len(sh), len(other)) < threshold:
                continue
            checks += 1
            intersection = len(np.intersect1d(sh, other, assume_unique=True))
            similarity = intersection / (len(sh)+len(other)-intersection)
            if similarity < threshold:
                continue
            edges += 1
            # Quarantined party-platform rows cannot connect every document via
            # their source-wide placeholder group. Their known diagnostic copies
            # still block development reuse of the corresponding passage.
            if row['tier'] != 'platform_quarantine' and rows[j]['tier'] != 'platform_quarantine':
                union.join(i, j)
            if row['label'] != rows[j]['label']:
                conflicts += 1
                reasons[i].add('near_duplicate_label_conflict')
                reasons[j].add('near_duplicate_label_conflict')
            if row.get('previously_diagnostic') or rows[j].get('previously_diagnostic'):
                reasons[i].add('near_previous_diagnostic')
                reasons[j].add('near_previous_diagnostic')
        previous.append(sh)
        for key in keys:
            bucket = buckets.setdefault(key, [])
            if len(bucket) < bucket_limit:
                bucket.append(i)
        if (i+1) % 10000 == 0:
            print(f'Near-duplicate indexing: {i+1}/{len(rows)}', flush=True)
    return dict(candidate_pairs_verified=checks, verified_edges=edges,
                conflicting_edges=conflicts, crowded_bucket_events=crowded,
                method='64-permutation MinHash, 16 bands of 4; exact hashed word-trigram Jaccard >= 0.85',
                limitations='Probabilistic candidate recall; not semantic paraphrase detection. Crowded buckets quarantine rather than silently skip. Existing Manifesto char-TFIDF 0.93 closure is additionally retained.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/assembled-v1')
    parser.add_argument('--tweet-admission', default='data/tweet-admission-v1/eligible.jsonl')
    parser.add_argument('--skip-near', action='store_true', help='Smoke test only; output is not training-admitted')
    args = parser.parse_args()
    start = time.time()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not Path(args.tweet_admission).exists():
        raise ValueError('Tweet admission audit must finish before assembly')
    tweet_admission = {content_id(row): row for row in read_jsonl(args.tweet_admission)}
    owners, test_ids, hashes = legacy_owners()
    hashes[args.tweet_admission] = hashlib.sha256(Path(args.tweet_admission).read_bytes()).hexdigest()
    inputs = [
        ('data/manifesto/candidates.jsonl', 'policy_mapping'),
        ('data/synthetic.jsonl', 'synthetic'),
        ('data/available/mlburnham_dem_rep_party_platform_topics/candidates.jsonl', 'platform_quarantine'),
        ('data/available/Jacobvs_PoliticalTweets/candidates.jsonl', 'tweet_candidate'),
        ('data/external/external.jsonl', 'diagnostic'),
    ]
    diagnostic_ids = {content_id(row) for row in read_jsonl('data/external/external.jsonl')}
    raw = collections.defaultdict(list)
    raw_counts, excluded = collections.Counter(), collections.Counter()
    for name, tier in inputs:
        hashes[name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for original in read_jsonl(name):
            if tier == 'diagnostic' and original['source'] != 'ibc_official_sample':
                continue
            raw_counts[original['source']] += 1
            if original.get('label') not in (0, 1) or original.get('excluded'):
                excluded['nonbinary_or_excluded'] += 1
                continue
            if not 20 <= len(original['text'].split()) <= 150:
                excluded['length'] += 1
                continue
            if original.get('label_kind') not in ('human', 'weak', 'synthetic') or not original.get('group_ids'):
                raise ValueError('Invalid provenance record')
            row = dict(original, tier=tier)
            key = content_id(row)
            if tier == 'tweet_candidate' and key in tweet_admission:
                row.update(tweet_admission[key])
                row['tier'] = 'heuristic_weak'
            raw[key].append(row)
    priority = {'policy_mapping': 0, 'synthetic': 1, 'heuristic_weak': 2, 'tweet_candidate': 3, 'diagnostic': 4, 'platform_quarantine': 5}
    rows, reasons = [], []
    for key, copies in sorted(raw.items()):
        copies.sort(key=lambda row: priority[row['tier']])
        row = dict(copies[0], id=key)
        # Preserve complete original provenance independently of effective split
        # groups; an unavailable platform document ID cannot merge all platforms.
        row['provenance'] = [{k: v for k, v in copy.items() if k != 'text'} for copy in copies]
        row['sources'] = sorted({copy['source'] for copy in copies})
        active = [copy for copy in copies if copy['tier'] != 'platform_quarantine'] or copies
        row['group_ids'] = sorted({g for copy in active for g in copy['group_ids']})
        row['previously_diagnostic'] = key in diagnostic_ids
        why = set()
        if len({copy['label'] for copy in copies}) > 1:
            why.add('exact_duplicate_label_conflict')
        if row['tier'] == 'platform_quarantine':
            why.add('platform_document_ids_missing')
        if row['tier'] == 'tweet_candidate':
            why.add('tweet_politicalness_or_stance_unverified')
        if key in diagnostic_ids:
            why.add('previously_seen_external_diagnostic')
        rows.append(row)
        reasons.append(why)
    del raw
    union = Union(len(rows))
    by_id = {row['id']: i for i, row in enumerate(rows)}
    group_owner = {}
    for i, row in enumerate(rows):
        if row['tier'] == 'platform_quarantine':
            continue
        for group in row['group_ids']:
            if group in group_owner:
                union.join(i, group_owner[group])
            group_owner[group] = i
    print('Retaining original Manifesto char-TFIDF duplicate closure', flush=True)
    legacy_manifesto = prepare(read_jsonl('data/manifesto/candidates.jsonl'))
    old_clusters = collections.defaultdict(list)
    for row in legacy_manifesto:
        old_clusters[row['group']].append(by_id[row['id']])
    for members in old_clusters.values():
        for i in members[1:]:
            union.join(members[0], i)
    near = {'status': 'skipped; no training admission'} if args.skip_near else near_edges(rows, union, reasons)
    clusters = collections.defaultdict(list)
    for i in range(len(rows)):
        clusters[union.root(i)].append(i)
    split_ids = collections.defaultdict(list)
    protected_groups, protected_ids = set(), set()
    partition_clashes = 0
    for members in clusters.values():
        groups = {g for i in members for g in rows[i]['group_ids']}
        parts = {owners[g] for g in groups if g in owners}
        group_id = min(rows[i]['id'] for i in members)
        if 'test' in parts or any(rows[i]['id'] in test_ids for i in members):
            part = 'protected_test'
            protected_groups.update(groups)
            protected_ids.update(rows[i]['id'] for i in members)
        elif len(parts) > 1:
            part = 'quarantine'
            partition_clashes += 1
            for i in members:
                reasons[i].add('legacy_partition_collision')
        elif parts:
            part = next(iter(parts))
        else:
            # Approximately 80/20 by connected source group, never by passage.
            u = int(hashlib.sha256(('split42:'+group_id).encode()).hexdigest()[:16], 16) / 2**64
            part = 'protected_test' if u < .2 else 'validation' if u < .32 else 'gate' if u < .456 else 'calibration' if u < .592 else 'fit'
            if part == 'protected_test':
                protected_groups.update(groups)
                protected_ids.update(rows[i]['id'] for i in members)
        for i in members:
            row = rows[i]
            row['group'] = group_id
            row['assigned_partition'] = part
            row['quarantine_reasons'] = sorted(reasons[i])
            row['split'] = 'external_diagnostic' if row['tier'] == 'diagnostic' else 'quarantine' if reasons[i] else part
            if args.skip_near:
                row['split'] = 'quarantine'
                row['quarantine_reasons'].append('near_duplicate_audit_not_run')
            row['development_admitted'] = row['split'] in ('fit', 'calibration', 'gate', 'validation')
            row['backbone_fit_admitted'] = row['split'] == 'fit'
            split_ids[row['split']].append(row['id'])
    # Enforce every active source group has a unique effective partition. Reserved
    # examples may be quarantined for conflicts but never return to development.
    assigned = collections.defaultdict(set)
    for row in rows:
        if row['development_admitted']:
            assert row['id'] not in protected_ids
            assert not set(row['group_ids']) & protected_groups
            for group in row['group_ids']:
                assigned[group].add(row['split'])
    assert all(len(parts) == 1 for parts in assigned.values())
    assert test_ids <= protected_ids
    write_rows(output/'candidates.jsonl', rows)
    for part, ids in split_ids.items():
        wanted = set(ids)
        write_rows(output/f'{part}.jsonl', (row for row in rows if row['id'] in wanted))
    admitted = [row for row in rows if row['development_admitted']]
    write_rows(output/'development.jsonl', admitted)
    write_rows(output/'core-development.jsonl', (row for row in admitted if row['tier'] in ('policy_mapping', 'synthetic')))
    original_eligible = [row for row in rows if row['id'] in test_ids and row['split'] == 'protected_test']
    original_quarantined = [row for row in rows if row['id'] in test_ids and row['split'] != 'protected_test']
    new_weak_holdout = [row for row in rows if row['split'] == 'protected_test' and row['tier'] == 'heuristic_weak' and row['id'] not in test_ids]
    closure_buffer = [row for row in rows if row['split'] == 'protected_test' and row['id'] not in test_ids and row['tier'] != 'heuristic_weak']
    write_rows(output/'original-test-eligible.jsonl', original_eligible)
    write_rows(output/'original-test-quarantined.jsonl', original_quarantined)
    write_rows(output/'new-weak-holdout.jsonl', new_weak_holdout)
    write_rows(output/'test-closure-buffer.jsonl', closure_buffer)
    protected = dict(ids=sorted(protected_ids), group_ids=sorted(protected_groups), original_reserved_ids=sorted(test_ids))
    (output/'protected.json').write_text(json.dumps(protected, indent=2)+'\n')
    (output/'split.json').write_text(json.dumps(dict(split_ids), indent=2)+'\n')
    per_source = {}
    for source in sorted({row['source'] for row in rows}):
        source_rows = [row for row in rows if row['source'] == source]
        per_source[source] = dict(unique_rows=len(source_rows), splits=dict(collections.Counter(row['split'] for row in source_rows)),
            development_groups=len({row['group'] for row in source_rows if row['development_admitted']}))
    report = dict(version='assembled-v1', raw_counts=dict(raw_counts), input_rows=sum(raw_counts.values()),
        unique_candidates=len(rows), exact_duplicate_occurrences=sum(raw_counts.values())-len(rows), initial_exclusions=dict(excluded),
        partitions={part:len(ids) for part,ids in split_ids.items()}, admitted_development=len(admitted),
        development_by_tier=dict(collections.Counter(row['tier'] for row in admitted)),
        development_by_partition_and_label={part:dict(collections.Counter(str(row['label']) for row in admitted if row['split']==part)) for part in ('fit','calibration','gate','validation')},
        quarantine_reasons=dict(collections.Counter(why for row in rows for why in row['quarantine_reasons'])),
        legacy_partition_collision_components=partition_clashes, source_summary=per_source, near_duplicates=near,
        protected=dict(original_reserved_ids=len(test_ids), closure_ids=len(protected_ids), closure_group_ids=len(protected_groups), invariant='No protected ID or source group admitted to development'),
        test_inventory=dict(original_test_eligible=len(original_eligible), original_test_quarantined=len(original_quarantined),
            duplicate_closure_buffer=len(closure_buffer), new_author_group_weak_holdout=len(new_weak_holdout),
            warning='protected_test.jsonl is a protection inventory, not a replacement final test. Keep original IDs and report quarantined original IDs explicitly. Closure buffers must not be scored as fresh final examples; new weak tweet holdout is not human-verified.'),
        input_sha256=hashes, elapsed_seconds=time.time()-start,
        interpretation=['Policy labels are weak mappings of human policy annotations, not human binary ideology judgments.',
          'Tweets remain author-party weak supervision even after heuristic policy/stance filtering; neither politicalness nor viewpoint labels are human verified.',
          'Synthetic issue groups are proxies; missing generation-batch provenance limits independence.',
          'Prior test/document closures and all unambiguous legacy development partitions are retained. Joined legacy development partitions are quarantined.',
          'IBC is previously seen diagnostic-only; platform rows lack document IDs and are quarantined.',
          'Group holdouts target 80/20 for new groups; row ratios differ. No reserved test scores or labels are summarized.',
          'Use explicit split files: only fit is for backbone adaptation; calibration, gate and validation are separate. Do not re-split development.jsonl.',
          'Research admission does not establish permission to redistribute source text or commercial model weights.'])
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    Path('reports/assembled-corpus.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2), flush=True)


if __name__ == '__main__':
    main()
