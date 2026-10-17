"""Build a reviewed, provenance-rich local dataset without changing historical splits.

Review decisions and source text stay under ignored data/. This script emits a
workbook payload, local provenance ledger, and aggregate report; the companion
JavaScript exporter writes the CSV after spreadsheet round-trip validation.
"""
from __future__ import annotations
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from possum.data import normalize, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'data/possum-v1-build'
OUT = ROOT / 'data/PoSSUMDatasetv1.0'
FACETS = (3, 4, 5, 6)
US_TOPICS = {
    'Inflation Reduction Act', 'CHIPS and Science Act', 'Capitol Hill Riot',
    'Abortion', 'Mass Shootings', 'George Floyd', 'Black Lives Matter',
    'Political Parties', 'Mexico–US Border',
}
COLUMNS = [
    'record_id', 'text', 'label', 'label_name', 'sub_dataset', 'quality_tier',
    'label_provenance', 'human_annotated', 'synthetic', 'original_labels_json',
    'label_mapping', 'topic', 'us_scope_basis', 'word_count', 'source_url',
    'source_revision', 'source_file_sha256', 'upstream_id', 'upstream_split',
    'group_ids_json', 'group_quality', 'partition', 'allowed_use',
    'previously_used', 'curation_method', 'curation_reason',
    'duplicate_cluster_id', 'provenance_json',
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cid(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def json_cell(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def clean(text):
    # Keep wording, punctuation, hashtags and quotations; no inferred context.
    return ' '.join(text.replace('\ufeff', '').split())


def review_key(row):
    # Content duplicates may carry distinct human facets; review each source row.
    return (row['id'], str(row.get('upstream_row', '')))


def load_decisions(paths, rows):
    decisions = {}
    for path in paths:
        for d in read_jsonl(path):
            key = review_key(d)
            if key in decisions:
                raise ValueError(f'Duplicate decision: {path}')
            if d['decision'] not in {'keep', 'exclude'} or not d.get('reason') or not d.get('reason_code'):
                raise ValueError(f'Incomplete semantic decision: {path}')
            decisions[key] = d
    expected = {review_key(r) for r in rows}
    assert len(expected) == len(rows), 'Source records need unique review keys'
    if set(decisions) != expected:
        raise ValueError(f'Review coverage mismatch: missing={len(expected-set(decisions))}; extra={len(set(decisions)-expected)}')
    return decisions


def lexical_edges(rows, references=None, cosine=.93, jaccard=.85):
    """Exhaustive sparse cosine candidate retrieval, with exact trigram backstop.

    Unlike LSH, this has no probabilistic candidate retrieval. Character TF-IDF
    catches punctuation and small edits; word trigrams catch copied wording.
    Neither claims semantic paraphrase detection.
    """
    same = references is None
    references = rows if same else references
    texts = [normalize(r['text']) for r in references]
    if not same:
        texts += [normalize(r['text']) for r in rows]
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=40000, dtype=np.float32)
    matrix = vectorizer.fit_transform(texts)
    ref = matrix[:len(references)]
    target = ref if same else matrix[len(references):]
    edges = set()
    for start in range(0, len(rows), 64):
        sim = (target[start:start+64] @ ref.T).tocoo()
        for a, b in zip(sim.row[sim.data >= cosine], sim.col[sim.data >= cosine]):
            i, j = start+int(a), int(b)
            if not same or i < j:
                edges.add((i, j))
    def shingles(text):
        words = normalize(text).split()
        return {' '.join(words[i:i+3]) for i in range(len(words)-2)}
    ref_sh = [shingles(r['text']) for r in references]
    inverted = collections.defaultdict(list)
    for j, sh in enumerate(ref_sh):
        for token in sh:
            inverted[token].append(j)
    for i, row in enumerate(rows):
        sh = ref_sh[i] if same else shingles(row['text'])
        overlap = collections.Counter(j for token in sh for j in inverted.get(token, ()))
        for j, common in overlap.items():
            if same and i >= j:
                continue
            denominator = len(sh) + len(ref_sh[j]) - common
            if denominator and common / denominator >= jaccard:
                edges.add((i, j))
    return sorted(edges)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected_path = ROOT/'data/assembled-v1/protected.json'
    protected = json.loads(protected_path.read_text())
    protected_ids, protected_groups = set(protected['ids']), set(protected['group_ids'])
    immutable = sorted((ROOT/'data/assembled-v1').glob('*.jsonl')) + [protected_path, ROOT/'data/assembled-v1/split.json']
    immutable += sorted((ROOT/'data/assembled-v2-no-manifesto').glob('*.jsonl'))
    hashes_before = {str(p.relative_to(ROOT)): digest(p) for p in immutable}
    mit_meta = json.loads((ROOT/'data/available/MITweet/download.json').read_text())
    mit_path = ROOT/'data/available/MITweet/MITweet.csv'
    assert digest(mit_path) == mit_meta['files']['MITweet.csv']
    with mit_path.open(newline='', encoding='utf-8-sig') as stream:
        mit_raw = list(csv.DictReader(stream))
    raw_facet_labels = collections.defaultdict(set)
    for raw in mit_raw:
        raw_facet_labels[cid(raw['tweet'])].update(int(raw[f'I{i}']) for i in FACETS if int(raw[f'I{i}']) != -1)
    raw_mitweet_conflicts = {key for key, values in raw_facet_labels.items() if len(values) > 1}
    mit = []
    prefilter = collections.Counter()
    for line, row in enumerate(mit_raw, 2):
        text = row['tweet']
        if row['topic'] not in US_TOPICS:
            prefilter['topic_outside_us_screen'] += 1
            continue
        if not 20 <= len(text.split()) <= 150:
            prefilter['word_count_outside_20_150'] += 1
            continue
        labels = {int(row[f'I{i}']) for i in FACETS if int(row[f'I{i}']) != -1}
        if len(labels) != 1 or labels == {1}:
            prefilter['neutral_unrelated_or_discordant_selected_facets'] += 1
            continue
        mit.append(dict(id=cid(text), text=text, label=0 if labels == {0} else 1,
                        topic=row['topic'], upstream_row=line,
                        original_labels={f'I{i}': int(row[f'I{i}']) for i in range(1, 13)},
                        mapped_facets=[f'I{i}' for i in FACETS if int(row[f'I{i}']) != -1]))
    assert len(mit) == 2141
    synthetic_path = ROOT/'data/synthetic.jsonl'
    syn_all = read_jsonl(synthetic_path)
    syn = [dict(r, id=cid(r['text'])) for r in syn_all if cid(r['text']) not in protected_ids]
    ibc = read_jsonl(ROOT/'data/assembled-v1/external_diagnostic.jsonl')
    assert all(r['source'] == 'ibc_official_sample' for r in ibc)
    all_candidates = read_jsonl(ROOT/'data/assembled-v1/candidates.jsonl')
    existing = {r['id']: r for r in all_candidates}
    roles = {}
    for part in ['fit', 'calibration', 'gate', 'validation']:
        for r in read_jsonl(ROOT/f'data/assembled-v2-no-manifesto/{part}.jsonl'):
            roles[r['id']] = part
    sources = {'MITweet': mit, 'GPT-4 Political Ideologies': syn, 'IBC public sample': ibc}
    decisions = {
        'MITweet': load_decisions([BUILD/f'mitweet-decisions-{i}.jsonl' for i in range(3)], mit),
        'GPT-4 Political Ideologies': load_decisions([BUILD/f'synthetic-decisions-{i}.jsonl' for i in range(3)], syn),
        'IBC public sample': load_decisions([BUILD/'ibc-decisions.jsonl'], ibc),
    }
    overrides_path = BUILD/'quality-overrides.jsonl'
    if overrides_path.exists():
        for override in read_jsonl(overrides_path):
            assert override['decision'] == 'exclude', 'Secondary audit may exclude, never relabel or reverse an exclusion'
            matches = [(ds, key) for ds in decisions.values() for key in ds if key[0] == override['id']]
            assert matches
            for ds, key in matches:
                ds[key] = override
    ibc_meta = json.loads((ROOT/'data/external/manifest.json').read_text())['ibc']
    candidate_labels = collections.defaultdict(set)
    for candidates in sources.values():
        for r in candidates:
            candidate_labels[r['id']].add(r['label'])
    pre_review_conflicts = {key for key, labels in candidate_labels.items() if len(labels) > 1}
    pre_review_conflicts.update(raw_mitweet_conflicts)
    records, excluded = [], []
    for name, candidates in sources.items():
        for r in candidates:
            if r['id'] in pre_review_conflicts:
                excluded.append(dict(id=r['id'], source=name, reason='raw_annotation_conflict_before_semantic_selection'))
                continue
            d = decisions[name][review_key(r)]
            if d['decision'] != 'keep':
                excluded.append(dict(id=r['id'], source=name, reason=d['reason_code']))
                continue
            text = clean(r['text'])
            if not 20 <= len(text.split()) <= 150:
                excluded.append(dict(id=r['id'], source=name, reason='cleaned_length'))
                continue
            base = dict(record_id=cid(text), text=text, label=r['label'], label_name=['Left', 'Right'][r['label']],
                        sub_dataset=name, word_count=len(text.split()), curation_method='automated_semantic_review_not_human_reannotation',
                        curation_reason=d['reason'], duplicate_cluster_id=cid(text))
            if name == 'MITweet':
                groups = ['mitweet:topic:' + re.sub(r'[^a-z0-9]+', '_', r['topic'].lower()).strip('_')]
                base.update(quality_tier='human_facet_derived', label_provenance='binary_derived_from_human_facet_annotations',
                    human_annotated=True, synthetic=False, original_labels_json=json_cell(r['original_labels']),
                    label_mapping='unanimous_noncenter_I3_I4_I5_I6; upstream_0_to_0_and_2_to_1; unrelated_ignored',
                    topic=r['topic'], us_scope_basis='US_event_topic_plus_passage_review; author_location_unknown',
                    source_url='https://github.com/LST1836/MITweet', source_revision=mit_meta['revision'],
                    source_file_sha256=mit_meta['files']['MITweet.csv'], upstream_id=f'csv_row_including_header:{r["upstream_row"]}',
                    upstream_split='unsplit_public_csv', group_quality='topic_proxy_author_and_thread_ids_unavailable',
                    partition='unassigned_development', allowed_use='development_after_group_split', previously_used=False)
            elif name == 'GPT-4 Political Ideologies':
                if r['id'] not in roles:
                    excluded.append(dict(id=r['id'], source=name, reason='not_previously_admitted_development'))
                    continue
                groups = r['group_ids']
                base.update(quality_tier='synthetic_supplement', label_provenance='synthetic_generation_label',
                    human_annotated=False, synthetic=True, original_labels_json=json_cell({'upstream_label': 1-r['label']}),
                    label_mapping='upstream_1_to_left_0_and_0_to_right_1', topic=groups[0].split(':issue:')[-1],
                    us_scope_basis='explicit_US_compatible_policy_viewpoint_review; generation_location_not_applicable',
                    source_url='https://huggingface.co/datasets/JyotiNayak/political_ideologies', source_revision='unrecorded_upstream_revision_local_snapshot_hashed',
                    source_file_sha256=digest(synthetic_path), upstream_id=str(r['upstream_id']), upstream_split=r['upstream_split'],
                    group_quality=r['group_quality'], partition=roles[r['id']], allowed_use=f'preserve_existing_{roles[r["id"]]}_role', previously_used=True)
            else:
                groups = r['group_ids']
                base.update(quality_tier='human_diagnostic_only', label_provenance='human_sentence_ideology',
                    human_annotated=True, synthetic=False, original_labels_json=json_cell({'root_label': ['Liberal','Conservative'][r['label']]}),
                    label_mapping='Liberal_to_0_Conservative_to_1; Neutral_excluded', topic='mixed_book_sentences',
                    us_scope_basis='US_or_general_ideological_argument_passage_review; book_author_ids_unavailable',
                    source_url=ibc_meta['url'], source_revision='unversioned_public_sample_archive_hashed',
                    source_file_sha256=ibc_meta['sha256'], upstream_id=str(r['upstream_id']), upstream_split='public_sample',
                    group_quality=r['group_quality'], partition='external_diagnostic', allowed_use='previously_seen_diagnostic_only_never_train', previously_used=True)
            if r['id'] in protected_ids or protected_groups.intersection(groups):
                excluded.append(dict(id=r['id'], source=name, reason='protected_id_or_group'))
                continue
            base['group_ids_json'] = json_cell(groups)
            base['provenance_json'] = json_cell([dict(source=name, source_url=base['source_url'], revision=base['source_revision'],
                source_file_sha256=base['source_file_sha256'], upstream_id=base['upstream_id'], upstream_split=base['upstream_split'],
                original_text_sha256=hashlib.sha256(r['text'].encode()).hexdigest(), original_labels=json.loads(base['original_labels_json']))])
            records.append(base)
    # Compare against every protected row and all previously exposed diagnostics.
    refs = {r['id']: dict(text=r['text'], id=r['id'], kind='protected') for r in all_candidates if r['id'] in protected_ids}
    assert set(refs) == protected_ids, 'Every protected hash needs source text for leakage checks'
    for r in read_jsonl(ROOT/'data/external/external.jsonl'):
        refs.setdefault(cid(r['text']), dict(text=r['text'], id=cid(r['text']), kind='previous_diagnostic'))
    for r in read_jsonl(ROOT/'data/assembled-v1/development.jsonl'):
        refs.setdefault(r['id'], dict(text=r['text'], id=r['id'], kind='previous_development', partition=r['split']))
    for candidates in sources.values():
        for r in candidates:
            if r['id'] in pre_review_conflicts:
                refs.setdefault(r['id'], dict(text=r['text'], id=r['id'], kind='raw_annotation_conflict'))
                refs[r['id']]['annotation_conflict'] = True
    references = list(refs.values())
    print(f'Checking {len(records)} reviewed passages against {len(references)} protected/historically used passages', flush=True)
    blocked = set()
    conflict_reference_hits = set()
    reference_edges = lexical_edges(records, references)
    for i, j in reference_edges:
        r, ref = records[i], references[j]
        if ref.get('annotation_conflict'):
            conflict_reference_hits.add(i)
        if ref['kind'] == 'raw_annotation_conflict':
            continue
        if r['partition'] == 'external_diagnostic' and ref['kind'] == 'previous_diagnostic':
            continue
        if r['sub_dataset'] == 'GPT-4 Political Ideologies' and ref['kind'] == 'previous_development' and r['partition'] == ref['partition']:
            continue
        blocked.add(i)
    # Discover connectivity BEFORE removing a direct hit, otherwise its near-copy
    # could survive as an indirect bridge into protected data.
    edges = lexical_edges(records)
    parent = list(range(len(records)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in edges:
        parent[root(j)] = root(i)
    clusters = collections.defaultdict(list)
    for i in range(len(records)):
        clusters[root(i)].append(i)
    # Leakage connectivity also propagates through declared independence groups;
    # duplicate connectivity remains separate so distinct topic rows are retained.
    leak_parent = parent.copy()
    def leak_root(i):
        while leak_parent[i] != i:
            leak_parent[i] = leak_parent[leak_parent[i]]
            i = leak_parent[i]
        return i
    group_owner = {}
    for i, r in enumerate(records):
        for group in json.loads(r['group_ids_json']):
            if group in group_owner:
                leak_parent[leak_root(i)] = leak_root(group_owner[group])
            else:
                group_owner[group] = i
    blocked_components = {leak_root(i) for i in blocked}
    blocked = {i for i in range(len(records)) if leak_root(i) in blocked_components}
    final = []
    rank = {'human_facet_derived': 0, 'human_diagnostic_only': 1, 'synthetic_supplement': 2}
    for ids in clusters.values():
        members = [records[i] for i in ids]
        if any(i in blocked for i in ids):
            for r in members:
                excluded.append(dict(id=r['record_id'], source=r['sub_dataset'], reason='protected_or_incompatible_history_component'))
            continue
        if any(i in conflict_reference_hits for i in ids) or len({r['label'] for r in members}) > 1:
            for r in members:
                excluded.append(dict(id=r['record_id'], source=r['sub_dataset'], reason='duplicate_cluster_label_conflict'))
            continue
        historical_roles = {r['partition'] for r in members if r['previously_used']}
        if len(historical_roles) > 1 or ('external_diagnostic' in historical_roles and len(members) > 1):
            for r in members:
                excluded.append(dict(id=r['record_id'], source=r['sub_dataset'], reason='duplicate_cluster_partition_conflict'))
            continue
        chosen = min(members, key=lambda r: (rank[r['quality_tier']], r['record_id']))
        # A newly sourced copy cannot evade an existing development assignment.
        if historical_roles:
            inherited_role = next(iter(historical_roles))
            if chosen['partition'] != inherited_role:
                chosen['partition'] = inherited_role
                chosen['allowed_use'] = f'preserve_existing_{inherited_role}_role'
            chosen['previously_used'] = True
        chosen['duplicate_cluster_id'] = min(r['record_id'] for r in members)
        chosen['group_ids_json'] = json_cell(sorted({g for r in members for g in json.loads(r['group_ids_json'])}))
        chosen['provenance_json'] = json_cell([p for r in members for p in json.loads(r['provenance_json'])])
        final.append(chosen)
        for r in members:
            if r is not chosen:
                excluded.append(dict(id=r['record_id'], source=r['sub_dataset'], reason='duplicate_collapsed_provenance_preserved'))
    final.sort(key=lambda r: (r['sub_dataset'], r['label'], r['record_id']))
    assert len({r['record_id'] for r in final}) == len(final)
    assert not protected_ids.intersection(r['record_id'] for r in final)
    assert all(r['sub_dataset'] != 'Manifesto Project' for r in final)
    for r in final:
        assert set(r) == set(COLUMNS), (set(COLUMNS)-set(r), set(r)-set(COLUMNS))
        assert not protected_groups.intersection(json.loads(r['group_ids_json']))
        assert r['label'] in (0, 1) and 20 <= r['word_count'] <= 150
        assert '\x00' not in r['text'] and not any(ord(c) < 32 for c in r['text'])
    hashes_after = {str(p.relative_to(ROOT)): digest(p) for p in immutable}
    assert hashes_before == hashes_after, 'Immutable historical inputs changed'
    for r in final:
        for field in ['human_annotated', 'synthetic', 'previously_used']:
            r[field] = str(r[field]).lower()
    payload = dict(columns=COLUMNS, rows=[[r[k] for k in COLUMNS] for r in final])
    (OUT/'workbook-payload.json').write_text(json.dumps(payload, ensure_ascii=False))
    (OUT/'exclusions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in excluded))
    decision_paths = sorted(BUILD.glob('*decisions*.jsonl')) + ([overrides_path] if overrides_path.exists() else [])
    input_hashes = {str(p.relative_to(ROOT)): digest(p) for p in decision_paths + [mit_path, synthetic_path, protected_path, ROOT/'data/external/sample_ibc.tar.gz']}
    summary = dict(
        version='PoSSUMDatasetv1.0', rows=len(final), label_mapping={'0':'Left','1':'Right'},
        source_counts=dict(collections.Counter(r['sub_dataset'] for r in final)),
        source_label_counts=dict(collections.Counter(f'{r["sub_dataset"]}:{r["label_name"]}' for r in final)),
        quality_tiers=dict(collections.Counter(r['quality_tier'] for r in final)),
        partitions=dict(collections.Counter(r['partition'] for r in final)),
        mitweet_topic_counts=dict(collections.Counter(r['topic'] for r in final if r['sub_dataset']=='MITweet')),
        candidate_rows={k:len(v) for k,v in sources.items()},
        semantic_review_kept={k:sum(d['decision']=='keep' for d in v.values()) for k,v in decisions.items()},
        mitweet_raw_rows=len(mit_raw), mitweet_prefilter_exclusions=dict(prefilter),
        exclusions=dict(collections.Counter(f'{r["source"]}:{r["reason"]}' for r in excluded)),
        duplicate_edges=len(edges), reference_edges_checked=len(reference_edges), protected_ids_checked=len(protected_ids),
        source_input_sha256=input_hashes, immutable_historical_inputs_sha256=hashes_before,
        historical_reserved_review_exposure_rows=13,
        checks={'all_candidate_rows_semantically_reviewed':True, 'historical_inputs_unchanged':True,
                'no_protected_ids_or_groups_exported':True, 'csv_roundtrip_verified':False},
        text_cleaning='Whitespace normalization and removal of BOM only; no sentence completion, label correction, or paraphrasing.',
        duplicate_method='Exhaustive sparse character TF-IDF cosine >=0.93 OR exact word-trigram Jaccard >=0.85; transitive clusters; not semantic paraphrase detection.',
        excluded_sources={
            'Manifesto Project':'Policy-category/party mappings can label contextless fragments; excluded from this version.',
            'Political Tweets':'Author-party proxy labels lack passage-level verification; excluded from this version.',
            'Party Platform Topics':'Party-derived labels and missing document boundaries; excluded from this version.',
            'GPT-4 Political Bias':'All available nonneutral rows fall below the 20-word minimum; remaining length-eligible rows are neutral.',
            'Full IBC':'Not available; only the public sample is locally accessible.',
            'Qbias and SCCC':'No admitted passage-level binary source with complete usable text and provenance available locally.'},
        limitations=[
            'Semantic screening is automated, not new human adjudication. Labels were never changed to fit the screening model.',
            'MITweet binary labels derive from selected human facets, not direct human overall-US-left/right annotations.',
            'MITweet is topic-selected and has no recoverable author/thread identifiers; US relevance does not establish author nationality.',
            'MITweet development rows require a group-aware split before training; there is no newly certified unseen gold test set.',
            'Synthetic records retain their historical fit/calibration/gate/validation roles and must not be pooled across them.',
            'Thirteen historical reserved synthetic passages were source-audited; they remain excluded and cannot be described as unseen. No historical model fitting or scoring was rerun.',
            'IBC is a selected previously exposed diagnostic subset, never training data or an unbiased final performance estimate.',
            'Source class/topic imbalance is retained transparently, without oversampling or synthetic duplication.',
            'Public availability does not establish redistribution rights; source texts are local-only and excluded from Git.',
        ])
    (OUT/'manifest.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['rows','source_counts','source_label_counts','partitions','mitweet_topic_counts']}, indent=2))


if __name__ == '__main__':
    main()
