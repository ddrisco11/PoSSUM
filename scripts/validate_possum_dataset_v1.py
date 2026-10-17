"""Verify the exported CSV, full lineage, labels and immutable historical roles."""
import collections
import csv
import json
import hashlib
import io
import tarfile
from pathlib import Path
from prepare_possum_dataset_v1 import ROOT, OUT, COLUMNS, cid, clean, digest, lexical_edges
from possum.data import read_jsonl


def main():
    manifest = json.loads((OUT/'manifest.json').read_text())
    path = OUT/'PoSSUMDatasetv1.0.csv'
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == COLUMNS
        rows = list(reader)
    payload = json.loads((OUT/'workbook-payload.json').read_text())
    assert [[r[k] for k in COLUMNS] for r in rows] == [[str(v) for v in r] for r in payload['rows']]
    assert len(rows) == manifest['rows']
    assert dict(collections.Counter(r['sub_dataset'] for r in rows)) == manifest['source_counts']
    protected = json.loads((ROOT/'data/assembled-v1/protected.json').read_text())
    forbidden = set(protected['ids'])
    blocked_groups = set(protected['group_ids'])
    roles = {}
    for part in ['fit', 'calibration', 'gate', 'validation']:
        for r in read_jsonl(ROOT/f'data/assembled-v2-no-manifesto/{part}.jsonl'):
            roles[r['id']] = part
    source_rows = {}
    with (ROOT/'data/available/MITweet/MITweet.csv').open(encoding='utf-8-sig', newline='') as f:
        for line, raw in enumerate(csv.DictReader(f), 2):
            source_rows[('MITweet',f'csv_row_including_header:{line}','unsplit_public_csv')] = (
                raw['tweet'], {f'I{i}':int(raw[f'I{i}']) for i in range(1,13)})
    for raw in read_jsonl(ROOT/'data/synthetic.jsonl'):
        source_rows[('GPT-4 Political Ideologies',str(raw['upstream_id']),raw['upstream_split'])] = (
            raw['text'], {'upstream_label':1-raw['label']})
    from fetch_external_eval import RestrictedTreeReader, tree_words
    with tarfile.open(ROOT/'data/external/sample_ibc.tar.gz') as archive:
        trees = RestrictedTreeReader(io.BytesIO(archive.extractfile('sampleData.pkl').read()), encoding='latin1').load()
        for label, group in enumerate(trees[:2]):
            for index, tree in enumerate(group):
                assert tree.label == ['Liberal','Conservative'][label]
                source_rows[('IBC public sample',f'{label}:{index}','public_sample')] = (
                    ' '.join(tree_words(tree)), {'root_label':tree.label})
    assert len({r['record_id'] for r in rows}) == len(rows)
    for r in rows:
        assert r['record_id'] == cid(r['text'])
        assert r['record_id'] not in forbidden
        assert int(r['word_count']) == len(r['text'].split())
        assert 20 <= int(r['word_count']) <= 150
        assert r['label'] in {'0','1'} and r['label_name'] == ['Left','Right'][int(r['label'])]
        assert not blocked_groups.intersection(json.loads(r['group_ids_json']))
        assert r['source_url'].startswith('https://') and len(r['source_file_sha256']) == 64
        assert r['curation_method'] == 'automated_semantic_review_not_human_reannotation'
        assert r['curation_reason']
        lineage = json.loads(r['provenance_json'])
        assert lineage and all(p['upstream_id'] and p['source_file_sha256'] and p['original_labels'] for p in lineage)
        primary_found = False
        for p in lineage:
            original, labels = source_rows[(p['source'],p['upstream_id'],p['upstream_split'])]
            assert hashlib.sha256(original.encode()).hexdigest() == p['original_text_sha256']
            assert labels == p['original_labels']
            if p['source'] == r['sub_dataset'] and p['upstream_id'] == r['upstream_id'] and p['upstream_split'] == r['upstream_split']:
                assert clean(original) == r['text']
                assert json.loads(r['original_labels_json']) == labels
                primary_found = True
        assert primary_found, 'Primary source row must reconstruct exported text and labels'
        if r['sub_dataset'] == 'IBC public sample':
            assert r['partition'] == 'external_diagnostic'
            assert r['allowed_use'] == 'previously_seen_diagnostic_only_never_train'
        if r['sub_dataset'] == 'GPT-4 Political Ideologies':
            assert r['partition'] == roles[r['record_id']]
            assert r['synthetic'] == 'true' and r['human_annotated'] == 'false'
        if r['sub_dataset'] == 'MITweet':
            labels = json.loads(r['original_labels_json'])
            selected = {labels[f'I{i}'] for i in [3,4,5,6] if labels[f'I{i}'] != -1}
            assert selected == ({0} if r['label'] == '0' else {2})
            assert r['human_annotated'] == 'true' and r['label_provenance'] == 'binary_derived_from_human_facet_annotations'
    assert not lexical_edges(rows), 'Lexical near duplicates remain in export'
    for name, sha in manifest['immutable_historical_inputs_sha256'].items():
        assert digest(ROOT/name) == sha, f'Historical input mutated: {name}'
    for name, sha in manifest['source_input_sha256'].items():
        assert digest(ROOT/name) == sha, f'Source input mutated: {name}'
    manifest['checks'].update(csv_roundtrip_verified=True, provenance_validated=True, source_text_and_original_labels_reconstructed=True,
        final_csv_no_lexical_near_duplicates=True, original_role_assignments_preserved=True)
    manifest['csv_sha256'] = digest(path)
    manifest['csv_bytes'] = path.stat().st_size
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    # Aggregate report contains no source passages, per-record IDs or credentials.
    (ROOT/'reports/possum-dataset-v1.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'rows':len(rows), 'csv_sha256':manifest['csv_sha256'], 'checks':manifest['checks']}, indent=2))


if __name__ == '__main__':
    main()
