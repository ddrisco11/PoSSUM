"""Prepare deterministic, local-only review samples without exposing held-out text."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from possum.data import normalize, read_jsonl
from fetch_manifesto import label_for


URLS = {
    'IBC public sample': 'https://people.cs.umass.edu/~miyyer/ibc/',
    'Manifesto U.S.': 'https://manifesto-project.wzb.eu/',
    'Party Platform Topics': 'https://huggingface.co/datasets/mlburnham/dem_rep_party_platform_topics',
    'GPT-4 Political Ideologies': 'https://huggingface.co/datasets/JyotiNayak/political_ideologies',
    'Political Tweets': 'https://huggingface.co/datasets/Jacobvs/PoliticalTweets',
    'GPT-4 Political Bias': 'https://huggingface.co/datasets/cajcodes/political-bias',
    'Qbias': 'https://github.com/matous-volf/political-leaning-prediction/tree/main/datasets/political_leaning/notebooks',
    'SCCC': 'https://congressionalcomms.com/',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('data/review-20261017'))
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Use a new review output to preserve any human annotations')
    root = Path('data/assembled-v1')
    protected = json.loads((root/'protected.json').read_text())
    reserved, groups = set(protected['ids']), set(protected['group_ids'])
    # Read provenance to avoid exposing any reserved content, even through another source.
    index = {r['id']: r for r in read_jsonl(root/'candidates.jsonl')}
    def cid(r):
        return hashlib.sha256(normalize(r['text']).encode()).hexdigest()
    def safe(r):
        key = cid(r)
        provenance_groups = (set(r.get('group_ids', [])) | set(index.get(key, {}).get('group_ids', []))) - {'platform:entire_source'}
        return key not in reserved and not provenance_groups & groups
    def sample(name, rows):
        unique = {cid(r): r for r in rows if safe(r)}
        selected = sorted(unique, key=lambda k: hashlib.sha256(('review-20261017:'+name+':'+k).encode()).hexdigest())[:50]
        assert len(selected) == 50, (name, len(selected))
        return [unique[k] for k in selected], len(unique)
    sources = [
        ('IBC public sample', read_jsonl(root/'external_diagnostic.jsonl'), 'Human passage ideology; previously inspected diagnostic'),
        ('Manifesto U.S.', [r for r in read_jsonl(root/'development.jsonl') if 'manifesto' in r['source']], 'Human policy category mapped to weak ideology'),
        ('Party Platform Topics', read_jsonl('data/available/mlburnham_dem_rep_party_platform_topics/candidates.jsonl'), 'Party provenance; not passage ideology'),
        ('GPT-4 Political Ideologies', read_jsonl('data/synthetic.jsonl'), 'Synthetic ideology labels'),
        ('Political Tweets', read_jsonl('data/available/Jacobvs_PoliticalTweets/candidates.jsonl'), 'Author party; not passage ideology'),
    ]
    bias = []
    for i, r in enumerate(pq.read_table('data/available/cajcodes_political-bias/train-0.parquet').to_pylist()):
        original = r['label']
        bias.append(dict(r, label=None if original == 2 else int(original < 2), original_label=original,
                         label_kind='synthetic', group_ids=['cajcodes:entire_generation_source'], upstream_id=i))
    sources.append(('GPT-4 Political Bias', bias, 'Synthetic five-way labels; 0/1 right, 2 neutral, 3/4 left'))
    headers = ['dataset', 'text', 'label', 'label_name', 'word_count', 'overnight_role', 'next_run_status',
               'label_basis', 'exclusion_reason', 'review_label', 'self_contained_stance', 'review_notes',
               'document_or_author', 'policy_code', 'original_five_way_label', 'record_id', 'upstream_id', 'source_url']
    def output_row(name, r, basis, raw=False):
        old = index.get(cid(r), {})
        role = old.get('split', 'excluded_before_assembly')
        reasons = list(old.get('quarantine_reasons', []))
        next_status = role
        if name == 'Manifesto U.S.':
            next_status = 'excluded_policy_mapping'
            reasons.append('Policy category may require context absent from excerpt')
        if name == 'GPT-4 Political Bias':
            role = next_status = 'excluded_before_assembly'
            reasons = ['Neutral original label'] if r['label'] is None else ['Below 20-word minimum']
        if raw:
            next_status = 'excluded_policy_mapping'
            if not 20 <= len(r['text'].split()) <= 150:
                role = 'excluded_before_assembly'
                reasons.append('Outside 20–150-word contract')
            if r['label'] is None:
                role = 'excluded_before_assembly'
                reasons.append('Unmapped policy code; no binary label')
        label = r.get('label')
        return [name, r['text'], label, {0:'Left',1:'Right'}.get(label,'Neutral / unmapped'), len(r['text'].split()),
                role, next_status, basis, '; '.join(reasons), '', '', '', '; '.join(r.get('group_ids', [])),
                r.get('policy_code',''), r.get('original_label',''), cid(r), str(r.get('upstream_id','')), URLS[name]]
    main_rows, pools, sampled_ids = [], {}, {}
    for name, rows, basis in sources:
        selected, size = sample(name, rows)
        pools[name] = size
        sampled_ids[name] = [cid(r) for r in selected]
        main_rows.extend(output_row(name, r, basis) for r in selected)
    fitted = [r for r in read_jsonl(root/'fit.jsonl') if r['source']=='Jacobvs/PoliticalTweets']
    selected, size = sample('Tweets used for adapter fitting', fitted)
    extra_rows = [['Admitted tweet fit', *output_row('Political Tweets', r, sources[4][2])] for r in selected]
    raw_manifesto = []
    for doc in json.loads(Path('data/manifesto/us-texts.json').read_text())['items']:
        if 'manifesto:document:'+doc['key'] in groups:
            continue
        for i, r in enumerate(doc['items']):
            raw_manifesto.append(dict(text=r['text'], label=label_for(r['cmp_code']), policy_code=r['cmp_code'],
                                     group_ids=['manifesto:document:'+doc['key']], upstream_id=f"{doc['key']}:{i}"))
    selected, raw_size = sample('Raw Manifesto fragments', raw_manifesto)
    extra_rows.extend(['Raw Manifesto before filters', *output_row('Manifesto U.S.', r, sources[1][2],raw=True)] for r in selected)
    inventory_headers = ['dataset', 'local_inventory', 'overnight_fit', 'overnight_calibration', 'overnight_gate', 'overnight_validation', 'other_usage', 'unused_or_excluded_reason', 'next_run_status', 'source_url']
    inventory_rows = [
        ['IBC public sample','97 eligible passages',0,0,0,0,'97 previously inspected external diagnostic examples','Full corpus unavailable; public sample reserved for diagnostics','Diagnostic only',URLS['IBC public sample']],
        ['Manifesto U.S.','4289 candidate occurrences; 3827 normalized unique',54,0,0,243,'Earlier pilot also used Manifesto','Context-dependent policy categories do not establish standalone viewpoint','Excluded from new development',URLS['Manifesto U.S.']],
        ['Party Platform Topics','8535 candidate passages',0,0,0,0,'300 used in earlier diagnostics','Missing document/year IDs; cross-source duplicate label conflicts','Quarantined',URLS['Party Platform Topics']],
        ['GPT-4 Political Ideologies','3200 synthetic statements',1200,400,400,400,'800 protected holdout statements','Synthetic source; generation-batch provenance unavailable','Unchanged',URLS['GPT-4 Political Ideologies']],
        ['Political Tweets','114183 candidates from 190491 raw rows',6398,1567,1795,1256,'11850 passed initial screen; 832 reserved holdout','102333 candidates failed initial screen; further assembly exclusions apply','Unchanged weak labels; human review needed',URLS['Political Tweets']],
        ['GPT-4 Political Bias','657 raw statements; 456 nonneutral and 201 neutral',0,0,0,0,'No model fitting or evaluation','All 456 nonneutral statements below 20 words; neutral excluded','Excluded pending short-text protocol',URLS['GPT-4 Political Bias']],
        ['Qbias','No raw text collection acquired',0,0,0,0,'Source notebook inspected only','Referenced raw CSV not available locally','Unavailable',URLS['Qbias']],
        ['SCCC','No message text collection acquired',0,0,0,0,'No model fitting or evaluation','Full message text must be acquired separately','Unavailable',URLS['SCCC']],
    ]
    # Reconcile the displayed usage counts with actual immutable development inputs.
    names = {'ibc_official_sample':'IBC public sample','manifesto_us_2026_1':'Manifesto U.S.',
             'mlburnham/dem_rep_party_platform_topics':'Party Platform Topics',
             'JyotiNayak/political_ideologies':'GPT-4 Political Ideologies','Jacobvs/PoliticalTweets':'Political Tweets'}
    counts = {part: collections.Counter(names.get(r['source'],r['source']) for r in read_jsonl(root/f'{part}.jsonl'))
              for part in ('fit','calibration','gate','validation')}
    for row in inventory_rows:
        assert row[2:6] == [counts[part][row[0]] for part in counts], row[0]
    a.output.mkdir(parents=True)
    packet = dict(headers=headers, samples=main_rows, supplementary_headers=['sample_pool',*headers],
                  supplementary=extra_rows, inventory_headers=inventory_headers, inventory=inventory_rows)
    (a.output/'review-input.json').write_text(json.dumps(packet,ensure_ascii=False))
    audit = dict(seed='review-20261017', method='Smallest SHA-256 ranks of seed, pool name and normalized content ID; unique texts, no label balancing or hand selection',
                 main_sample_count=len(main_rows), main_sample_counts={name:50 for name,_,_ in sources},
                 safe_pool_counts=pools, supplementary_counts={'admitted_fit_tweets':50,'raw_manifesto':50},
                 protected_ids_and_known_document_author_groups_excluded=True, sampled_ids=sampled_ids,
                 limitations='Sample pools exclude protected IDs and known document/author groups. The platform:entire_source placeholder is not a document identity: platform audit samples exclude known protected content and linked real documents, but unknown document provenance remains a limitation. Main Manifesto pool is historical admitted development; raw fragments are separate. Bias samples include neutral and short records. Samples are for label review, not new test evidence. Cross-source near-duplicate recall is limited to the prior assembly audit.')
    (a.output/'sampling.json').write_text(json.dumps(audit,indent=2)+'\n')
    public = {k:v for k,v in audit.items() if k!='sampled_ids'}
    Path('reports/dataset-review-sampling.json').write_text(json.dumps(public,indent=2)+'\n')
    print(json.dumps(public,indent=2))


if __name__ == '__main__':
    main()
