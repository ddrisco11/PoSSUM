"""Fetch annotated U.S. manifestos and map human policy codes to RILE labels.
Credentials are supplied via MANIFESTO_API_KEY, never stored in artifacts.
"""
import collections,csv,hashlib,json,os
from pathlib import Path
import requests

LEFT={103,105,106,107,202,403,404,406,412,413,504,506,701}
RIGHT={104,201,203,305,401,402,407,414,505,601,603,605,606}
BASE='https://manifesto-project.wzb.eu/api/v1/'


def label_for(code):
    try:main=int(str(code).split('.')[0])
    except ValueError:return None
    return 0 if main in LEFT else 1 if main in RIGHT else None


def normalize(root):
    docs=json.loads((root/'us-texts.json').read_text())['items']
    rows=[];counts=collections.Counter()
    for doc in docs:
        for i,item in enumerate(doc['items']):
            counts['raw_quasi_sentences']+=1
            label=label_for(item['cmp_code'])
            if label is None:
                counts['unmapped_or_neutral']+=1;continue
            text=' '.join(item['text'].split())
            if not 20<=len(text.split())<=150:
                counts['length_excluded']+=1;continue
            rows.append(dict(text=text,label=label,source='manifesto_us_2026_1',label_kind='weak',annotation_origin='human_policy_code_RILE_mapping',group_ids=['manifesto:document:'+doc['key']],group_quality='document',upstream_id=f"{doc['key']}:{i}",policy_code=item['cmp_code'],corpus_version='2026-1',admission='candidate_policy_mapping'))
    counts['candidates']=len(rows);counts['left']=sum(r['label']==0 for r in rows);counts['right']=sum(r['label']==1 for r in rows)
    (root/'candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=dict(core_version='MPDS2026a',corpus_version='2026-1',country='United States',documents=len(docs),document_ids=[d['key'] for d in docs],counts=dict(counts),left_codes=sorted(LEFT),right_codes=sorted(RIGHT),mapping_source='https://manifesto-project.wzb.eu/down/tutorials/main-dataset.html',annotation='Human-coded policy categories mapped to ideology; not direct human left/right judgments',sha256=hashlib.sha256((root/'us-texts.json').read_bytes()).hexdigest())
    (root/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


def fetch(root):
    session=requests.Session();session.headers['API_KEY']=os.environ['MANIFESTO_API_KEY']
    r=session.get(BASE+'get_core',params={'key':'MPDS2026a','raw':'true'},timeout=90);r.raise_for_status();(root/'core.csv').write_bytes(r.content)
    rows=list(csv.DictReader((root/'core.csv').open()))
    keys=[r['party']+'_'+r['date'] for r in rows if r['countryname']=='United States']
    r=session.get(BASE+'metadata',params=[('version','2026-1')]+[('keys[]',k) for k in keys],timeout=90);r.raise_for_status()
    metadata=r.json();(root/'us-metadata.json').write_text(json.dumps(metadata,indent=2))
    keys=sorted({str(r['manifesto_id']) for r in metadata['items'] if r.get('annotations') and r.get('manifesto_id') and str(r.get('language','')).lower()=='english'})
    r=session.get(BASE+'texts_and_annotations',params=[('version','2026-1')]+[('keys[]',k) for k in keys],timeout=120);r.raise_for_status();(root/'us-texts.json').write_text(json.dumps(r.json()))

if __name__=='__main__':
    root=Path('data/manifesto');root.mkdir(parents=True,exist_ok=True)
    if not (root/'us-texts.json').exists():fetch(root)
    normalize(root)
