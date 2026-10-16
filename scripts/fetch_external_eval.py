"""Prepare source-held-out human IBC and weak-label platform evaluation sets."""
import collections
import hashlib
import io
import json
import pickle
import tarfile
from pathlib import Path
import numpy as np
import requests
from possum.data import normalize

IBC_URL='https://people.cs.umass.edu/~miyyer/data/sample_ibc.tar.gz'

class TreeRecord:
    """Inert storage only: upstream tree code is never imported or executed."""
    def __init__(self,*args):
        pass

class RestrictedTreeReader(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='treeUtil' and name in {'leafObj','nodeObj'}:
            return TreeRecord
        raise pickle.UnpicklingError(f'Forbidden pickle global: {module}.{name}')


def tree_words(node):
    if hasattr(node,'word'):
        return [node.word]
    return tree_words(node.c1)+tree_words(node.c2)


def fetch_ibc(root):
    response=requests.get(IBC_URL,timeout=60);response.raise_for_status()
    archive=response.content
    (root/'sample_ibc.tar.gz').write_bytes(archive)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        data=tar.extractfile('sampleData.pkl').read()
    groups=RestrictedTreeReader(io.BytesIO(data),encoding='latin1').load()
    if len(groups)!=3:
        raise ValueError('IBC must contain liberal, conservative, neutral lists')
    rows=[]
    for label,trees in enumerate(groups[:2]):
        for i,tree in enumerate(trees):
            words=tree_words(tree)
            text=' '.join(words)
            if not 20<=len(text.split())<=150:
                continue
            rows.append(dict(text=text,label=label,source='ibc_official_sample',label_kind='human',group_ids=['ibc:entire_sample'],group_quality='source_holdout_no_document_ids',upstream_id=f'{label}:{i}'))
    return rows,dict(url=IBC_URL,sha256=hashlib.sha256(archive).hexdigest(),raw_label_counts=[len(g) for g in groups],eligible=len(rows),neutral_excluded=len(groups[2]))


def fetch_platform(root):
    repo='mlburnham/dem_rep_party_platform_topics'
    rows=[];raw=[]
    # Deterministic spread through the published training split, not a new row split.
    for offset in np.linspace(0,15338,16,dtype=int):
        r=requests.get('https://datasets-server.huggingface.co/rows',params=dict(dataset=repo,config='default',split='train',offset=int(offset),length=100),timeout=60)
        r.raise_for_status();batch=r.json()['rows'];raw.extend(batch)
        for item in batch:
            x=item['row'];party=x['dataset']
            if party not in {'Dem. Party Platforms','Rep. Party Platforms'}:
                raise ValueError(f'Unknown platform provenance {party}')
            text=x['premise']
            if 20<=len(text.split())<=150:
                rows.append(dict(text=text,label=int(party=='Rep. Party Platforms'),source='party_platform',label_kind='weak',group_ids=['platform:entire_source'],group_quality='source_holdout_no_document_ids',upstream_id=item['row_idx']))
    (root/'platform_rows.json').write_text(json.dumps(raw))
    unique={}
    for row in rows:
        key=hashlib.sha256(normalize(row['text']).encode()).hexdigest()
        if key in unique and unique[key]['label']!=row['label']:
            unique[key]['excluded']=True
        else:
            unique.setdefault(key,row)
    candidates=[r for r in unique.values() if not r.get('excluded')]
    rng=np.random.default_rng(42)
    selected=[]
    for label in (0,1):
        pool=[r for r in candidates if r['label']==label]
        ids=rng.choice(len(pool),min(150,len(pool)),replace=False)
        selected.extend(pool[int(i)] for i in ids)
    return selected,dict(dataset=repo,sampling='16 evenly spaced 100-row windows from upstream train; 150 unique eligible rows per label maximum',raw_rows=len(raw),eligible_unique=len(candidates),selected=len(selected),sha256=hashlib.sha256((root/'platform_rows.json').read_bytes()).hexdigest())


def main():
    root=Path('data/external');root.mkdir(parents=True,exist_ok=True)
    ibc,ibc_meta=fetch_ibc(root)
    platform,platform_meta=fetch_platform(root)
    rows=ibc+platform
    (root/'external.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    meta=dict(ibc=ibc_meta,platform=platform_meta,label_counts=dict(collections.Counter(f"{r['source']}:{r['label']}" for r in rows)),role='external diagnostic evaluation only; never fit or calibrate on these labels')
    (root/'manifest.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
