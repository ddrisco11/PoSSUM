"""Version a broader reviewed corpus with immutable development and topic holdouts."""
import collections,csv,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from possum.data import normalize,read_jsonl
from possum.checkpoints import save_json
from prepare_possum_dataset_v1 import lexical_edges,clean

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/improvement-20261018'
REVIEW=ROOT/'data/possum-v1.1-build'
FACETS=(1,3,4,5,6)
STRESS={'Political Parties','Energy Crisis','Mass Shootings'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cid(text):return hashlib.sha256(normalize(text).encode()).hexdigest()
def key(r):return (r['id'],r['upstream_row'])
def decisions(paths,inputs):
 d={}
 for p in paths:
  for r in read_jsonl(p):
   k=key(r)
   if k in d:raise ValueError('Duplicate occurrence review')
   assert r['decision'] in {'keep','exclude','uncertain'} and r['reason']
   d[k]=r
 assert set(d)=={key(r) for r in inputs},'Incomplete review'
 return d

class Union:
 def __init__(self,n):self.parent=list(range(n))
 def root(self,i):
  while self.parent[i]!=i:self.parent[i]=self.parent[self.parent[i]];i=self.parent[i]
  return i
 def join(self,i,j):self.parent[self.root(j)]=self.root(i)

def main():
 source=ROOT/'data/available/MITweet/MITweet.csv'
 files=[source,ROOT/'data/PoSSUMDatasetv1.0/PoSSUMDatasetv1.0.csv',ROOT/'data/assembled-v1/protected.json']
 files+=sorted((ROOT/'data/possum-v1-build').glob('mitweet-decisions-*.jsonl'))
 files += [REVIEW/'expansion-decisions-0.jsonl',REVIEW/'expansion-decisions-1.jsonl',REVIEW/'reconsider-decisions.jsonl']
 files += [ROOT/'data/assembled-v1/candidates.jsonl',ROOT/'data/assembled-v1/development.jsonl',ROOT/'data/external/external.jsonl',ROOT/'data/possum-v1-build/mitweet-candidates.jsonl',REVIEW/'expansion-review-0.jsonl',REVIEW/'expansion-review-1.jsonl',REVIEW/'reconsider-review.jsonl',ROOT/'scripts/prepare_possum_dataset_v1.py',Path(__file__)]
 hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
 if (DEST/'manifest.json').exists():
  old=json.loads((DEST/'manifest.json').read_text());assert old['inputs']==hashes,'Immutable dataset cannot be replaced'
  for part,expected in old['split_hashes'].items():assert sha(DEST/f'{part}.jsonl')==expected,'Frozen split changed'
  print('Dataset already frozen and hashes verified');return
 old_input=read_jsonl(ROOT/'data/possum-v1-build/mitweet-candidates.jsonl')
 d=decisions(files[3:6],old_input)
 expansions=sum([read_jsonl(REVIEW/f'expansion-review-{i}.jsonl') for i in range(2)],[])
 d.update(decisions([REVIEW/f'expansion-decisions-{i}.jsonl' for i in range(2)],expansions))
 d.update(decisions([REVIEW/'reconsider-decisions.jsonl'],read_jsonl(REVIEW/'reconsider-review.jsonl')))
 raw=list(csv.DictReader(source.open(encoding='utf-8-sig',newline='')))
 mapped=collections.defaultdict(set)
 for r in raw:mapped[cid(r['tweet'])].update(int(r[f'I{i}']) for i in FACETS if int(r[f'I{i}'])!=-1)
 conflicts={k for k,v in mapped.items() if len(v)>1}
 rows=[];reject=[]
 for number,r in enumerate(raw,2):
  rid=cid(r['tweet']);decision=d.get((rid,number))
  if not decision:continue
  reason='raw_annotation_conflict' if rid in conflicts else decision['decision'] if decision['decision']!='keep' else None
  vals={int(r[f'I{i}']) for i in FACETS if int(r[f'I{i}'])!=-1}
  if vals not in ({0},{2}):reason='no_concordant_binary_mapping'
  if reason:reject.append(dict(id=rid,upstream_row=number,reason=reason,review=decision));continue
  text=clean(r['tweet']);assert 12<=len(text.split())<=150
  rows.append(dict(id=rid,text=text,label=0 if vals=={0} else 1,source='MITweet',label_kind='human_facet_derived',
   topic=r['topic'],original_labels={f'I{i}':int(r[f'I{i}']) for i in range(1,13)},upstream_row=number,
   group_ids=[],group_quality='lexical_duplicate_component_author_unavailable',curation=decision,
   source_revision='ef6d2175ca562b2a8ec042a19d5a263374f2407b',previously_used=False))
 for r in csv.DictReader(files[1].open(encoding='utf-8-sig',newline='')):
  if r['sub_dataset']!='GPT-4 Political Ideologies':continue
  rows.append(dict(id=r['record_id'],text=r['text'],label=int(r['label']),source=r['sub_dataset'],label_kind='synthetic',
   topic='synthetic_issue_'+r['topic'],original_labels=json.loads(r['original_labels_json']),group_ids=json.loads(r['group_ids_json']),
   group_quality=r['group_quality'],split=r['partition'],previously_used=True,provenance=json.loads(r['provenance_json'])))
 protected=json.loads(files[2].read_text());pids=set(protected['ids']);pgroups=set(protected['group_ids'])
 references={r['id']:dict(text=r['text'],id=r['id'],kind='protected') for r in read_jsonl(ROOT/'data/assembled-v1/candidates.jsonl') if r['id'] in pids}
 assert set(references)==pids
 for r in read_jsonl(ROOT/'data/assembled-v1/development.jsonl'):
  references.setdefault(r['id'],dict(text=r['text'],id=r['id'],kind='development',split=r['split']))
 for r in read_jsonl(ROOT/'data/external/external.jsonl'):
  k=cid(r['text'])
  if references.get(k,{}).get('kind')!='protected':references[k]=dict(text=r['text'],id=k,kind='diagnostic')
 for r in raw:
  rid=cid(r['tweet'])
  if rid in conflicts:
   references.setdefault(rid,dict(id=rid,text=r['tweet'],kind='annotation_conflict'))
   references[rid]['conflict']=True
 refs=list(references.values());print('Reference scan',len(rows),len(refs),flush=True)
 blocked=set();conflicted=set()
 for i,j in lexical_edges(rows,refs):
  row,ref=rows[i],refs[j]
  if ref.get('conflict'):conflicted.add(i)
  if ref['kind']=='annotation_conflict':continue
  if row['source']!='MITweet' and ref['kind']=='development' and row['split']==ref['split']:continue
  blocked.add(i)
 duplicates=lexical_edges(rows);u=Union(len(rows));leak=Union(len(rows))
 for i,j in duplicates:u.join(i,j);leak.join(i,j)
 group_owner={}
 for i,r in enumerate(rows):
  if r['id'] in pids or pgroups.intersection(r['group_ids']):blocked.add(i)
  for group in r['group_ids']:
   if group in group_owner:leak.join(i,group_owner[group])
   else:group_owner[group]=i
 blocked_roots={leak.root(i) for i in blocked}
 clusters=collections.defaultdict(list)
 for i in range(len(rows)):clusters[u.root(i)].append(i)
 accepted=[]
 for indices in clusters.values():
  members=[rows[i] for i in indices]
  if any(leak.root(i) in blocked_roots for i in indices) or any(i in conflicted for i in indices) or len({r['label'] for r in members})>1:
   reject.extend(dict(id=r['id'],reason='protected_history_or_annotation_conflict_component') for r in members);continue
  prior={r['split'] for r in members if 'split' in r}
  if len(prior)>1:reject.extend(dict(id=r['id'],reason='historical_partition_collision') for r in members);continue
  chosen=min(members,key=lambda r:(r['source']!='MITweet',r['id']))
  chosen['group']=min(r['id'] for r in members)
  chosen['provenance_occurrences']=[{k:v for k,v in r.items() if k not in {'text','curation'}} for r in members]
  chosen['topics']=sorted({r['topic'] for r in members})
  if chosen['source']=='MITweet':chosen['group_ids']=['mitweet:lexical_component:'+chosen['group']]
  accepted.append(chosen)
 human=[r for r in accepted if r['source']=='MITweet']
 stress=[r for r in human if STRESS.intersection(r['topics'])]
 dev=[r for r in human if not STRESS.intersection(r['topics'])]
 def cut(data,test_size,seed):
  a,b=train_test_split(np.arange(len(data)),test_size=test_size,random_state=seed,stratify=[r['label'] for r in data]);return [data[i] for i in a],[data[i] for i in b]
 remaining,val=cut(dev,.2,20261018)
 fit,cg=cut(remaining,.25,20261019)
 cal,gate=cut(cg,.5,20261020)
 for part,rr in [('fit',fit),('calibration',cal),('gate',gate),('validation',val),('topic_holdout',stress)]:
  assert {r['label'] for r in rr}=={0,1},f'{part} lacks bothclasses'
  for r in rr:r['split']=part
 groups={}
 for r in accepted:
  assert r['id'] not in pids
  for g in r['group_ids']:
   assert g not in pgroups
   previous=groups.setdefault(g,r['split']);assert previous==r['split']
 DEST.mkdir(exist_ok=True)
 for part in ['fit','calibration','gate','validation','topic_holdout']:
  rr=sorted([r for r in accepted if r['split']==part],key=lambda r:r['id'])
  (DEST/f'{part}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr))
 (DEST/'review-quarantine.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reject))
 counts=dict(collections.Counter(f'{r["source"]}:{r["split"]}:{r["label"]}' for r in accepted))
 manifest=dict(version='improvement-20261018',inputs=hashes,rows=len(accepted),human_rows=len(human),counts=counts,
  split_hashes={p:sha(DEST/f'{p}.jsonl') for p in ['fit','calibration','gate','validation','topic_holdout']},
  stress_topics=sorted(STRESS),selection='Choose development candidates by MITweet validation log loss, with macro-F1, balanced accuracy and calibration reported. Topic holdout is opened only after candidate selection is frozen.',
  grouping='New MITweet partitions use transitive lexical duplicate components, not topic-as-author proxies. Author/thread identities are unavailable; no author-disjoint claim. Topic stress split contains entire preselected topics and linked duplicates.',
  limitations=['Automated screening of human facet annotations, not direct overall-US gold labels.','12–150 words admitted to retain meaningful short tweets.','Historical synthetic roles preserved; no protected historical holdout or diagnostic enters fitting.','All original datasets and earlier metrics remain separate and unchanged.'])
 save_json(DEST/'manifest.json',manifest);print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
