"""Resume frozen feature extraction with pinned inputs and an absolute deadline."""
import argparse, datetime, fcntl, hashlib, json, os, time
from pathlib import Path
import numpy as np
from possum.checkpoints import save_json,save_scores
from possum.data import read_jsonl

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--deadline',required=True);a=p.parse_args()
lock=(a.run/'feature-process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
source=a.run/'feature-input.jsonl';signature=hashlib.sha256(source.read_bytes()).hexdigest();rows=read_jsonl(source)
end=datetime.datetime.fromisoformat(a.deadline).timestamp()
meta=a.run/'feature-run.json'
if meta.exists():
 old=json.loads(meta.read_text());assert old['input_sha256']==signature and old['deadline']==end
else:save_json(meta,dict(input_sha256=signature,deadline=end))
import torch
torch.set_num_threads(4)
if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; do not silently fall back for this run')
for name in ['bge','bert']:
 target=a.run/f'{name}_features.npz';values=[];seconds=[]
 ids=[r['id'] for r in rows]
 if target.exists():
  with np.load(target,allow_pickle=False) as z:
   assert z['ids'].tolist()==ids[:len(z['ids'])];values=z['values'].tolist();seconds=z['chunk_seconds'].tolist()
 if len(values)==len(rows):continue
 if time.time()>=end:break
 if name=='bge':
  from possum.components.bge import BGE
  model=BGE(device='mps')
 else:
  from possum.components.bert import BERT
  model=BERT(device='mps')
 truncated=sum(len(model.tokenizer.encode(r['text']))>256 for r in rows)
 for start in range(len(values),len(rows),128):
  if time.time()>=end:break
  t=time.perf_counter();v=model.encode([r['text'] for r in rows[start:start+128]])
  assert np.isfinite(v).all();values.extend(v.tolist());seconds.append(time.perf_counter()-t)
  save_scores(target,values,ids[:len(values)],seconds)
  progress=dict(pid=os.getpid(),stage=name,completed=len(values),total=len(rows),seconds=sum(seconds),updated_at=time.time(),truncated_at_256=truncated,status='complete' if len(values)==len(rows) else 'running')
  save_json(a.run/'feature-progress.json',progress);print(json.dumps(progress),flush=True)
 del model
 import gc;gc.collect();torch.mps.empty_cache()
save_json(a.run/'feature-finished.json',dict(finished_at=time.time(),deadline=end,complete=all((a.run/f'{n}_features.npz').exists() and len(np.load(a.run/f'{n}_features.npz')['ids'])==len(rows) for n in ['bge','bert'])))
