"""Run sequential adapter training, caching and evaluation under a wall-clock budget."""
import datetime
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
from possum.checkpoints import save_json
from possum.data import read_jsonl

p=argparse.ArgumentParser();p.add_argument('--run',default='runs/option-a-no-manifesto');p.add_argument('--dataset',type=Path,default=Path('data/assembled-v2-no-manifesto'));p.add_argument('--hours',type=float,default=10);p.add_argument('--smoke',action='store_true');p.add_argument('--deadline',help='Absolute ISO-8601 deadline with time zone');p.add_argument('--allow-battery',action='store_true');a=p.parse_args()
root=Path(a.run).resolve();root.mkdir(parents=True,exist_ok=True)
lock=(root/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
parts=('fit','calibration','gate','validation')
if not (root/'run.json').exists():
 subprocess.run([sys.executable,'scripts/check_assembled_corpus.py',str(a.dataset)],check=True)
 for part in parts:
  if any('manifesto' in s.lower() for row in read_jsonl(a.dataset/f'{part}.jsonl') for s in row.get('sources',[row['source']])):
   raise ValueError('New runs exclude Manifesto policy-mapped supervision; derive the no-Manifesto dataset first')
 (root/'input').mkdir(exist_ok=True)
 for part in parts:
  source=a.dataset/f'{part}.jsonl'
  if a.smoke:
   rows=read_jsonl(source);n=4 if part=='fit' else 2
   chosen=sum([[r for r in rows if r['label']==y][:n] for y in (0,1)],[])
   (root/'input'/source.name).write_text(''.join(json.dumps(r)+'\n' for r in chosen))
  else:shutil.copyfile(source,root/'input'/source.name)
 hashes={p:hashlib.sha256((root/'input'/f'{p}.jsonl').read_bytes()).hexdigest() for p in parts}
 save_json(root/'run.json',dict(created_at=time.time(),smoke=a.smoke,hours=a.hours,dataset=str(a.dataset.resolve()),input_sha256=hashes,status='preparing'))
meta=json.loads((root/'run.json').read_text())
if meta['smoke']!=a.smoke:raise ValueError('Cannot change run mode')
for part,expected in meta['input_sha256'].items():
 if hashlib.sha256((root/'input'/f'{part}.jsonl').read_bytes()).hexdigest()!=expected:raise ValueError('Run input changed')
if not a.smoke and not a.allow_battery:
 while "'AC Power'" not in subprocess.check_output(['pmset','-g','batt'],text=True):
  save_json(root/'power.json',dict(status='waiting_for_ac',updated_at=time.time()))
  print('Waiting for AC power before starting training.',flush=True)
  if a.deadline and time.time() >= datetime.datetime.fromisoformat(a.deadline).timestamp():
   meta.update(status='budget_exhausted');save_json(root/'run.json',meta);sys.exit(0)
  time.sleep(30)
if 'deadline' not in meta:
 meta.update(started_at=time.time(),deadline=datetime.datetime.fromisoformat(a.deadline).timestamp() if a.deadline else time.time()+a.hours*3600,status='running');save_json(root/'run.json',meta)
# Resume preserves the original deadline; it never silently starts another10hours.
stages=['mistral_train','bert_train','bge_score','bert_score','gliner_score','mistral_score','qwen_score','evaluate']
for stage in stages:
 progress=root/f'{stage}_progress.json'
 if progress.exists() and json.loads(progress.read_text()).get('status') in ('complete','partial'):continue
 if time.time()>=meta['deadline']:
  meta.update(status='budget_exhausted');break
 limit=3*3600 if stage=='mistral_train' else 3600 if stage=='bert_train' else meta['deadline']-time.time()
 control_path=root/'control.json'
 previous=json.loads(control_path.read_text()) if control_path.exists() else {}
 stage_deadline=previous['stage_deadline'] if previous.get('stage')==stage else min(meta['deadline'],time.time()+limit)
 save_json(control_path,dict(stage=stage,stage_deadline=stage_deadline,require_ac=not (a.smoke or a.allow_battery)))
 meta.pop('error',None);meta.update(status='running',stage=stage);save_json(root/'run.json',meta)
 print(f'STAGE {stage}',flush=True)
 try:
  subprocess.run([sys.executable,'-u','-m','possum.option_a',stage,str(root)],check=True,timeout=max(1,meta['deadline']-time.time()+60))
 except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
  meta.update(status='needs_attention',error=str(exc));save_json(root/'run.json',meta);raise
else:
 meta.update(status='complete',completed_at=time.time())
save_json(root/'run.json',meta)
print(json.dumps(meta,indent=2),flush=True)
