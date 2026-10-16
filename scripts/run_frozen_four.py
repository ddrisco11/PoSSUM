"""Sequential frozen-expert pilot with resumable scoring and measured throughput."""
import argparse, json, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np
from possum.checkpoints import save_scores, save_json

MODELS = {
 'mistral': ('mlx-community/Mistral-7B-v0.3-4bit','5e8b40756eaa1cb5131715937c77107b4421c3a5'),
 'qwen': ('mlx-community/Qwen3.5-9B-4bit','8b2b98c00a6b4d291155e4890773ca8f769aee53'),
}
p=argparse.ArgumentParser()
p.add_argument('--input',default='data/frozen-four-pilot.jsonl')
p.add_argument('--run',default='runs/frozen-four-42')
p.add_argument('--worker',choices=list(MODELS))
a=p.parse_args();root=Path(a.run);root.mkdir(parents=True,exist_ok=True)
if a.worker:
 from huggingface_hub import snapshot_download
 from possum.components.causal import MLXCausal
 repo,rev=MODELS[a.worker]
 rows=[json.loads(l) for l in (root/'development.jsonl').read_text().splitlines()]
 target=root/f'{a.worker}_raw.npz';values=[];elapsed=[]
 progress=root/f'{a.worker}_progress.json'
 if target.exists():
  try:
   with np.load(target,allow_pickle=False) as old:
    cached_ids=old['ids'].tolist();values=old['values'].tolist()
    elapsed=old['chunk_seconds'].tolist() if 'chunk_seconds' in old else []
  except (OSError,ValueError,EOFError,zipfile.BadZipFile):
   with np.load(root/f'{a.worker}_recovery.npz',allow_pickle=False) as old:
    cached_ids=old['ids'].tolist();values=old['values'].tolist();elapsed=[]
   print(f'{a.worker}: restored recovery snapshot',flush=True)
  assert cached_ids==[r['id'] for r in rows[:len(cached_ids)]]
  if not elapsed and progress.exists():
   elapsed=json.loads(progress.read_text()).get('chunk_seconds',[])
 if len(values)==len(rows):
  save_json(target.with_suffix('.json'),dict(kind='scores',frozen_zero_shot=True,model_id=repo,revision=rev))
  print(f'{a.worker}: complete cache reused; no model load needed',flush=True)
  sys.exit(0)
 print(f'Downloading/loading {repo} @ {rev}; resume from {len(values)}/{len(rows)}',flush=True)
 path=snapshot_download(repo,revision=rev)
 model=MLXCausal(path)
 for start in range(len(values),len(rows),10):
  t=time.perf_counter();values.extend(model.score([r['text'] for r in rows[start:start+10]]).tolist());elapsed.append(time.perf_counter()-t)
  save_scores(target,values,[r['id'] for r in rows[:len(values)]],elapsed)
  seconds=sum(elapsed)/len(values)
  status=dict(completed=len(values),total=len(rows),chunk_seconds=elapsed,seconds_per_excerpt=seconds,projected_100k_hours=seconds*100000/3600)
  save_json(progress,status)
  print(f'{a.worker}: {len(values)}/{len(rows)}, {seconds:.3f} seconds/excerpt',flush=True)
 save_json(target.with_suffix('.json'),dict(kind='scores',frozen_zero_shot=True,model_id=repo,revision=rev))
else:
 from possum.data import prepare,read_jsonl
 if not Path(a.input).exists():
  candidates=read_jsonl('data/manifesto/candidates.jsonl')
  np.random.default_rng(42).shuffle(candidates)
  Path(a.input).parent.mkdir(parents=True,exist_ok=True)
  Path(a.input).write_text(''.join(json.dumps(r)+'\n' for r in candidates[:1200]))
 def run(*args):subprocess.run([sys.executable,*args],check=True)
 if not (root/'report.json').exists():
  run('-m','possum.cli','pilot',a.input,'--output',a.run,'--components','bge','bert','gliner','--device','mps','--limit','10000')
 rows={r['id']:r for r in prepare(read_jsonl(a.input))}
 split=json.loads((root/'split.json').read_text())['pilot']
 ids=sum(split.values(),[])
 (root/'development.jsonl').write_text(''.join(json.dumps(rows[i])+'\n' for i in ids))
 for expert in MODELS:
  run(__file__,'--input',a.input,'--run',a.run,'--worker',expert)
 run('scripts/replay_mixtures.py',a.run,'--expert',f'mistral={root}/mistral_raw.npz','--expert',f'qwen={root}/qwen_raw.npz','--only','bert','gliner','mistral','qwen','--output',str(root/'four-expert-report.json'))
 print('Frozen four-expert routing evaluation complete.',flush=True)
