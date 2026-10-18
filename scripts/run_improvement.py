"""Sequential local development stages with exclusive ownership and a fixed deadline."""
import argparse,datetime,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from possum.checkpoints import save_json
from possum.improvement import digest
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run
lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state_path=root/'state.json';state=json.loads(state_path.read_text());end=datetime.datetime.fromisoformat(state['deadline']).timestamp()
def update(**fields):
 state.update(fields,updated_at=time.time());save_json(state_path,state)
dataset=Path('data/improvement-20261018').resolve();assert (dataset/'manifest.json').exists()
signature={'path':str(dataset),'manifest_sha256':digest(dataset/'manifest.json')}
if (root/'dataset.json').exists():assert json.loads((root/'dataset.json').read_text())==signature
else:save_json(root/'dataset.json',signature)
assert json.loads((root/'feature-finished.json').read_text())['complete']
for stage,command,complete in [
 ('baselines',[sys.executable,'-u','-m','possum.improvement','baseline',str(root)],root/'baselines/selection.json'),
 ('compact_training',[sys.executable,'-u','-m','possum.compact_training',str(root)],root/'compact-finished.json')]:
 if complete.exists() and (stage!='compact_training' or json.loads(complete.read_text()).get('complete')):continue
 if time.time()>=end:update(status='deadline_reached',active_worker=None);sys.exit(0)
 update(status='running',stage=stage,active_worker={'pid':os.getpid(),'purpose':stage,'log':'pipeline.log'})
 print('STAGE',stage,flush=True)
 try:subprocess.run(command,check=True,timeout=max(1,end-time.time()+60))
 except Exception as e:update(status='needs_attention',error=repr(e),active_worker=None);raise
update(status='ready_for_next_iteration',stage='baseline_and_compact_comparison',active_worker=None)
print('Initial comparative stage finished; topic holdout remains sealed.',flush=True)
