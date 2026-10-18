"""Wait for compact training, then compare frozen experts without GPU contention."""
import argparse
import datetime
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from possum.checkpoints import save_json

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
args = parser.parse_args()
root = args.run
state_path = root / 'state.json'
deadline = datetime.datetime.fromisoformat(json.loads(state_path.read_text())['deadline']).timestamp()
lock = (root / 'pipeline.lock').open('a')
while True:
    if time.time() >= deadline:
        sys.exit(0)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        break
    except BlockingIOError:
        time.sleep(2)
assert json.loads((root / 'compact-finished.json').read_text())['complete'], 'Compact stage incomplete'


def update(**fields):
    state = json.loads(state_path.read_text())
    if fields.get('status') == 'running':
        state.pop('error', None)
    state.update(fields, updated_at=time.time())
    save_json(state_path, state)


for expert in ['gliner', 'mistral', 'qwen']:
    metadata = root / 'experts' / f'{expert}.json'
    if metadata.exists() and json.loads(metadata.read_text()).get('complete'):
        continue
    if time.time() >= deadline:
        update(status='deadline_reached', active_worker=None)
        sys.exit(0)
    update(status='running', stage=f'frozen_{expert}', active_worker={'pid': os.getpid(), 'purpose': f'frozen_{expert}', 'log': 'experts.log'})
    print('STAGE', expert, flush=True)
    try:
        subprocess.run([sys.executable, '-u', '-m', 'possum.improvement_experts', str(root), expert],
                       check=True, timeout=max(1, deadline - time.time() + 60))
    except Exception as error:
        update(status='needs_attention', error=repr(error), active_worker=None)
        raise
update(status='ready_for_next_iteration', stage='frozen_expert_comparison', active_worker=None)
print('Frozen expert comparison complete; topic holdout remains sealed.', flush=True)
