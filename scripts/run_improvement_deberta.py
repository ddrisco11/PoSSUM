"""Queue DeBERTa comparisons after the human adapter workflow."""
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
from possum.compact_training import deberta_plan
plan = deberta_plan(root)
deadline = min(datetime.datetime.fromisoformat(json.loads(state_path.read_text())['deadline']).timestamp(), plan['compute_until'])
lock = (root / 'pipeline.lock').open('a')
while time.time() < deadline:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        time.sleep(2)
        continue
    ready = root / 'experts/mistral_human_lora_e2.json'
    if ready.exists() and json.loads(ready.read_text()).get('complete'):
        break
    fcntl.flock(lock, fcntl.LOCK_UN)
    time.sleep(2)
else:
    sys.exit(0)


def update(**fields):
    state = json.loads(state_path.read_text())
    if fields.get('status') == 'running':
        state.pop('error', None)
    state.update(fields, updated_at=time.time())
    save_json(state_path, state)


commands = [
    ('deberta_training', [sys.executable, '-u', '-m', 'possum.compact_training', str(root), '--family', 'deberta']),
    ('deberta_mixture_comparison', [sys.executable, '-u', '-m', 'possum.improvement_mixtures', str(root)]),
    ('deberta_export_verification', [sys.executable, '-u', '-m', 'possum.improvement_inference', 'export', str(root)]),
]
for stage, command in commands:
    if time.time() >= deadline:
        update(status='development_budget_reached', active_worker=None)
        sys.exit(0)
    update(status='running', stage=stage, active_worker={'pid': os.getpid(), 'purpose': stage, 'log': 'deberta.log'})
    print('STAGE', stage, flush=True)
    try:
        subprocess.run(command, check=True, timeout=max(1, deadline - time.time() + 60))
    except Exception as error:
        update(status='needs_attention', error=repr(error), active_worker=None)
        raise
update(status='ready_for_next_iteration', stage='deberta_verified_export', active_worker=None)
print('DeBERTa comparison complete. Continue useful development; topic holdout remains sealed.', flush=True)
