"""Queue fit-only teacher scoring and compact distillation after seed replication."""
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
while time.time() < deadline:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        time.sleep(2)
        continue
    ready = root / 'replication-finished.json'
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
    ('teacher_fit_scoring', [sys.executable, '-u', '-m', 'possum.distillation', str(root)]),
    ('compact_distillation', [sys.executable, '-u', '-m', 'possum.compact_training', str(root), '--family', 'distillation']),
    ('distilled_mixture_comparison', [sys.executable, '-u', '-m', 'possum.improvement_mixtures', str(root)]),
    ('distilled_export_verification', [sys.executable, '-u', '-m', 'possum.improvement_inference', 'export', str(root)]),
]
for stage, command in commands:
    if time.time() >= deadline:
        update(status='deadline_reached', active_worker=None)
        sys.exit(0)
    update(status='running', stage=stage, active_worker={'pid': os.getpid(), 'purpose': stage, 'log': 'distillation.log'})
    print('STAGE', stage, flush=True)
    try:
        subprocess.run(command, check=True, timeout=max(1, deadline - time.time() + 60))
    except Exception as error:
        update(status='needs_attention', error=repr(error), active_worker=None)
        raise
update(status='ready_for_next_iteration', stage='distilled_verified_export', active_worker=None)
print('Distillation comparison complete. Continue useful development; topic holdout remains sealed.', flush=True)
