"""Evaluate two human-only Mistral LoRA epochs after the frozen prompt comparison."""
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
    ready = root / 'experts/qwen_chat.json'
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


commands = []
for epoch in [1, 2]:
    commands += [
        (f'mistral_human_train_e{epoch}', [sys.executable, '-u', '-m', 'possum.human_adapter', str(root), '--epoch', str(epoch)]),
        (f'mistral_human_score_e{epoch}', [sys.executable, '-u', '-m', 'possum.improvement_experts', str(root), f'mistral_human_lora_e{epoch}']),
    ]
commands += [
    ('human_adapter_comparison', [sys.executable, '-u', '-m', 'possum.improvement_mixtures', str(root)]),
    ('human_adapter_export', [sys.executable, '-u', '-m', 'possum.improvement_inference', 'export', str(root)]),
]
for stage, command in commands:
    if time.time() >= deadline:
        update(status='deadline_reached', active_worker=None)
        sys.exit(0)
    update(status='running', stage=stage, active_worker={'pid': os.getpid(), 'purpose': stage, 'log': 'human-adapter.log'})
    print('STAGE', stage, flush=True)
    try:
        subprocess.run(command, check=True, timeout=max(1, deadline - time.time() + 60))
    except Exception as error:
        update(status='needs_attention', error=repr(error), active_worker=None)
        raise
update(status='ready_for_next_iteration', stage='human_adapter_verified_export', active_worker=None)
print('Human adapter comparison complete. Continue useful development; topic holdout remains sealed.', flush=True)
