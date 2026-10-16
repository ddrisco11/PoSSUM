"""Measure BERT fine-tuning throughput; discard benchmark weights."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from possum.data import normalize,read_jsonl
from possum.revisions import resolve
import hashlib

p=argparse.ArgumentParser()
p.add_argument('--pilot',default='runs/full-available-42')
p.add_argument('--output',default='reports/bert-training-throughput.json')
p.add_argument('--steps',type=int,default=12)
a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(42)
if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
ids=json.loads((Path(a.pilot)/'split.json').read_text())['pilot']['fit']
lookup={hashlib.sha256(normalize(r['text']).encode()).hexdigest():r for r in read_jsonl('data/synthetic.jsonl')}
rows=[lookup[i] for i in ids[:8]]
model_id='google-bert/bert-base-uncased';revision=resolve(model_id)
tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision)
model=AutoModelForSequenceClassification.from_pretrained(model_id,revision=revision,num_labels=2,attn_implementation='eager').to('mps').train()
batch=tokenizer([r['text'] for r in rows],padding='max_length',truncation=True,max_length=256,return_tensors='pt').to('mps')
batch['labels']=torch.tensor([r['label'] for r in rows],device='mps')
opt=torch.optim.AdamW(model.parameters(),lr=2e-5)
seconds=[]
for step in range(a.steps+3):
    torch.mps.synchronize();start=time.perf_counter()
    opt.zero_grad(set_to_none=True)
    model(**batch).loss.backward();opt.step()
    torch.mps.synchronize();elapsed=time.perf_counter()-start
    if step>=3:seconds.append(elapsed)
    print(f'BERT fine-tune benchmark step {step+1}/{a.steps+3}: {elapsed:.3f}s',flush=True)
median=float(np.median(seconds))
report=dict(model_id=model_id,revision=revision,device='mps',batch_size=8,sequence_length=256,precision='float32',optimizer='AdamW',trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),warmup_steps=3,measured_steps=a.steps,median_step_seconds=median,min_step_seconds=min(seconds),max_step_seconds=max(seconds),examples_per_second=8/median,current_mps_allocated_bytes=torch.mps.current_allocated_memory(),estimates_hours_three_epochs={str(n):math.ceil(n/8)*3*median/3600 for n in [10000,50000,100000,180000]},limitations=['Timing benchmark only, not a trained model or accuracy result.','Repeated one real training batch at fixed 256-token padding; excludes tokenization, validation, checkpointing, and data transfer overhead.','Extrapolation assumes stable throughput; long runs may throttle or encounter memory pressure.','Estimates cover one BERT fine-tuning run, not all experts, folds, seeds, or parameter searches.'])
Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
