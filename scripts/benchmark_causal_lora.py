"""Bounded M2 causal-model QLoRA throughput probe; updates are discarded, never evaluated as a trained model."""
import argparse,json,time
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=12);p.add_argument('--stress-tokens',type=int,default=0);p.add_argument('--output');p.add_argument('--model',choices=['mistral','qwen'],default='mistral');a=p.parse_args()
a.output=a.output or f'reports/{a.model}-lora-throughput.json'
if a.stress_tokens not in (0,256):raise ValueError('Stress probe supports 256 tokens only')
if not 3<=a.steps<=30:raise ValueError('Profiling is limited to 3–30 steps')
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.trainer import grad_checkpoint
from huggingface_hub import snapshot_download
from possum.components.causal import prompt,candidate_ids
mx.set_memory_limit(8*1024**3);mx.set_cache_limit(256*1024**2);mx.random.seed(42)
repo,revision={'mistral':('mlx-community/Mistral-7B-v0.3-4bit','5e8b40756eaa1cb5131715937c77107b4421c3a5'),'qwen':('mlx-community/Qwen3.5-9B-4bit','8b2b98c00a6b4d291155e4890773ca8f769aee53')}[a.model]
path=snapshot_download(repo,revision=revision,local_files_only=True)
model,tokenizer=load(path);model.freeze()
config={'rank':8,'scale':16.,'dropout':0.,'keys':['self_attn.q_proj','self_attn.v_proj']}
if a.model=='qwen':config['keys']+=['linear_attn.in_proj_qkv','linear_attn.in_proj_z']
linear_to_lora_layers(model,4,config);grad_checkpoint(model.layers[-1]);model.train()
trainable=sum(v.size for _,v in tree_flatten(model.trainable_parameters()))
assert 0<trainable<5000000
rows=[json.loads(l) for l in Path('runs/frozen-four-42/development.jsonl').read_text().splitlines()]
fit=set(json.loads(Path('runs/frozen-four-42/split.json').read_text())['pilot']['fit'])
samples=[]
for r in rows:
 if r['id'] not in fit:continue
 prefix,full=candidate_ids(tokenizer,prompt(r['text']),' left' if r['label']==0 else ' right')
 if len(prefix)<=256 and len(full)==len(prefix)+1:samples.append((prefix,full[-1]))
samples.sort(key=lambda x:len(x[0]))
selected=[samples[int(i)] for i in np.linspace(0,len(samples)-1,a.steps)]
if a.stress_tokens:
 selected=[((ids*((a.stress_tokens+len(ids)-1)//len(ids)))[:a.stress_tokens],target) for ids,target in selected]
opt=optim.Adam(learning_rate=1e-4)
def loss(m,ids,target):return nn.losses.cross_entropy(m(ids)[:,-1,:].astype(mx.float32),target).mean()
value_grad=nn.value_and_grad(model,loss);timings=[];tokens=[]
print(f'QLoRA profile: {trainable} trainable parameters; batch 1; last 4 layers; rank 8',flush=True)
for i,(ids,target) in enumerate(selected):
 start=time.perf_counter();value,grad=value_grad(model,mx.array([ids]),mx.array([target]));opt.update(model,grad);mx.eval(model.parameters(),opt.state,value)
 elapsed=time.perf_counter()-start;timings.append(elapsed);tokens.append(len(ids))
 print(f'{i+1}/{a.steps}: {elapsed:.3f}s, {len(ids)} tokens, peak={mx.get_peak_memory()/1024**3:.2f} GiB',flush=True);mx.clear_cache()
measured=timings[2:];report=dict(model=repo,revision=revision,method='4-bit QLoRA, rank8, selected attention projections within last4layers',lora_config=config,device='Apple M2 16GB',synthetic_length_stress=bool(a.stress_tokens),trainable_parameters=trainable,batch_size=1,gradient_checkpointing=True,max_input_tokens=256,token_lengths=tokens,step_seconds=timings,warmup_steps=2,mean_seconds_per_example=float(np.mean(measured)),median_seconds_per_example=float(np.median(measured)),peak_mlx_gib=mx.get_peak_memory()/1024**3,limitations=['Bounded timing probe; adapter updates discarded. No accuracy claim.','No optimizer compilation; synthetic repeated-token length stress, not quality evaluation.' if a.stress_tokens else 'No optimizer compilation; length-varied passages from existing fit partition only.','Excludes loading, data cleaning, evaluation, checkpoint I/O, long-run thermal throttling and competing workloads.','Training selected attention projections within the last four layers; not full-model fine-tuning or all-layer LoRA.'])
Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
