"""Refit mixtures on cached pilot outputs; never run a backbone or access test texts."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from possum.learning import fit_gate,predict_gate,metrics,temperature

p=argparse.ArgumentParser()
p.add_argument('run')
p.add_argument('--expert',action='append',default=[],help='name=score.npz')
p.add_argument('--output',required=True)
p.add_argument('--only', nargs='+', help='Restrict the mixture to named experts')
a=p.parse_args()
torch.set_num_threads(4)
root=Path(a.run)
z=np.load(root/'scores.npz',allow_pickle=False)
x,y,probs=z['embeddings'],z['labels'],z['probabilities']
names=z['names'].tolist()
split=json.loads((root/'split.json').read_text())['pilot']
ids=sum(split.values(),[])
positions={};offset=0
for k,v in split.items():
 positions[k]=np.arange(offset,offset+len(v));offset+=len(v)
c,g,v=(positions[k] for k in ('calibration','gate','validation'))
extra={}
for spec in a.expert:
 name,path=spec.split('=',1)
 metadata=json.loads(Path(path).with_suffix('.json').read_text())
 if metadata.get('kind') != 'scores' or not metadata.get('frozen_zero_shot'):
  raise ValueError('Replay requires frozen zero-shot score artifacts')
 data=np.load(path,allow_pickle=False)
 if len(set(data['ids'].tolist())) != len(data['ids']):
  raise ValueError('Duplicate score IDs')
 lookup=dict(zip(data['ids'].tolist(),data['values']))
 raw=np.asarray([lookup[i] for i in ids])
 t=temperature(raw[c],y[c])
 probs=np.column_stack([probs,softmax(raw/t,axis=1)[:,1]])
 names.append(name);extra[name]={'temperature':t,'metrics':metrics(y[v],probs[v,-1])}
if a.only:
 keep=[names.index(n) for n in a.only]
 probs=probs[:,keep];names=list(a.only)
started=time.perf_counter()
report={'single_experts':{n:metrics(y[v],probs[v,j]) for j,n in enumerate(names)},'equal_mixture':metrics(y[v],probs[v].mean(1)),'experts':names,'note':'Same frozen expert predictions and fixed data split; seeds vary only mixture initialization. Development validation diagnostic, not a significance test.','extra_experts':extra,'seeds':{}}
for seed in (42,43,44):
 results={}
 for name,fixed,own in [('fixed',True,False),('expert_gate',False,False),('hybrid_gate',False,True)]:
  model=fit_gate(x[g],probs[g],y[g],fixed=fixed,seed=seed,self_branch=own)
  checkpoint=Path(a.output).parent/f'{Path(a.output).stem}-{name}-{seed}.pt'
  checkpoint.parent.mkdir(parents=True,exist_ok=True)
  torch.save(model.state_dict(),checkpoint)
  pred=predict_gate(model,x[v],probs[v])
  with torch.inference_mode():w=model.weights(torch.tensor(x[v],dtype=torch.float32)).mean(0).tolist()
  results[name]={'metrics':metrics(y[v],pred),'mean_weights':dict(zip(names+(['self'] if own else []),w))}
  if own:
   results[name]['leave_one_out']={}
   for j,expert in enumerate(names):
    keep=[i for i in range(len(names)) if i!=j]
    ablated=fit_gate(x[g],probs[g][:,keep],y[g],seed=seed,self_branch=True)
    results[name]['leave_one_out'][expert]=metrics(y[v],predict_gate(ablated,x[v],probs[v][:,keep]))
 report['seeds'][str(seed)]=results
report['routing_training_seconds']=time.perf_counter()-started
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
Path(a.output).write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps(report,indent=2))
