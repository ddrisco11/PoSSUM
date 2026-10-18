"""Resumable compact-encoder experiments with calibration and a sealed topic holdout."""
import argparse,datetime,gc,json,time
from pathlib import Path
import numpy as np
from .checkpoints import atomic_write,save_json,save_scores
from .improvement import load_data,sample_weights,calibration,report_predictions,digest,PARTS
from .revisions import resolve

TRIALS=[
 dict(name='bert_lora_human',mode='lora',regime='human',lr=2e-4,epochs=4,seed=42),
 dict(name='bert_full_human',mode='full',regime='human',lr=2e-5,epochs=4,seed=42),
 dict(name='bert_full_mixed',mode='full',regime='mixed',lr=2e-5,epochs=4,seed=42),
 dict(name='bert_lora_mixed',mode='lora',regime='mixed',lr=1e-4,epochs=4,seed=42),
]
MODERN_TRIALS=[
 dict(name='modernbert_full_human',model_id='answerdotai/ModernBERT-base',mode='full',regime='human',lr=2e-5,epochs=4,seed=42),
 dict(name='modernbert_full_mixed',model_id='answerdotai/ModernBERT-base',mode='full',regime='mixed',lr=2e-5,epochs=4,seed=42),
]

def run(root, family='bert'):
 import torch,joblib,fcntl
 from transformers import AutoTokenizer,AutoModelForSequenceClassification
 from peft import get_peft_model,LoraConfig,TaskType,get_peft_model_state_dict,set_peft_model_state_dict
 lock=(root/'compact-process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert torch.backends.mps.is_available(),'MPS unavailable'
 torch.set_num_threads(4)
 parts=load_data(root);deadline=datetime.datetime.fromisoformat(json.loads((root/'state.json').read_text())['deadline']).timestamp()
 signature=json.loads((root/'dataset.json').read_text())['manifest_sha256']
 trials=TRIALS if family=='bert' else MODERN_TRIALS
 directory=root/'compact';directory.mkdir(exist_ok=True)
 registry_path=directory/'report.json';registry=json.loads(registry_path.read_text()) if registry_path.exists() else {}
 for trial in trials:
  if time.time()>=deadline:break
  out=directory/trial['name'];out.mkdir(exist_ok=True)
  if (out/'complete.json').exists():continue
  name=trial.get('model_id','google-bert/bert-base-uncased');revision=resolve(name)
  tok=AutoTokenizer.from_pretrained(name,revision=revision,local_files_only=True)
  encoded={p:[tok(r['text'],truncation=False) for r in rr] for p,rr in parts.items()}
  assert max(len(e['input_ids']) for enc in encoded.values() for e in enc)<=256,'No silent truncation'
  torch.manual_seed(trial['seed']);torch.mps.manual_seed(trial['seed'])
  extra={'reference_compile':False} if 'ModernBERT' in name else {}
  model=AutoModelForSequenceClassification.from_pretrained(name,revision=revision,num_labels=2,attn_implementation='eager',local_files_only=True,**extra)
  if trial['mode']=='lora':model=get_peft_model(model,LoraConfig(task_type=TaskType.SEQ_CLS,r=8,lora_alpha=16,lora_dropout=0.,target_modules=['query','value'],modules_to_save=['classifier']))
  model.to('mps');opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=trial['lr'],weight_decay=.01)
  train_indices=np.array([i for i,r in enumerate(parts['fit']) if trial['regime']=='mixed' or r['source']=='MITweet'])
  fit_rows=[parts['fit'][i] for i in train_indices];weights=sample_weights(fit_rows)
  checkpoint=out/'resume.pt';epoch=offset=steps=0;best=float('inf');best_epoch=None
  def save():
   params=get_peft_model_state_dict(model) if trial['mode']=='lora' else model.state_dict()
   state=dict(model={k:v.detach().cpu() for k,v in params.items()},optimizer=opt.state_dict(),epoch=epoch,offset=offset,steps=steps,best=best,best_epoch=best_epoch,
    trial=trial,input_sha256=signature,cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state())
   atomic_write(checkpoint,lambda f:torch.save(state,f))
  if checkpoint.exists():
   state=torch.load(checkpoint,map_location='cpu',weights_only=True);assert state['input_sha256']==signature and state['trial']==trial
   if trial['mode']=='lora':set_peft_model_state_dict(model,state['model'])
   else:model.load_state_dict(state['model'])
   opt.load_state_dict(state['optimizer']);epoch=state['epoch'];offset=state['offset'];steps=state['steps'];best=state['best'];best_epoch=state['best_epoch']
   torch.set_rng_state(state['cpu_rng']);torch.mps.set_rng_state(state['mps_rng']);del state
  total_steps=trial['epochs']*int(np.ceil(len(train_indices)/8));stop=False
  for ep in range(epoch,trial['epochs']):
   epoch=ep;order=np.random.default_rng(trial['seed']+ep).permutation(len(train_indices));model.train();losses=[]
   try:
    for start in range(offset,len(order),8):
     if time.time()>=deadline:stop=True;break
     local=order[start:start+8];indices=train_indices[local]
     batch=tok.pad([encoded['fit'][int(i)] for i in indices],padding=True,return_tensors='pt').to('mps')
     y=torch.tensor([parts['fit'][int(i)]['label'] for i in indices],device='mps')
     w=torch.tensor(weights[local],dtype=torch.float32,device='mps')
     warmup=max(1,int(.1*total_steps));factor=min((steps+1)/warmup,max(.05,(total_steps-steps)/max(1,total_steps-warmup)))
     for group in opt.param_groups:group['lr']=trial['lr']*factor
     opt.zero_grad();loss=(torch.nn.functional.cross_entropy(model(**batch).logits,y,reduction='none')*w).mean()
     if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
     loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();torch.mps.synchronize()
     losses.append(float(loss.detach().cpu()));steps+=1;offset=start+len(local)
     if steps%50==0:
      save();progress=dict(trial=trial['name'],epoch=ep+1,completed=offset,total=len(order),steps=steps,loss=float(np.mean(losses[-50:])),updated_at=time.time())
      save_json(root/'compact-progress.json',progress);print(json.dumps(progress),flush=True)
   finally:save()
   if stop:break
   model.eval();raw={}
   with torch.inference_mode():
    for part in PARTS[1:]:
     scores=[]
     for start in range(0,len(parts[part]),16):
      if time.time()>=deadline:stop=True;break
      batch=tok.pad(encoded[part][start:start+16],padding=True,return_tensors='pt').to('mps')
      scores.extend(model(**batch).logits.detach().cpu().numpy().tolist())
     if stop:break
     z=np.array(scores);raw[part]=z[:,1]-z[:,0]
   if stop:break
   calibrator=calibration(raw['calibration'],parts['calibration']);probs={p:calibrator.predict_proba(raw[p][:,None])[:,1] for p in PARTS[1:]}
   report=report_predictions(parts['validation'],probs['validation']);score=report['sources']['MITweet']['log_loss']
   result=dict(trial=trial,epoch=ep+1,fit_rows=len(fit_rows),validation=report,input_sha256=signature)
   registry[f'{trial["name"]}_epoch{ep+1}']=result;save_json(registry_path,registry)
   print(trial['name'],ep+1,json.dumps(report['sources']['MITweet']),flush=True)
   if score<best:
    version=out/f'epoch-{ep+1}';version.mkdir(exist_ok=True);model.save_pretrained(version,safe_serialization=True);tok.save_pretrained(version)
    atomic_write(version/'calibrator.joblib',lambda f:joblib.dump(calibrator,f));save_json(version/'evaluation.json',result)
    save_scores(version/'scores.npz',np.concatenate([probs[p] for p in PARTS[1:]]),[r['id'] for p in PARTS[1:] for r in parts[p]],[])
    best=score;best_epoch=ep+1;save_json(out/'best.json',dict(folder=version.name,log_loss=best,epoch=best_epoch,trial=trial,input_sha256=signature))
   epoch=ep+1;offset=0;save()
  if not stop:save_json(out/'complete.json',dict(best=best,best_epoch=best_epoch,completed_at=time.time(),input_sha256=signature))
  del model,opt;gc.collect();torch.mps.empty_cache()
  if stop:break
 finished='compact-finished.json' if family=='bert' else 'modernbert-finished.json'
 save_json(root/finished,dict(complete=all((directory/t['name']/'complete.json').exists() for t in trials),updated_at=time.time()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--family',choices=['bert','modernbert'],default='bert');a=p.parse_args();run(a.run,a.family)
