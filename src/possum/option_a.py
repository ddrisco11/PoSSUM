"""Adapter training and held-out stacking workers for a bounded local run."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from .checkpoints import atomic_write, save_json, save_scores
from .data import read_jsonl
from .revisions import resolve

MODELS = {
    'mistral': ('mlx-community/Mistral-7B-v0.3-4bit','5e8b40756eaa1cb5131715937c77107b4421c3a5'),
    'qwen': ('mlx-community/Qwen3.5-9B-4bit','8b2b98c00a6b4d291155e4890773ca8f769aee53'),
}
LORA = {'rank':8,'scale':16.,'dropout':0.,'keys':['self_attn.q_proj','self_attn.v_proj']}
PARTS = ('fit','calibration','gate','validation')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data(root):
    return {p:read_jsonl(root/'input'/f'{p}.jsonl') for p in PARTS}


def update(root, stage, **fields):
    path=root/f'{stage}_progress.json'
    old=json.loads(path.read_text()) if path.exists() else {}
    old.update(fields,updated_at=time.time(),stage=stage)
    save_json(path,old)
    if 'completed' in fields:
        print(f"{stage}: {fields['completed']}/{fields.get('total','?')} {fields.get('status','running')}",flush=True)


def guard(root):
    control=json.loads((root/'control.json').read_text())
    if time.time()>=control['stage_deadline']:raise TimeoutError('Stage deadline reached')
    if not control.get('require_ac',True):return
    while "'AC Power'" not in subprocess.check_output(['pmset','-g','batt'],text=True):
        save_json(root/'power.json',dict(status='waiting_for_ac',updated_at=time.time()))
        if time.time()>=control['stage_deadline']:raise TimeoutError('Deadline reached while waiting for power')
        time.sleep(10)
    save_json(root/'power.json',dict(status='ac',updated_at=time.time()))


def durable_file(path, save):
    """Serialize through a temporary path and use durable atomic replacement."""
    temporary=path.with_name('.'+path.stem+'.tmp'+path.suffix)
    try:
        save(temporary)
        atomic_write(path,lambda f:f.write(temporary.read_bytes()))
    finally:
        temporary.unlink(missing_ok=True)


def train_mistral(root):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten,tree_unflatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.trainer import grad_checkpoint
    from huggingface_hub import snapshot_download
    from .components.causal import prompt,candidate_ids
    stage='mistral_train';out=root/'mistral_adapter';out.mkdir(exist_ok=True)
    rows=data(root)['fit'];repo,rev=MODELS['mistral'];mx.random.seed(42)
    mx.set_cache_limit(256*1024**2)
    model,tok=load(snapshot_download(repo,revision=rev,local_files_only=True));model.freeze()
    linear_to_lora_layers(model,4,LORA);grad_checkpoint(model.layers[-1]);model.train()
    opt=optim.Adam(learning_rate=5e-5)
    signature=digest(root/'input/fit.jsonl')
    order=np.random.default_rng(42).permutation(len(rows)).tolist()
    counts=Counter(r['label'] for r in rows);weights={y:len(rows)/(2*n) for y,n in counts.items()}
    def encode(row):
        pairs=[candidate_ids(tok,prompt(row['text']),c) for c in (' left',' right')]
        if any(len(b)!=len(a)+1 for a,b in pairs):raise ValueError('Training requires verified single-token candidates')
        if len(pairs[0][0])>256:raise ValueError('Input exceeds profiled 256-token limit; no silent truncation')
        return mx.array([pairs[0][0]]),mx.array([b[-1] for _,b in pairs]),mx.array([row['label']])
    def loss(m,x,labels,y,w):
        return nn.losses.cross_entropy(m(x)[:,-1,labels].astype(mx.float32),y).mean()*w
    value_grad=nn.value_and_grad(model,loss)
    def step(index):
        row=rows[order[index]];x,labels,y=encode(row)
        value,grad=value_grad(model,x,labels,y,weights[row['label']]);opt.update(model,grad)
        mx.eval(model.parameters(),opt.state,value)
        result=float(value)
        if not np.isfinite(result):raise ValueError('Nonfinite training loss')
        mx.clear_cache();return result
    def checkpoint(n,seconds):
        folder=out/f'step-{n:06d}';folder.mkdir(exist_ok=True)
        durable_file(folder/'adapters.safetensors',lambda p:mx.save_safetensors(str(p),dict(tree_flatten(model.trainable_parameters()))))
        state={k:mx.array(v) for k,v in tree_flatten(opt.state)}
        durable_file(folder/'optimizer.safetensors',lambda p:mx.save_safetensors(str(p),state))
        save_json(folder/'state.json',dict(step=n,seconds=seconds,input_sha256=signature,seed=42,loss='class-balanced binary next-token cross entropy'))
        save_json(out/'latest.json',{'folder':folder.name})
        # Keep two completed recovery versions; published adapter remains separate.
        versions=sorted(out.glob('step-*'))
        import shutil
        for old in versions[:-2]:shutil.rmtree(old)
        return folder
    def restore(folder):
        info=json.loads((folder/'state.json').read_text())
        if info['input_sha256']!=signature:raise ValueError('Training split changed')
        model.load_weights(str(folder/'adapters.safetensors'),strict=False)
        state=mx.load(str(folder/'optimizer.safetensors'))
        opt.state=tree_unflatten(list(state.items()));return info
    n=0;elapsed=0.;resume_verified=False
    if (out/'latest.json').exists():
        info=restore(out/json.loads((out/'latest.json').read_text())['folder']);n=info['step'];elapsed=info['seconds']
        resume_verified=(out/'resume-check.json').exists()
    if n==0 and len(rows)>=2:
        guard(root);t=time.perf_counter();step(0);elapsed+=time.perf_counter()-t;n=1
        folder=checkpoint(n,elapsed);step(1)
        expected={k:np.array(v) for k,v in tree_flatten(model.trainable_parameters())}
        restore(folder);step(1)
        maximum=max(float(np.max(np.abs(np.array(v)-expected[k]))) for k,v in tree_flatten(model.trainable_parameters()))
        if maximum>1e-6:raise ValueError(f'Optimizer resume mismatch: {maximum}')
        restore(folder);resume_verified=True
        save_json(out/'resume-check.json',dict(passed=True,max_parameter_difference=maximum,scope='One update after optimizer+adapter reload with deterministic order and zero dropout on identical next example'))
    partial=False;losses=[];last_save=n
    try:
        for i in range(n,len(rows)):
            if i%25==0:guard(root)
            t=time.perf_counter();losses.append(step(i));elapsed+=time.perf_counter()-t;n=i+1
            if n%50==0 or n==len(rows):
                checkpoint(n,elapsed);last_save=n
                update(root,stage,completed=n,total=len(rows),seconds=elapsed,seconds_per_excerpt=elapsed/n,loss=float(np.mean(losses[-50:])),status='training',peak_mlx_gib=mx.get_peak_memory()/1024**3)
    except TimeoutError:
        partial=True
    finally:
        if n!=last_save:checkpoint(n,elapsed)
    if n==0:raise RuntimeError('No adapter update completed')
    final=out/json.loads((out/'latest.json').read_text())['folder']
    atomic_write(out/'adapters.safetensors',lambda f:f.write((final/'adapters.safetensors').read_bytes()))
    save_json(out/'adapter_config.json',dict(model=repo,fine_tune_type='lora',num_layers=4,lora_parameters=LORA))
    save_json(out/'training.json',dict(model_id=repo,revision=rev,input_sha256=signature,examples_seen=n,total_fit=len(rows),full_pass=n==len(rows),partial=partial,seconds=elapsed,resume_verified=resume_verified,adapter_sha256=digest(out/'adapters.safetensors'),trainable_parameters=sum(v.size for _,v in tree_flatten(model.trainable_parameters()))))
    model.eval();sample=encode(rows[0]);before=np.array(model(sample[0])[:,-1,sample[1]])
    # Reload into this existing adapter model would re-wrap layers: use saved tensor reload here.
    model.load_weights(str(out/'adapters.safetensors'),strict=False)
    after=np.array(model(sample[0])[:,-1,sample[1]])
    if not np.allclose(before,after,atol=1e-5):raise ValueError('Adapter serialization check failed')
    update(root,stage,completed=n,total=len(rows),seconds=elapsed,status='partial' if partial else 'complete')


def bert_model(adapter=None):
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    from peft import get_peft_model,LoraConfig,TaskType,PeftModel
    name='google-bert/bert-base-uncased';rev=resolve(name)
    model=AutoModelForSequenceClassification.from_pretrained(name,revision=rev,num_labels=2,attn_implementation='eager',local_files_only=True)
    if adapter:model=PeftModel.from_pretrained(model,str(adapter))
    else:model=get_peft_model(model,LoraConfig(task_type=TaskType.SEQ_CLS,r=8,lora_alpha=16,lora_dropout=0.,target_modules=['query','value'],modules_to_save=['classifier']))
    tok=AutoTokenizer.from_pretrained(name,revision=rev,local_files_only=True)
    return model.to('mps'),tok


def train_bert(root):
    import torch
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    torch.set_num_threads(4);torch.manual_seed(42);torch.mps.manual_seed(42)
    rows=data(root)['fit'];out=root/'bert_adapter';out.mkdir(exist_ok=True)
    model,tok=bert_model();model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=.01)
    order=np.random.default_rng(42).permutation(len(rows));n=0;elapsed=0.;losses=[]
    counts=Counter(r['label'] for r in rows);weights=torch.tensor([len(rows)/(2*counts[k]) for k in (0,1)],device='mps')
    signature=digest(root/'input/fit.jsonl');path=out/'training-state.pt'
    if path.exists():
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state['input_sha256']!=signature:raise ValueError('BERT training split changed')
        set_peft_model_state_dict(model,state['adapter']);opt.load_state_dict(state['optimizer']);n=state['step'];elapsed=state['seconds'];torch.set_rng_state(state['cpu_rng']);torch.mps.set_rng_state(state['mps_rng'])
    def save():
        state=dict(adapter={k:v.detach().cpu() for k,v in get_peft_model_state_dict(model).items()},optimizer=opt.state_dict(),step=n,seconds=elapsed,cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state(),input_sha256=signature)
        atomic_write(path,lambda f:torch.save(state,f))
    partial=False
    try:
        for start in range(n,len(rows),8):
            if start%200==0:guard(root)
            batch=[rows[int(i)] for i in order[start:start+8]];t=time.perf_counter()
            encoded=tok([r['text'] for r in batch],padding=True,truncation=False,return_tensors='pt').to('mps')
            if encoded['input_ids'].shape[1]>256:raise ValueError('BERT input exceeds 256 tokens')
            y=torch.tensor([r['label'] for r in batch],device='mps');opt.zero_grad()
            loss=torch.nn.functional.cross_entropy(model(**encoded).logits,y,weight=weights)
            if not torch.isfinite(loss):raise ValueError('Nonfinite BERT loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();torch.mps.synchronize()
            elapsed+=time.perf_counter()-t;n=start+len(batch);losses.append(float(loss.detach().cpu()))
            if n%200==0 or n==len(rows):
                save();update(root,'bert_train',completed=n,total=len(rows),seconds=elapsed,loss=float(np.mean(losses[-25:])),status='training')
    except TimeoutError:partial=True
    finally:save()
    model.save_pretrained(out,safe_serialization=True)
    save_json(out/'training.json',dict(model_id='google-bert/bert-base-uncased',revision=resolve('google-bert/bert-base-uncased'),input_sha256=signature,examples_seen=n,total_fit=len(rows),full_pass=n==len(rows),partial=partial,seconds=elapsed,adapter_sha256=digest(out/'adapter_model.safetensors'),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    model.eval();probe=tok(rows[0]['text'],return_tensors='pt').to('mps')
    with torch.inference_mode():before=model(**probe).logits.detach().cpu().numpy()
    # Verify serialized tensors reload without changing logits.
    from safetensors.torch import load_file
    set_peft_model_state_dict(model,load_file(str(out/'adapter_model.safetensors')))
    with torch.inference_mode():after=model(**probe).logits.detach().cpu().numpy()
    if not np.allclose(before,after,atol=1e-5):raise ValueError('BERT adapter reload mismatch')
    update(root,'bert_train',completed=n,total=len(rows),seconds=elapsed,status='partial' if partial else 'complete')


def score(root,name):
    rows_by_part=data(root)
    rows=sum([rows_by_part[p] for p in PARTS if name=='bge' or p!='fit'],[])
    ids=[r['id'] for r in rows];stage=f'{name}_score';target=root/f'{name}_scores.npz'
    signature={'input_sha256':{p:digest(root/'input'/f'{p}.jsonl') for p in PARTS},'name':name}
    adapter=None
    if name in ('mistral','bert'):
        adapter=root/f'{name}_adapter'
        if not (adapter/'training.json').exists():raise ValueError(f'{name} adapter not ready')
        signature['training']=json.loads((adapter/'training.json').read_text())
    values=[];elapsed=[]
    meta=target.with_suffix('.json')
    if target.exists():
        if json.loads(meta.read_text())['signature']!=signature:raise ValueError('Stale score cache')
        with np.load(target,allow_pickle=False) as old:
            if old['ids'].tolist()!=ids[:len(old['ids'])]:raise ValueError('Score cache ID mismatch')
            values=old['values'].tolist();elapsed=old['chunk_seconds'].tolist()
    if len(values)==len(rows):
        save_json(meta,{'signature':signature,'complete':True})
        update(root,stage,completed=len(values),total=len(rows),seconds=sum(elapsed),status='complete')
        return
    save_json(meta,{'signature':signature,'complete':False})
    guard(root)
    if name in MODELS:
        from huggingface_hub import snapshot_download
        from .components.causal import MLXCausal
        from mlx_lm import load
        repo,rev=MODELS[name];path=snapshot_download(repo,revision=rev,local_files_only=True)
        model=MLXCausal.__new__(MLXCausal)
        model.model,model.tokenizer=load(path,adapter_path=str(adapter) if adapter else None)
        model.model.eval()
        predict=model.score
    elif name=='bert':
        import torch
        model,tok=bert_model(adapter);model.eval()
        def predict(texts):
            encoded=tok(texts,padding=True,truncation=False,return_tensors='pt').to('mps')
            if encoded['input_ids'].shape[1]>256:raise ValueError('BERT scoring length exceeds 256')
            with torch.inference_mode():return model(**encoded).logits.detach().cpu().numpy()
    elif name=='gliner':
        import torch
        torch.set_num_threads(4)
        from .components.gliner import GLiNER
        model=GLiNER();predict=model.score
    else:
        from .components.bge import BGE
        model=BGE(device='mps');predict=model.encode
    size=8 if name in ('bert','bge') else 10
    for start in range(len(values),len(rows),size):
        guard(root);t=time.perf_counter();result=predict([r['text'] for r in rows[start:start+size]])
        if not np.isfinite(result).all():raise ValueError('Nonfinite scores')
        values.extend(result.tolist());elapsed.append(time.perf_counter()-t)
        save_scores(target,values,ids[:len(values)],elapsed)
        update(root,stage,completed=len(values),total=len(rows),seconds=sum(elapsed),seconds_per_excerpt=sum(elapsed)/len(values),status='scoring')
    save_json(meta,{'signature':signature,'complete':True})
    update(root,stage,completed=len(values),total=len(rows),seconds=sum(elapsed),status='complete')


def evaluate(root):
    import joblib
    import torch
    from scipy.special import softmax
    from sklearn.linear_model import LogisticRegression
    from .learning import temperature,metrics
    torch.set_num_threads(4)
    parts=data(root);rows=sum([parts[p] for p in PARTS[1:]],[]);ids=[r['id'] for r in rows]
    positions={};offset=0
    for part in PARTS[1:]:positions[part]=np.arange(offset,offset+len(parts[part]));offset+=len(parts[part])
    c,g,v=[positions[p] for p in PARTS[1:]];y=np.array([r['label'] for r in rows])
    names=['bert','gliner','mistral','qwen'];raw={};temps={}
    for name in names:
        target=root/f'{name}_scores.npz'
        if not json.loads(target.with_suffix('.json').read_text())['complete']:raise ValueError('Incomplete scores')
        with np.load(target,allow_pickle=False) as z:
            if z['ids'].tolist()!=ids:raise ValueError('Score alignment mismatch')
            raw[name]=z['values']
        temps[name]=temperature(raw[name][c],y[c])
    with np.load(root/'bge_scores.npz',allow_pickle=False) as z:
        expected=[r['id'] for r in parts['fit']]+ids
        if z['ids'].tolist()!=expected:raise ValueError('BGE alignment mismatch')
        xf=z['values'][:len(parts['fit'])];x=z['values'][len(parts['fit']):]
    own=LogisticRegression(C=.1,max_iter=500,class_weight='balanced',random_state=42).fit(xf,[r['label'] for r in parts['fit']])
    own_raw=np.log(np.clip(own.predict_proba(x),1e-7,1));own_t=temperature(own_raw[c],y[c]);own_p=softmax(own_raw/own_t,axis=1)[:,1]
    probabilities=np.stack([softmax(raw[n]/temps[n],axis=1)[:,1] for n in names],axis=1)
    out=root/'export';out.mkdir(exist_ok=True)
    atomic_write(out/'self_head.joblib',lambda f:joblib.dump(own,f))
    predictions={n:probabilities[v,j] for j,n in enumerate(names)};predictions['equal']=probabilities[v].mean(1);predictions['self_only']=own_p[v]
    models={};all_metrics={};weights={}
    for seed in (42,43,44):
        for mode in ('fixed','router','router_self'):
            torch.manual_seed(seed);k=5 if mode=='router_self' else 4
            model=torch.nn.Linear(384,k,bias=True)
            torch.nn.init.normal_(model.weight,std=.001);torch.nn.init.zeros_(model.bias)
            if mode=='fixed':model.weight.requires_grad_(False);model.weight.data.zero_()
            xx=torch.tensor(x[g],dtype=torch.float32);pp=torch.tensor(probabilities[g],dtype=torch.float32);yy=torch.tensor(y[g],dtype=torch.float32);ss=torch.tensor(own_p[g],dtype=torch.float32)
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.005,weight_decay=.1)
            for epoch in range(100):
                w=router_weights(model,xx,mode);combined=torch.cat([pp,ss[:,None]],1) if k==5 else pp
                pred=(w*combined).sum(1).clamp(1e-6,1-1e-6)
                penalty=(w*torch.log((w*k).clamp_min(1e-6))).sum(1).mean()
                loss=torch.nn.functional.binary_cross_entropy(pred,yy)+.05*penalty
                opt.zero_grad();loss.backward();opt.step()
            key=f'{mode}_{seed}'
            with torch.inference_mode():
                w=router_weights(model,torch.tensor(x[v],dtype=torch.float32),mode).numpy()
            ppv=np.column_stack([probabilities[v],own_p[v]]) if k==5 else probabilities[v]
            predictions[key]=(w*ppv).sum(1);weights[key]=dict(zip(names+(['self'] if k==5 else []),w.mean(0).tolist()))
            atomic_write(out/f'{key}.pt',lambda f:torch.save(model.state_dict(),f));models[key]=mode
    for j,name in enumerate(names):predictions[f'equal_without_{name}']=np.delete(probabilities[v],j,axis=1).mean(1)
    predictions['compact_equal']=probabilities[v,:2].mean(1)
    all_metrics={k:metrics(y[v],pred) for k,pred in predictions.items()}
    # Deployment selection uses the declared development metric, never locked test.
    selectable=['equal','compact_equal',*models]
    selected=min(selectable,key=lambda k:all_metrics[k]['log_loss'])
    report=dict(status='development_evaluation_complete',scope='Weak/synthetic development validation; no locked test scored',counts={p:len(parts[p]) for p in PARTS},validation_groups=len({r['group'] for r in parts['validation']}),metrics=all_metrics,temperatures=temps,self_temperature=own_t,mean_weights=weights,selected=selected,selection_metric='validation log_loss',self_branch='Fit-partition BGE logistic head, frozen for router fitting; self weight capped at0.25',routing_regularization=dict(epochs=100,weight_decay=.1,uniform_weight_kl=.05),by_source={source:{k:metrics(y[v][mask],pred[mask]) for k,pred in predictions.items()} for source in sorted({r['source'] for r in parts['validation']}) if (mask:=np.array([r['source']==source for r in parts['validation']])).any()},training={n:json.loads((root/f'{n}_adapter/training.json').read_text()) for n in ('bert','mistral')},limitations=['All source labels in development are weak or synthetic; no accuracy guarantee for new human text.','Validation used for development selection; three router seeds share one split.','Confidence-based abstention and neutral/nonpolitical detection are not validated.'])
    config=dict(version=root.name,experts=names,temperatures=temps,self_temperature=own_t,selected=selected,router_mode=models.get(selected),backbones={**MODELS,'bert':['google-bert/bert-base-uncased',resolve('google-bert/bert-base-uncased')],'gliner':['fastino/gliner2.5-base-v1',resolve('fastino/gliner2.5-base-v1')],'bge':['BAAI/bge-small-en-v1.5',resolve('BAAI/bge-small-en-v1.5')]},bert_adapter='../bert_adapter',mistral_adapter='../mistral_adapter',status='research prototype; unvalidated for production')
    save_json(out/'config.json',config)
    for target in [root/'report.json',out/'validation.json']:save_json(target,report)
    # Fresh file reload path compared with evaluation arithmetic.
    for key,mode in models.items():
        m=torch.nn.Linear(384,5 if mode=='router_self' else 4)
        m.load_state_dict(torch.load(out/f'{key}.pt',weights_only=True));m.eval()
        with torch.inference_mode():w=router_weights(m,torch.tensor(x[v],dtype=torch.float32),mode).numpy()
        combined=np.column_stack([probabilities[v],own_p[v]]) if mode=='router_self' else probabilities[v]
        if not np.allclose((w*combined).sum(1),predictions[key],atol=1e-6):raise ValueError('Router export reload mismatch')
    save_json(out/'reload-check.json',dict(passed=True,router_count=len(models),sample_count=len(v)))
    update(root,'evaluate',completed=1,total=1,status='complete')
    print(json.dumps({'selected':selected,'metrics':all_metrics},indent=2),flush=True)


def router_weights(model,x,mode):
    import torch
    logits=model(x)
    if mode=='router_self':
        own=.25*torch.sigmoid(logits[:,-1:])
        return torch.cat([torch.softmax(logits[:,:4],dim=-1)*(1-own),own],dim=1)
    return torch.softmax(logits,dim=-1)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('run',type=Path);a=p.parse_args()
    if a.stage=='mistral_train':train_mistral(a.run)
    elif a.stage=='bert_train':train_bert(a.run)
    elif a.stage=='evaluate':evaluate(a.run)
    elif a.stage.endswith('_score'):score(a.run,a.stage.removesuffix('_score'))
    else:raise ValueError('Unknown stage')


if __name__=='__main__':main()
