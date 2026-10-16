"""Source-held-out diagnostics using the frozen compact-pilot recipe.
External labels are used only for metrics, never optimization or calibration.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from possum.data import prepare,read_jsonl,normalize
from possum.learning import temperature,metrics,fit_neural,fit_gate,predict_gate
from possum.components.bge import BGE
from possum.components.bert import BERT
from possum.components.gliner import GLiNER


def screen_overlap(external,training,threshold=.93):
    exact={normalize(r['text']) for r in training}
    external=[r for r in external if normalize(r['text']) not in exact]
    if not external:return [],0
    vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),max_features=40000)
    matrix=vec.fit_transform([normalize(r['text']) for r in training+external])
    distance,_=NearestNeighbors(metric='cosine',n_neighbors=1).fit(matrix[:len(training)]).kneighbors(matrix[len(training):])
    keep=distance[:,0] > 1-threshold
    return [r for i,r in enumerate(external) if keep[i]],int((~keep).sum())


def scored(y,p):
    return dict(**metrics(y,p),confusion_matrix_left_right=confusion_matrix(y,p>=.5,labels=[0,1]).tolist(),predicted_right_fraction=float((p>=.5).mean()))


def run(args):
    torch.set_num_threads(4)
    out=Path(args.output)
    if (out/'report.json').exists():raise ValueError('Completed output exists; use a new directory')
    out.mkdir(parents=True,exist_ok=True)
    root=Path(args.pilot)
    original=json.loads((root/'report.json').read_text())
    assert original['config']['seed']==42,'This recipe reproduces pilot seed 42'
    membership=json.loads((root/'split.json').read_text())['pilot']
    ids=sum(membership.values(),[])
    synthetic=read_jsonl(args.training)
    lookup={hashlib.sha256(normalize(r['text']).encode()).hexdigest():r for r in synthetic}
    dev=[lookup[i] for i in ids]
    positions={};offset=0
    for k,v in membership.items():
        positions[k]=np.arange(offset,offset+len(v));offset+=len(v)
    a,c,g=(positions[k] for k in ('fit','calibration','gate'))
    cached=np.load(root/'scores.npz',allow_pickle=False)
    y=cached['labels'];x=cached['embeddings'];train_probs=cached['probabilities']
    if not np.array_equal(y,[r['label'] for r in dev]):raise ValueError('Training cache order mismatch')
    raw=read_jsonl(args.external)
    clean=prepare(raw)
    external,near_removed=screen_overlap(clean,synthetic)
    if not external:raise ValueError('No unseen evaluation rows')
    texts=[r['text'] for r in external]
    ey=np.array([r['label'] for r in external])
    names=cached['names'].tolist()
    (out/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in external))
    print(f'Evaluating {len(external)} unseen passages, by source: '+str({s:sum(r['source']==s for r in external) for s in {r['source'] for r in external}}),flush=True)
    raw_scores={};runtime={};revisions={};truncations={}
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=15000)
    features=vec.fit_transform([dev[i]['text'] for i in a])
    clf=LogisticRegression(C=1,max_iter=500).fit(features,y[a])
    raw_scores['tfidf']=np.log(np.clip(clf.predict_proba(vec.transform(texts)),1e-7,1))
    ex=None
    for name,cls in [('bge',BGE),('bert',BERT)]:
        print('Scoring '+name,flush=True);start=time.perf_counter()
        model=cls(device=args.device,revision=original['checkpoint_revisions'][name])
        revisions[name]=model.revision
        lengths=[len(model.tokenizer.encode(t,add_special_tokens=True)) for t in texts]
        truncations[name]=sum(n>256 for n in lengths)
        z=model.encode(texts)
        np.save(out/f'{name}_features.npy',z)
        fit=np.load(root/f'{name}_features.npy')
        clf=LogisticRegression(C=1,max_iter=500).fit(fit[a],y[a])
        raw_scores[name]=np.log(np.clip(clf.predict_proba(z),1e-7,1))
        if name=='bge':ex=z
        runtime[name]=time.perf_counter()-start
        del model;gc.collect()
        if torch.backends.mps.is_available():torch.mps.empty_cache()
    neural=fit_neural(x[a],y[a],42)
    with torch.inference_mode():raw_scores['bge_neural']=neural(torch.tensor(ex,dtype=torch.float32)).numpy()
    temperatures=dict(original['temperature'])
    # GLiNER temperature uses only the original synthetic calibration partition.
    gliner_cache=np.load(args.gliner_cache,allow_pickle=False)
    gliner_lookup=dict(zip(gliner_cache['ids'].tolist(),gliner_cache['values']))
    gs=np.array([gliner_lookup[i] for i in ids])
    t=temperature(gs[c],y[c]);temperatures['gliner']=t
    if "gliner" not in names:
        train_probs=np.column_stack([train_probs,softmax(gs/t,axis=1)[:,1]])
    print('Scoring gliner',flush=True);start=time.perf_counter()
    model=GLiNER();revisions['gliner']=model.revision
    gs_external=model.score(texts)
    runtime['gliner']=time.perf_counter()-start
    raw_scores['gliner']=gs_external
    if 'gliner' not in names:names.append('gliner')
    ep=np.stack([softmax(raw_scores[name]/temperatures[name],axis=1)[:,1] for name in names],axis=1)
    del model;gc.collect()
    predictions={name:ep[:,i] for i,name in enumerate(names)}
    predictions['equal_mixture']=ep.mean(1)
    weights={}
    for seed in (42,43,44):
        for kind,fixed,own in [('fixed_mixture',True,False),('expert_gate',False,False),('hybrid_gate',False,True)]:
            model=fit_gate(x[g],train_probs[g],y[g],fixed=fixed,seed=seed,self_branch=own)
            key=f'{kind}_seed{seed}'
            predictions[key]=predict_gate(model,ex,ep)
            with torch.inference_mode():w=model.weights(torch.tensor(ex,dtype=torch.float32)).numpy()
            weights[key]={s:dict(zip(names+(['self'] if own else []),w[np.array([r['source']==s for r in external])].mean(0).tolist())) for s in sorted({r['source'] for r in external})}
    by_source={}
    for source in sorted({r['source'] for r in external}):
        mask=np.array([r['source']==source for r in external])
        by_source[source]=dict(n=int(mask.sum()),label_kind=external[int(np.flatnonzero(mask)[0])]['label_kind'],label_counts_left_right=np.bincount(ey[mask],minlength=2).tolist(),metrics={name:scored(ey[mask],p[mask]) for name,p in predictions.items()})
    np.savez_compressed(out/'predictions.npz',labels=ey,ids=np.array([r['id'] for r in external]),**predictions)
    report=dict(protocol='Frozen synthetic-trained recipe, source-held-out external diagnostic. No external labels used for training, calibration, seed selection or hyperparameter tuning.',training_counts=original['pilot_counts'],external_input_count=len(raw),after_length_and_dedup=len(clean),overlap_removed=len(clean)-len(external),near_overlap_removed=near_removed,by_source=by_source,mean_weights=weights,temperature=temperatures,checkpoint_revisions=revisions,runtime_seconds=runtime,encoder_truncated_rows=truncations,device=args.device,packages={k:importlib.metadata.version(k) for k in ['torch','transformers','gliner2','scikit-learn']},input_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [args.training,args.external]},limitations=['IBC official sample only; full corpus is not included.','Book/author and platform document IDs are unavailable; entire sources are held out, and no independence-based confidence interval is claimed.','Platform labels are party-derived proxies; not human passage-level ideological annotations and may include neutral text.','Historical party labels do not necessarily match contemporary left/right positions.','Source-level holdout cannot rule out backbone pretraining contamination or semantic paraphrase overlap.','Synthetic final-test partition remains untouched. External sets become diagnostic benchmarks after reporting and cannot serve as a future pristine final test.'])
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(by_source,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--pilot',default='runs/synthetic-42')
    p.add_argument('--training',default='data/synthetic.jsonl')
    p.add_argument('--external',default='data/external/external.jsonl')
    p.add_argument('--gliner-cache',default='runs/gliner.npz')
    p.add_argument('--output',default='runs/external-ibc-platform')
    p.add_argument('--device',default='mps',choices=['mps','cpu'])
    run(p.parse_args())
