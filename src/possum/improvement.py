"""Comparable compact classifiers on versioned human-facet development data."""
import argparse,hashlib,json,time
from pathlib import Path
import joblib
import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from .checkpoints import atomic_write,save_json,save_scores
from .data import read_jsonl
from .learning import metrics

PARTS=('fit','calibration','gate','validation')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_data(root):
 meta=json.loads((root/'dataset.json').read_text());path=Path(meta['path']);manifest=json.loads((path/'manifest.json').read_text())
 assert digest(path/'manifest.json')==meta['manifest_sha256'],'Dataset manifest changed'
 out={}
 for part in PARTS:
  assert digest(path/f'{part}.jsonl')==manifest['split_hashes'][part]
  out[part]=read_jsonl(path/f'{part}.jsonl')
 return out

def sample_weights(rows):
 from collections import Counter
 counts=Counter((r['source'],r['label']) for r in rows)
 return np.array([len(rows)/(len(counts)*counts[(r['source'],r['label'])]) for r in rows])

def calibration(raw,rows):
 mask=np.array([r['source']=='MITweet' for r in rows]);y=np.array([r['label'] for r in rows])
 assert set(y[mask])=={0,1}
 return LogisticRegression(C=1,max_iter=500,random_state=42).fit(np.asarray(raw)[mask,None],y[mask])

def report_predictions(rows,p):
 y=np.array([r['label'] for r in rows]);out={'all':dict(n=len(rows),**metrics(y,p)),'sources':{},'topics':{}}
 for field,key in [('source','sources'),('topic','topics')]:
  for name in sorted({r[field] for r in rows}):
   mask=np.array([r[field]==name for r in rows]);yy=y[mask]
   if len(set(yy))==2:out[key][name]=dict(n=int(mask.sum()),**metrics(yy,p[mask]))
   else:out[key][name]={'n':int(mask.sum()),'single_class':int(yy[0]),'accuracy':float(((p[mask]>=.5)==yy).mean())}
 return out

def require_development_open(root):
 if (root/'selection-freeze.json').exists():raise RuntimeError('Candidate selection is frozen; further development fitting is disabled for this run')

def baseline(root):
 require_development_open(root)
 parts=load_data(root);out=root/'baselines';out.mkdir(exist_ok=True)
 report_path=out/'report.json';results=json.loads(report_path.read_text()) if report_path.exists() else {}
 embeddings={}
 for name in ['bge','bert']:
  z=np.load(root/f'{name}_features.npz',allow_pickle=False);lookup={k:i for i,k in enumerate(z['ids'].tolist())}
  embeddings[name]={p:z['values'][[lookup[r['id']] for r in rr]] for p,rr in parts.items()}
 for regime in ['human','mixed']:
  indices=np.array([i for i,r in enumerate(parts['fit']) if regime=='mixed' or r['source']=='MITweet'])
  rows=[parts['fit'][i] for i in indices];y=np.array([r['label'] for r in rows]);weights=sample_weights(rows)
  for features in ['lexical','bge','bert']:
   if features=='lexical':
    vectorizer=FeatureUnion([('words',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=30000,sublinear_tf=True)),('chars',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,max_features=40000,sublinear_tf=True))])
    vectorizer.fit([r['text'] for r in rows]);matrix={p:vectorizer.transform([r['text'] for r in rr]) for p,rr in parts.items()}
   else:vectorizer=None;matrix=embeddings[features]
   for c in [.1,1.,10.]:
    name=f'{features}_{regime}_C{c:g}'
    if name in results:continue
    config=json.loads((root/'state.json').read_text())
    import datetime
    if time.time()>=datetime.datetime.fromisoformat(config['deadline']).timestamp():return
    started=time.perf_counter();head=LogisticRegression(C=c,max_iter=1000,random_state=42).fit(matrix['fit'][indices],y,sample_weight=weights)
    scores={p:head.decision_function(matrix[p]) for p in PARTS[1:]};cal=calibration(scores['calibration'],parts['calibration'])
    probs={p:cal.predict_proba(scores[p][:,None])[:,1] for p in PARTS[1:]}
    model={'features':features,'head':head,'vectorizer':vectorizer,'calibrator':cal,'regime':regime,'C':c,'input_sha256':json.loads((root/'dataset.json').read_text())['manifest_sha256']}
    atomic_write(out/f'{name}.joblib',lambda f:joblib.dump(model,f))
    save_scores(out/f'{name}_scores.npz',np.concatenate([probs[p] for p in PARTS[1:]]),[r['id'] for p in PARTS[1:] for r in parts[p]],[time.perf_counter()-started])
    results[name]={'validation':report_predictions(parts['validation'],probs['validation']),'seconds':time.perf_counter()-started,'fit_rows':len(rows),'calibration':'L2-regularized Platt on human calibration only'}
    save_json(report_path,results);print(name,json.dumps(results[name]['validation']['sources']['MITweet']),flush=True)
 ranked=sorted(results,key=lambda n:results[n]['validation']['sources']['MITweet']['log_loss'])
 save_json(out/'selection.json',{'primary_metric':'MITweet validation log_loss','selected':ranked[0],'ranked':ranked,'topic_holdout_scored':False})

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['baseline']);p.add_argument('run',type=Path);args=p.parse_args()
 baseline(args.run)

if __name__=='__main__':main()
