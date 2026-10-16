"""Held-out stacking pilot. Reserved test rows are never encoded or scored."""
import gc
import hashlib
import json
import platform
import time
from pathlib import Path
import importlib.metadata
import numpy as np
import torch
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .data import prepare, split_rows, read_jsonl
from .learning import temperature, metrics, fit_gate, predict_gate, fit_neural, bootstrap_delta


def run(path, output, components=("bge", "bert"), seed=42, limit=800, device="cpu", expert_scores=()):
    torch.set_num_threads(4)
    out = Path(output)
    if (out/"report.json").exists():
        raise ValueError("Use a new output directory; completed runs are immutable")
    out.mkdir(parents=True, exist_ok=True)
    rows = prepare(read_jsonl(path))
    parts = split_rows(rows, 42)
    # Cap development after reserving test, proportionally across partitions.
    rng = np.random.default_rng(seed)
    dev_total = sum(len(v) for k, v in parts.items() if k != "test")
    chosen = {}
    for k, ids in parts.items():
        if k == "test":
            continue
        count = min(len(ids), max(2, round(limit*len(ids)/dev_total)))
        chosen[k] = np.sort(rng.choice(ids, count, replace=False))
    indexes = np.concatenate(list(chosen.values()))
    dev = [rows[i] for i in indexes]
    remap = {int(old): new for new, old in enumerate(indexes)}
    local = {k: np.array([remap[int(i)] for i in ids]) for k, ids in chosen.items()}
    for k, ids in local.items():
        if len({dev[i]["label"] for i in ids}) < 2:
            raise ValueError(f"Pilot subset {k} lacks both labels; increase limit")
    xtext = [r["text"] for r in dev]
    y = np.array([r["label"] for r in dev])
    a, c, g, v = (local[k] for k in ("fit", "calibration", "gate", "validation"))
    membership = {k: [rows[i]["id"] for i in ids] for k, ids in parts.items()}
    (out/"split.json").write_text(json.dumps({"full": membership, "pilot": {k:[rows[i]["id"] for i in ids] for k,ids in chosen.items()}}, indent=2))
    config = dict(seed=seed, split_seed=42, components=list(components), limit=limit, device=device, input_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), python=platform.python_version(), packages={k:importlib.metadata.version(k) for k in ("torch","transformers","numpy","scikit-learn")})
    scores, durations, revisions = {}, {}, {}
    start = time.perf_counter()
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=15000)
    tfidf.fit([xtext[i] for i in a])
    xt = tfidf.transform(xtext)
    head = LogisticRegression(C=1, max_iter=500).fit(xt[a], y[a])
    scores["tfidf"] = np.log(np.clip(head.predict_proba(xt), 1e-7, 1))
    durations["tfidf"] = time.perf_counter()-start
    bge = None
    from .components.bge import BGE
    from .components.bert import BERT
    for name in dict.fromkeys(["bge", *components]):
        print(f"Running {name} on {len(dev)} development excerpts", flush=True)
        start = time.perf_counter()
        if name in ("bge", "bert", "modernbert"):
            model = BGE(device=device) if name == "bge" else BERT(device=device, **({"model_id":"answerdotai/ModernBERT-base"} if name == "modernbert" else {}))
            revisions[name] = model.revision
            features = model.encode(xtext)
            if name == "bge":
                bge = features
            head = LogisticRegression(C=1, max_iter=500).fit(features[a], y[a])
            scores[name] = np.log(np.clip(head.predict_proba(features), 1e-7, 1))
            np.save(out/f"{name}_features.npy", features)
        elif name in ("gliner", "gliner2"):
            from .components.gliner import GLiNER
            model = GLiNER("fastino/gliner2-base-v1" if name == "gliner2" else "fastino/gliner2.5-base-v1")
            scores[name] = model.score(xtext)
        else:
            raise ValueError("Pilot supports compact models. Run optional expert scores separately with possum score.")
        durations[name] = time.perf_counter()-start
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    for specification in expert_scores:
        name, filename = specification.split("=", 1)
        if name in scores:
            raise ValueError(f"Duplicate expert {name}")
        artifact = np.load(filename, allow_pickle=False)
        metadata = json.loads(Path(filename).with_suffix(".json").read_text())
        if metadata.get("kind") != "scores" or not metadata.get("frozen_zero_shot"):
            raise ValueError("External scores must be frozen zero-shot experts; trained experts need fold provenance")
        ids = artifact["ids"].tolist()
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate score IDs")
        lookup = dict(zip(ids, artifact["values"]))
        scores[name] = np.asarray([lookup[r["id"]] for r in dev])
        if scores[name].shape != (len(dev), 2):
            raise ValueError("Expected two scores per example, ordered left/right")
        revisions[name] = metadata
    neural = fit_neural(bge[a], y[a], seed)
    with torch.inference_mode():
        scores["bge_neural"] = neural(torch.tensor(bge, dtype=torch.float32)).numpy()
    names = list(scores)
    temperatures = {k:temperature(s[c], y[c]) for k,s in scores.items()}
    probs = np.stack([softmax(scores[k]/temperatures[k], axis=1)[:,1] for k in names], axis=1)
    np.savez_compressed(out/"scores.npz", probabilities=probs, labels=y, embeddings=bge, names=np.array(names))
    predicted = {k:probs[v,j] for j,k in enumerate(names)}
    predicted["equal_mixture"] = probs[v].mean(1)
    fitted = {}
    for fixed, name in ((True,"fixed_mixture"),(False,"neural_gate"),(False,"hybrid_gate")):
        model = fit_gate(bge[g], probs[g], y[g], fixed=fixed, seed=seed, self_branch=name=="hybrid_gate")
        predicted[name] = predict_gate(model,bge[v],probs[v])
        torch.save(model.state_dict(),out/f"{name}.pt")
        with torch.inference_mode():
            fitted[name] = model.weights(torch.tensor(bge[v], dtype=torch.float32)).mean(0).tolist()
    ablations = {}
    for j, name in enumerate(names):
        keep = [i for i in range(len(names)) if i != j]
        model = fit_gate(bge[g],probs[g][:,keep],y[g],seed=seed)
        p = predict_gate(model,bge[v],probs[v][:,keep])
        ablations[name] = metrics(y[v],p)
    measured = {k:metrics(y[v],p) for k,p in predicted.items()}
    best = min(names,key=lambda k:measured[k]["log_loss"])
    groups = np.array([dev[i]["group"] for i in v])
    intervals = {k:bootstrap_delta(y[v],predicted["neural_gate"],predicted[k],groups,seed) for k in (best,"fixed_mixture")}
    errors = np.stack([(predicted[k]>=.5).astype(int)-y[v] for k in names])
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(errors)
    report = dict(config=config, checkpoint_revisions=revisions, full_counts={k:len(ids) for k,ids in parts.items()}, pilot_counts={k:len(ids) for k,ids in local.items()}, validation_groups=len(set(groups)), metrics=measured, temperature=temperatures, runtime_seconds=durations, mean_gate_weights=dict(zip(names,fitted["neural_gate"])), hybrid_gate_weights=dict(zip(names+["self"],fitted["hybrid_gate"])), leave_one_out=ablations, paired_log_loss_delta=intervals, error_correlation=np.where(np.isfinite(corr),corr,0).tolist(), error_correlation_note="Undefined constant-error correlations stored as zero", source_metrics={s:{k:metrics(y[v][mask],p[mask]) for k,p in predicted.items()} for s in sorted(set(dev[i]["source"] for i in v)) if (mask:=np.array([dev[i]["source"]==s for i in v])).any()}, conclusion="Diagnostic pilot only. Final test untouched; no component-drop or MoE claim without diverse human-verified evaluation and repeated seeds.")
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({"output":str(out),"metrics":measured,"conclusion":report["conclusion"]},indent=2))
    return report
