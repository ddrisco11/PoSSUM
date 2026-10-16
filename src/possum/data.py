"""Strict records, transitive leakage groups, and deterministic held-out splits."""
import hashlib
import json
import re
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupShuffleSplit


def normalize(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def prepare(records, similarity=0.93):
    accepted, seen, conflicts = [], {}, set()
    for r in records:
        if r.get("label") not in (0, 1) or r.get("excluded", False):
            continue
        if not 20 <= len(r["text"].split()) <= 150:
            continue
        if not r.get("source") or not r.get("group_ids"):
            raise ValueError("Every row needs source and globally namespaced document/author/group IDs")
        if r.get("label_kind") not in {"human", "synthetic", "weak"}:
            raise ValueError("label_kind must identify human, synthetic, or weak supervision")
        key = hashlib.sha256(normalize(r["text"]).encode()).hexdigest()
        if key in seen:
            prior = accepted[seen[key]]
            if prior["label"] != r["label"]:
                conflicts.add(key)
            prior["group_ids"] = sorted(set(prior["group_ids"] + r["group_ids"]))
            continue
        seen[key] = len(accepted)
        accepted.append(dict(r, id=key))
    accepted = [r for r in accepted if r["id"] not in conflicts]
    if not accepted:
        raise ValueError("No eligible records")
    parent = list(range(len(accepted)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(i, j):
        parent[root(j)] = root(i)
    owners = {}
    for i, r in enumerate(accepted):
        for g in r["group_ids"]:
            if g in owners:
                join(i, owners[g])
            owners[g] = i
    # Conservative lexical near-duplicate clustering, not semantic paraphrase detection.
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=40000)
    x = vec.fit_transform([normalize(r["text"]) for r in accepted])
    neighbors = NearestNeighbors(metric="cosine", radius=1-similarity).fit(x)
    for i, js in enumerate(neighbors.radius_neighbors(x, return_distance=False)):
        for j in js:
            join(i, int(j))
    for i, r in enumerate(accepted):
        r["group"] = accepted[root(i)]["id"]
    return accepted


def split_rows(rows, seed=42):
    groups = np.array([r["group"] for r in rows])
    if len(set(groups)) < 7:
        raise ValueError("At least seven independent groups required for test + four development partitions")
    all_idx = np.arange(len(rows))
    def cut(indices, fraction, salt):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed+salt).split(indices, groups=groups[indices]))
        return indices[a], indices[b]
    dev, test = cut(all_idx, .2, 0)
    rest, validation = cut(dev, .15, 1)
    rest, gate = cut(rest, .2, 2)
    fit, calibration = cut(rest, .25, 3)
    parts = dict(fit=fit, calibration=calibration, gate=gate, validation=validation, test=test)
    for name, ids in parts.items():
        if set(rows[i]["label"] for i in ids) != {0, 1}:
            raise ValueError(f"{name} lacks both labels; add independent groups, do not split documents")
    return parts


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
