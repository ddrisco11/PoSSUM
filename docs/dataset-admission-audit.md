# Dataset admission and leakage audit

Audit date: 2026-10-16. Counts below were recomputed from the local downloaded records. Dataset availability, eligibility for training, and evidence of generalization are separate quantities.

## Available supervision

The dataset manifest names **eight sources**, not ten to twelve. Five have nonempty local text collections meeting the 20–150-word binary-label filter. Two of those remain external or quarantined rather than automatically entering training.

| Source | Locally eligible rows before joint admission | Supervision | Admission boundary |
|---|---:|---|---|
| Manifesto U.S. | 4,289 occurrences; **3,827 normalized unique texts** | Human policy categories mapped to weak left/right labels | Preserve document and duplicate-connected test closure. |
| Political Tweets | 114,183 source-deduplicated candidates | Author-party proxy | Retain author groups; remove nonpolitical/ambiguous candidates conservatively; report filter limitations. |
| GPT-4 Political Ideologies | 3,200 | Synthetic | Preserve issue-family grouping and the existing 800-row test holdout. |
| Party Platform Topics | 8,535 source-deduplicated candidates | Party proxy | Quarantine pending document recovery and resolution of overlap with Manifesto and diagnostic records. |
| IBC public sample | 97 | Human passage ideology | External diagnostic only; previously inspected, not a pristine final test. |
| GPT-4 Political Bias | 0 at 20-word minimum | Synthetic | All 456 nonneutral rows are too short under the present contract; 201 neutral rows excluded. |
| Qbias | No raw text collection acquired | Source-level proxy, pending review | Supplied notebook references an absent raw CSV. |
| SCCC | No message text collection acquired | Model-derived partisan scores | Metadata without text cannot train a text classifier. |

The sum **130,304** is a candidate inventory (including IBC and platforms), not a training-set size. It uses Manifesto occurrences and therefore includes known duplicates. Source counts do not establish a politicalness filter, label correctness, or distribution rights.

The first reproducible tweet admission screen retained **11,850 weak-label candidates across 175 authors** (7,495 left / 4,355 right). It quarantined 102,333 candidates, including 31,234 visibly truncated texts. The rule requires explicit policy vocabulary and a stance cue; its precision and recall have not been human measured. An independent normalized-text comparison found no exact overlaps between these retained tweets and the Manifesto, synthetic, or existing external collections. Cross-source near-duplicate analysis and author-separated assignment are still required. See `reports/tweet-admission.json` for filter counts and limitations.

## Protected boundaries

The original frozen-four split holds out 374 passages. Their document identities seed a transitive closure over the complete Manifesto collection. Recomputing the original `possum.data.prepare` closure, which `scripts/assemble_corpus.py` retains internally, yields:

- 456 normalized exact duplicate buckets spanning documents, of which 455 are also verbatim equal.
- 21 lexical near-duplicate edges spanning documents, using character 3–5-gram TF-IDF cosine similarity at least 0.93.
- **2,417 unique passages excluded from development** because their document/duplicate component touches the original test set.
- **1,410 remaining passages in only three connected components**. More rows do not create more independent validation units.

The initial protected documents are `61320_199211`, `61320_202411`, `61620_201611`, and `61620_202011`. The closure additionally connects Republican documents from 2004, 2008, and 2012. Keep all these documents outside fitting, calibration, routing, and development evaluation. Preserve the original final-test membership; the rest of its closure is a protective exclusion buffer, not extra test examples selected after results were seen.

The synthetic holdout contains entire issue families **5 and 7**, totaling 800 passages. Their grouping is a proxy because original generation batches are unavailable. A changed corpus must not reassign these issue families to training.

Previously viewed validation and IBC results are development diagnostics. They cannot become newly unseen final-test evidence just by changing a filename or split seed.

## Cross-source conflicts

Normalized-text comparisons found **491 exact shared texts between Manifesto and Party Platform Topics**. Labels disagree for **158** of them. This is expected from different targets: policy-category ideology and party provenance are not interchangeable labels. No arbitrary source precedence should silently resolve these examples.

The existing 300-passage platform diagnostic has **27 exact overlaps with Manifesto**, including **eight label disagreements**. It is consequently not a clean source-independent evaluation of a model trained on all Manifesto data. All 300 are also present in the full platform candidate collection. Both platform collections currently use `platform:entire_source`, because original document/year identities are absent.

Retain a quarantined platform tier and quarantine contradictory duplicate examples. If preserving the entire platform source as a protected diagnostic causes document-connected Manifesto examples to be excluded, report that additional loss explicitly. Alternatively, retire the platform diagnostic's unseen status before admitting related texts; this does not provide a new independent evaluation set. The original locked final-test boundaries must remain protected under either choice.

## Assembly verification

The assembled release should record hashes of source inputs, filter version, original supervision, source and author/document groups, exact and near-duplicate conflicts, protected exclusions, and actual split counts. Every accepted row must have a stable normalized-content ID, binary label, 20–150-word text, nonempty source, and original `label_kind`.

Check unique content IDs and transitive provenance components for disjointness across fit, calibration, router fit, validation, and test. Build near-duplicate connections before partition assignment. Conflicting-label near matches should be quarantined explicitly: the original pilot `prepare` function groups these matches but does not reject their labels. A scalable replacement must document its recall limitations; an approximate duplicate index does not prove the absence of paraphrase leakage.

Keep synthetic and weak rows distinguishable after duplicate merging. Use source-balanced training and per-source metrics so the tweet corpus does not dominate the objective. A keyword filter can establish a reproducible candidate screen, but it cannot establish human-verified politicalness or remove every neutral and ambiguous passage.

No local source currently supplies a large, diverse, never-inspected human-labeled final evaluation set. Expanded training can proceed with correctly marked weak and synthetic supervision; deployment claims still require that missing evaluation evidence.

## Assembled v1 verification

`data/assembled-v1` contains **129,351 unique candidate records** after joint exact deduplication. Independent structural verification passed: content IDs, partition files and manifests agree; no protected ID or author/document group enters development; development components are disjoint across partitions; exact label conflicts and prior diagnostic texts are excluded; supervision metadata is retained.

There are **13,713 development passages**: 11,016 screened weak tweets, 2,400 synthetic statements, and 297 policy-mapped Manifesto passages. The development files contain 7,652 expert-fitting, 1,967 calibration, 2,195 router-fitting, and 1,899 validation examples. These are distinct roles: **only `fit.jsonl` is admitted for expert adapter training**. Do not concatenate or randomly re-split the development files.

The Manifesto reduction beyond the earlier 1,410-row audit is chiefly caused by an expanded component joining previously separate development partitions. Its 1,084 rows are quarantined to preserve those earlier boundaries; remaining conflicting or diagnostic-related records are also excluded. The retained Manifesto data spans only two components. The assembled corpus is therefore dominated by weak tweets and synthetic statements, not diverse human-annotated ideology.

The file `protected_test.jsonl` contains **3,983 protection-eligible rows**, not 3,983 original final-test examples: 2,351 Manifesto rows from test-connected components, 832 newly reserved tweets, and 800 synthetic holdout statements. The original locked set is **1,174 IDs** (374 Manifesto plus 800 synthetic); 1,159 remain in that file and 15 are quarantined for conflicts or diagnostic proximity. All 1,174 remain protected. The additional Manifesto closure is a buffer and must not silently enlarge the original test. Separate `original-test-eligible.jsonl`, `original-test-quarantined.jsonl`, `new-weak-holdout.jsonl`, and `test-closure-buffer.jsonl` files distinguish these roles. Evaluate original locked IDs separately from any newly declared tweet test, and disclose quarantined exclusions.

The complete protected closure contains **21,198 IDs**, including 17,215 quarantined rows. The quarantine file overall contains 111,558 rows; IBC's 97 examples remain a separate external diagnostic. No test predictions or test-label performance summaries were produced by this audit.

Reproduce structural verification with:

```bash
.venv/bin/python scripts/check_assembled_corpus.py data/assembled-v1 --output reports/assembled-corpus-check.json
```

This verifies the recorded connections; it does not independently establish recall of the approximate duplicate index or verify weak labels. The assembly retains the original Manifesto character-TF-IDF closure and additionally uses MinHash candidates checked by word-trigram Jaccard. Those different similarity measures are not interchangeable guarantees.

## Reproduce assembly

From the repository root, after downloading the source candidates and generating the tweet admission screen, choose a versioned output directory:

```bash
.venv/bin/python scripts/assemble_corpus.py --output data/assembled-v1 --tweet-admission data/tweet-admission-v1/eligible.jsonl
.venv/bin/python scripts/check_assembled_corpus.py data/assembled-v1 --output reports/assembled-corpus-check.json
```

Assembly writes aggregate audit counts, partition files and a protected manifest. Neither command scores the locked final test. Source text stays in ignored local data files.
