# PoSSUMDataset v1.0

A local, provenance-preserving collection of **3,250 short political passages**. Labels are `0 = Left` and `1 = Right`. Every exported passage contains 20–150 whitespace-delimited words. No model has been trained or evaluated on this assembled version.

| Sub-dataset | Left | Right | Total | Label origin and permitted use |
|---|---:|---:|---:|---|
| MITweet | 965 | 141 | 1,106 | Binary mapping of selected human ideology facets; new development data requiring a group-aware split |
| GPT-4 Political Ideologies | 1,115 | 998 | 2,113 | Synthetic labels; retain existing fit/calibration/gate/validation roles |
| IBC public sample | 15 | 16 | 31 | Direct human sentence labels; previously exposed, selected diagnostic only |
| **Total** | **2,095** | **1,155** | **3,250** | **Mixed provenance, not an all-human gold dataset** |

## Curation

All 4,638 eligible candidate occurrences received a passage-by-passage automated semantic review: 2,141 MITweet, 2,400 nonprotected synthetic statements and 97 IBC sentences. Review excludes neutral or generic claims, missing context, unendorsed reporting, foreign-only passages, misleading label directions and incomplete text. Original labels were not corrected or replaced. This screening is an additional model judgment, **not human reannotation**, and can make mistakes.

MITweet labels use only Economic Orientation (I3), Economic Equality (I4), Ethical Pursuit (I5), and Church–State Relations (I6). At least one must be related; all related selected facets must agree on a noncentral direction. Upstream `0` maps to Left and `2` to Right. The other eight original facets remain in each row's provenance but do not determine the binary mapping. Conflicting annotations on repeated source text are excluded before semantic selection. US-event topic selection is followed by passage review; it does not establish author nationality.

Text receives whitespace normalization only. No invented context, paraphrasing or sentence completion is added. Exact and lexical near duplicates are collapsed with all retained occurrence provenance preserved. Near-duplicate detection uses character TF-IDF cosine at least 0.93 or word-trigram Jaccard at least 0.85, with transitive duplicate clusters. This does not certify removal of every semantic paraphrase.

The assembly checks 21,198 protected historical IDs and their groups, plus previously used development and diagnostic passages. Protection propagates through duplicate and declared group connectivity before filtering. Historical inputs remain byte-identical. Source reconstruction independently verifies every exported passage and all 3,258 provenance occurrences.

## Use the provenance and role columns

The local file is `data/PoSSUMDatasetv1.0/PoSSUMDatasetv1.0.csv`, UTF-8 with a BOM and standard quoted CSV fields. Its 28 columns include:

- `text`, `label`, `label_name`, `word_count` and stable normalized-content `record_id`.
- `sub_dataset`, `quality_tier`, `label_provenance`, `human_annotated` and `synthetic`.
- Original label JSON, exact mapping rule, source URL, source revision or explicit missing-revision status, source file hash, source record ID and upstream split.
- Topic, US relevance basis, declared groups and the limitations of their identifiers.
- `partition`, `allowed_use`, `previously_used`, semantic review method and reason.
- Duplicate cluster and complete source-occurrence provenance JSON.

MITweet identifiers count logical CSV records including the header, rather than physical lines: quoted tweets can contain line breaks. Synthetic hashes identify the preserved local normalized source snapshot because its original upstream revision was not recorded. IBC hashes identify the public sample archive, with its root annotations independently verified.

**Do not train on all CSV rows indiscriminately.** The 31 IBC rows are diagnostic only. Synthetic assignments remain 956 fit, 400 calibration, 359 router-fit and 398 validation. The 1,106 MITweet rows need a group-aware development split before fitting. Author and thread IDs are absent, so topic groups are a conservative proxy, not proof of author-disjoint partitions. No new unseen final test set is certified. Thirteen historical reserved synthetic passages were inspected during source auditing; they remain excluded and cannot be called unseen. Historical model fitting and scoring were not rerun.

## Coverage limits and exclusions

The collection is 65.0% synthetic. MITweet is 87.3% Left and 79.7% abortion-related. Counts are left intact rather than inflated through oversampling. This is a cleaner development resource, but it does not establish broad US political coverage or reliable human-text generalization. A selected IBC diagnostic cannot provide an unbiased final performance estimate.

Manifesto policy mappings, author-party Political Tweets labels and party-platform proxy labels are excluded from this version. Their text is not necessarily unusable, but their labels do not reliably establish the viewpoint expressed by each isolated passage. In the available GPT-4 Political Bias snapshot, every nonneutral row is below the 20-word cutoff; all remaining length-eligible rows are neutral. Full IBC is unavailable. Qbias and SCCC have no locally admitted passage-level source with usable text and complete provenance.

Source texts and per-record review ledgers remain local, outside Git. Public availability is not treated as permission to redistribute upstream datasets. The repository contains only assembly/validation code and the [aggregate manifest](../reports/possum-dataset-v1.json).

## Rebuild from preserved local inputs

```bash
.venv/bin/python scripts/prepare_possum_dataset_v1.py
node scripts/export_possum_dataset_v1.mjs
.venv/bin/python scripts/validate_possum_dataset_v1.py
```

The exporter requires `@oai/artifact-tool`; an installed module entry can be supplied through `ARTIFACT_TOOL_ENTRY`. It constructs a workbook, verifies every cell against the assembly payload and serializes the verified values to CSV. The validator round-trips the CSV, reconstructs source text and original labels, checks protected IDs/groups and historical assignments, and verifies duplicate removal. Missing or incomplete semantic-review ledgers fail closed.

Sources: [MITweet](https://github.com/LST1836/MITweet), [GPT-4 Political Ideologies](https://huggingface.co/datasets/JyotiNayak/political_ideologies), [IBC](https://people.cs.umass.edu/~miyyer/ibc/).
