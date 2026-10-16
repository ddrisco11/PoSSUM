# Corpus ingestion status — 2026-10-16

Counts below are observed downloads, not benchmark estimates. Candidate counts follow label normalization, 20–150-word filtering and exact deduplication within each source. They are not final training counts: weak-source politicalness/ambiguity filtering and cross-source near-duplicate grouping remain outstanding.

| Source | Observed raw rows | Eligible candidates | Status |
|---|---:|---:|---|
| IBC official sample | 150 | 97 binary examples | Human-labeled external diagnostic only. Full corpus requires maintainer access. |
| Manifesto U.S. | 20,144 quasi-sentences | 4,289 | Downloaded 12 English annotated documents. Human policy codes mapped through RILE; these are weak binary labels. |
| Party Platform Topics | 20,172 | 8,535 | Downloaded. Labels derive from party provenance. Missing document IDs require whole-source placement or provenance recovery before partitioning. |
| GPT-4 Political Ideologies | 3,200 | 3,200 | Ingested. Existing split reserves 800; 2,400 development rows support the larger frozen-encoder baseline. |
| Political Tweets | 190,491 | 114,183 | Downloaded, 481 author identities. Party-derived labels; nonpolitical content remains to be filtered. |
| GPT-4 Political Bias | 657 | 0 at the 20-word minimum | Downloaded. 201 neutral rows removed; all 456 remaining rows are shorter than 20 words. Can support a separately reported short-text setting. |
| Qbias | — | — | Linked benchmark repository contains a notebook expecting `raw/qbias.csv`, but not that raw file. Source acquisition remains outstanding. |
| SCCC | 7M+ metadata records advertised | — | Public release omits message text; rehydration or separate newsletter access is required. |

## Training admission

Weak labels are eligible for supervised training as a separately marked tier. They must not be silently promoted to human annotations. Tweets support author-separated partitions. The platform dataset must not be randomly row-split when original platform identities are unavailable; whole-source assignment is an alternative. Include source-balanced comparisons to prevent the tweet corpus from overwhelming smaller datasets.

The larger currently prepared baseline is **synthetic-only** and must not be described as the full multi-source model. The complete multi-source run has not started. Corpus acquisition and admission are distinct from model training.

```bash
pip install -e '.[data]'
python scripts/ingest_available_sources.py
```

Downloads and candidate JSONL files stay in ignored `data/available/`. Aggregate counts and download hashes are in `reports/available-source-ingestion.json`.

Sources: [IBC](https://people.cs.umass.edu/~miyyer/ibc/), [Manifesto access requirements](https://manifesto.wzb.eu/information/documents/corpus), [platforms](https://huggingface.co/datasets/mlburnham/dem_rep_party_platform_topics), [tweets](https://huggingface.co/datasets/Jacobvs/PoliticalTweets), [supplementary synthetic data](https://huggingface.co/datasets/cajcodes/political-bias), [Qbias notebook](https://github.com/matous-volf/political-leaning-prediction/blob/main/datasets/political_leaning/notebooks/qbias.ipynb), [SCCC text availability](https://congressionalcomms.com/).

## Frozen four-expert pilot

`python scripts/run_frozen_four.py` samples 1,200 Manifesto candidates with seed 42, then deduplicates and groups by source document. The initial partition contains 317 probe-fit, 116 calibration, 205 router-training, 159 validation and 374 untouched test passages. The 20% group holdout yields 31.9% of rows because documents vary in size; it is not an exact 80/20 row split. No sentence from a held-out document enters fitting or calibration.

Mistral and Qwen run sequentially in separate processes using pinned 4-bit MLX snapshots. Candidate probabilities come from exact teacher-forced left/right scores, calibrated only on the calibration partition. Each ten-passage chunk is cached for resumption and records elapsed scoring time. BERT and GLiNER backbones stay frozen. The final comparison includes exactly four experts plus the router's optional self-prediction branch, across three initialization seeds, with individual-expert and leave-one-out results.
