# Dataset manifest and admission rules

Dataset size estimates must be checked after filtering and deduplication. A normalized record is JSON with `text`, `label` (0 left / 1 right), `source`, `label_kind` (`human`, `weak`, `synthetic`), and nonempty `group_ids` (globally namespaced original document and author identities). Optional `excluded: true` removes neutral/ambiguous/nonpolitical rows. Never map unknown labels silently.

| Dataset | Supervision / action | Source |
|---|---|---|
| IBC | Human sentence/subsentence ideology. The official 150-sentence sample is used as a whole-source external diagnostic: 97 binary-labeled sentences meet the length filter. Neutral roots are excluded; phrase nodes are not independent examples. The importer uses inert tree records. Full-corpus access and original book/author IDs remain pending. | [UMass](https://people.cs.umass.edu/~miyyer/ibc/) |
| Manifesto Project | Policy-category RILE mapping is a proxy, not direct passage ideology. Restrict to U.S. English records; document the left/right category mapping and version. Obtain original document IDs and permitted corpus access. The model card is not the dataset itself. | [Project](https://manifesto-project.wzb.eu/), [RoBERTa-RILE](https://huggingface.co/niksmer/RoBERTa-RILE) |
| Party Platform Topics | Weak labels from `dataset` party provenance, NOT `entailment` or `validation_label` (topic annotations). Exposed schema lacks platform-year/document IDs; exclude from supervised fitting until recovered. A separate 300-passage, source-held-out stress test uses party-derived labels. Historical party positions need review. Repeated premises must be deduplicated. | [HF](https://huggingface.co/datasets/mlburnham/dem_rep_party_platform_topics) |
| GPT-4 Political Ideologies | Synthetic. Upstream conservative=0/liberal=1, so explicitly invert. Includes 8 issue categories; group by issue for the integration pilot. Generation batch IDs unavailable, so this is a documented proxy. Author reports manual checks, not independent unseen human-authored evaluation. | [HF](https://huggingface.co/datasets/JyotiNayak/political_ideologies) |
| Political Tweets | Weak party/author supervision. Require author IDs, remove retweets/duplicates and nonpolitical text after review, enforce length. Do not treat party as text-level ground truth. | [HF](https://huggingface.co/datasets/Jacobvs/PoliticalTweets) |
| GPT-4 Political Bias | Optional synthetic; inspect exact five-way labels, explicitly map extremes and drop neutral/ambiguous cases. | [HF](https://huggingface.co/datasets/cajcodes/political-bias) |
| Qbias | Optional news source labels; verify passage-level viewpoint and document grouping before admitting. | [GitHub](https://github.com/matous-volf/political-leaning-prediction/tree/main/datasets/political_leaning/notebooks) |
| SCCC | Optional expansion, model-derived partisan scores; obtain full text separately, retain author/message IDs and teacher provenance. | [Project](https://congressionalcomms.com/) |

Benchmark: [Political Leaning and Politicalness Classification of Texts (2025)](https://arxiv.org/html/2507.13913).

`prepare` applies length filtering, normalized exact deduplication, removes conflicting exact-label examples, joins source/author and lexical near-duplicate groups transitively before any split. Similarity uses character n-gram cosine >=0.93; semantic paraphrases can remain and require further review. This implementation is sized for the pilot; large-corpus deduplication needs a scalable index.

Supervised ingestion beyond the synthetic pilot remains pending provenance recovery and label-map checks. The IBC sample and bounded platform sample are available for external evaluation only; see [source-held-out evaluation](external-evaluation.md). No raw source texts are republished. All normalized corpora and per-example outputs are Git-ignored. Check upstream access and redistribution terms for each source before expanded use.

```bash
python scripts/fetch_synthetic_pilot.py
possum pilot data/synthetic.jsonl --components bge bert --limit 800 --device mps --output runs/synthetic-42
```

Eight issue groups cannot yield an exact 80/20 row split: two held-out groups produce a 75/25 split, with actual proportions reported. This synthetic-only pilot can verify software and expose gross signal failures; it cannot establish generalization or justify a final expert drop.
