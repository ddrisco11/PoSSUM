# Human-text and platform evaluation — 2026-10-16

**Finding: the synthetic-trained heads and mixture do not transfer reliably to these external sources. MoE benefit remains unestablished.**

## Design

The existing compact-pilot recipe was fixed before external inference. Expert heads were fit on 240 synthetic passages, temperature calibration used 80 different synthetic passages, and mixture fitting used a third set of 80. No IBC or platform labels were used for optimization, calibration or seed selection. All three mixture initialization seeds are reported.

The external set contains **97 human-labeled IBC sentences** (49 liberal, 48 conservative) and **300 party-platform passages** (150 Democratic, 150 Republican). The official IBC archive contains 50 examples per liberal, conservative and neutral class; neutral examples and three short binary examples were excluded. These are root sentence annotations, not author ideology or phrase-node labels.

Platform labels are party-derived proxies, not human passage-level ideological annotations. Historical party realignment and neutral statements limit their interpretation. Results are separated by source and label quality.

Both sources were entirely held out because book/author and platform-document IDs are unavailable. No exact or character-ngram near duplicates against the synthetic corpus were detected at the configured threshold. This does not rule out semantic overlap or pretrained-model exposure. The synthetic final-test partition remains unopened.

## Human-labeled IBC results

| Model | Macro-F1 | Balanced accuracy | Log loss |
|---|---:|---:|---:|
| tfidf | 0.5155 | 0.5383 | 1.1202 |
| bge | 0.4870 | 0.5393 | 0.9947 |
| bert | 0.5141 | 0.5491 | 1.5687 |
| bge_neural | 0.3302 | 0.4371 | 0.9370 |
| gliner | 0.5659 | 0.5676 | 0.7507 |
| equal_mixture | 0.4381 | 0.5193 | 0.7948 |
| fixed_mixture_seed42 | 0.4332 | 0.4983 | 0.8986 |
| expert_gate_seed42 | 0.5224 | 0.5595 | 1.4115 |
| hybrid_gate_seed42 | 0.5306 | 0.5699 | 1.3748 |
| hybrid_gate_seed43 | 0.5141 | 0.5491 | 1.4094 |
| hybrid_gate_seed44 | 0.5278 | 0.5593 | 1.4090 |

## Party-platform stress test (weak labels)

| Model | Macro-F1 | Balanced accuracy | Log loss |
|---|---:|---:|---:|
| tfidf | 0.4910 | 0.5000 | 0.9703 |
| bge | 0.5104 | 0.5200 | 0.8410 |
| bert | 0.4831 | 0.4833 | 1.5613 |
| bge_neural | 0.4924 | 0.5200 | 0.8398 |
| gliner | 0.6018 | 0.6100 | 0.7125 |
| equal_mixture | 0.5385 | 0.5467 | 0.7440 |
| fixed_mixture_seed42 | 0.5060 | 0.5100 | 0.8503 |
| expert_gate_seed42 | 0.4986 | 0.5000 | 1.3160 |
| hybrid_gate_seed42 | 0.4950 | 0.4967 | 1.3347 |
| hybrid_gate_seed43 | 0.4986 | 0.5000 | 1.3170 |
| hybrid_gate_seed44 | 0.4986 | 0.5000 | 1.3286 |

## Interpretation

- BERT fell from 0.987 macro-F1 on the synthetic validation topic to 0.514 on the human-labeled sample. The two sets differ in source and annotation, so the comparison is diagnostic of transfer rather than a controlled estimate of a causal effect.
- GLiNER was the strongest individual component by macro-F1 on both external sources despite its weaker synthetic result. Keep it in subsequent human-data comparisons.
- The hybrid gate still assigned about 82–83% of its average seed-42 routing weight to BERT. Synthetic-only gate fitting did not learn to favor the better-transferring expert.
- Every evaluated model had external log loss worse than the constant 0.5 predictor (0.6931). Calibration learned on synthetic text did not transfer well. High F1 on synthetic text did not imply reliable external probabilities.
- No permanent component drop or mixture promotion is justified. The next training comparison needs human-labeled fit/calibration groups and a fresh human-labeled evaluation set.

## Limits and reproducibility

The IBC sample is small and its selection may not be representative. Original document/author identities are unavailable, so row independence is unknown; no significance claim or row-wise bootstrap confidence interval is made. The platform window sample is bounded and is not a representative survey of all platforms. Multiple gate seeds reuse the same data and are not independent dataset replications.

BGE and BERT ran on MPS; GLiNER ran on CPU. Measured component loading and inference times were approximately bge: 4.3s, bert: 6.1s, gliner: 41.7s. No IBC/platform input exceeded the compact encoders' 256-token limit. Eleven local tests passed, including safe IBC decoding and duplicate-overlap screening.

The [machine-readable report](external-ibc-platform.json) includes all metrics, confusion matrices, routing weights, temperatures, package versions and model revisions. The [data manifest](external-data-manifest.json) records download hashes, sampling and class counts. The [evaluation protocol](../docs/external-evaluation.md) describes reproduction. Raw text and per-example outputs are excluded from Git.

Once observed, these external results are development evidence; future model selection needs fresh held-out human text. Full IBC access requires contacting the maintainer; the full corpus was not obtained.

Sources: [official IBC sample](https://people.cs.umass.edu/~miyyer/ibc/) and [Party Platform Topics](https://huggingface.co/datasets/mlburnham/dem_rep_party_platform_topics). IBC attribution: Sim et al. (2013), *Measuring Ideological Proportions in Political Speeches*; Iyyer et al. (2014), *Political Ideology Detection Using Recursive Neural Networks*.
