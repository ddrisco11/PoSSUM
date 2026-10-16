# Source-held-out evaluation protocol

Freeze the compact synthetic pilot recipe before evaluating new sources. Use its existing 240 expert-fit, 80 calibration and 80 gate-fit records, temperatures, encoder pooling, 256-token cap and head hyperparameters. Train mixture initialization seeds 42, 43 and 44 without selecting among them on external labels. Include the standalone experts, equal average, fixed mixture, experts-only gate and hybrid gate.

## Evaluation sources

- **Human-labeled IBC:** the official public 150-sentence sample contains 50 liberal, 50 conservative and 50 neutral examples. Exclude neutral roots and retain whole sentences of 20–150 whitespace-separated tokens. Use root sentence membership, not phrase annotations, as ground truth. Liberal maps to 0; conservative to 1.
- **Party platforms:** sample a bounded set of passages across the published dataset. Derive weak labels from Democratic/Republican provenance, never from topic-entailment columns. Retain at most 150 eligible unique examples per class. These labels are a separate stress test and are not passage-level human ideology annotations.

Hold both sources entirely outside fitting and calibration. Original document/author IDs are unavailable; do not create a within-source train/test split or invent independent document groups. Check exact and lexical near-duplicate overlap against the entire synthetic source corpus before inference. Report each source separately, including class counts, macro-F1, balanced accuracy, log loss, Brier score, calibration error and confusion matrices. Do not combine weak-label and human-label performance into a headline score. Do not claim significance from row-wise bootstrap intervals when document independence is unknown.

This is an external diagnostic evaluation, not the final 80/20 human-text benchmark. The original reserved synthetic test remains unopened. Once reported, these external samples are development evidence; a final assessment requires fresh human-labeled data.

## Reproduce

```bash
python scripts/fetch_external_eval.py
HF_HOME=cache/huggingface python scripts/evaluate_external.py
```

The evaluator consumes the compact pilot caches and the original GLiNER development scores. Raw source text and per-example predictions remain in ignored `data/` and `runs/` directories. The restricted IBC reader accepts only inert records for the two tree classes; it never imports or executes upstream Python code.

Sources: [IBC official sample and annotation description](https://people.cs.umass.edu/~miyyer/ibc/), [Party Platform Topics](https://huggingface.co/datasets/mlburnham/dem_rep_party_platform_topics).

IBC references: Sim, Acree, Gross and Smith (2013), *Measuring Ideological Proportions in Political Speeches*; Iyyer et al. (2014), *Political Ideology Detection Using Recursive Neural Networks*.
