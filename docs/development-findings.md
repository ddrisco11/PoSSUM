# Adapted-model development findings

The completed run establishes that narrow BERT and Mistral adapters, calibrated component scores and a reloadable routing export can be trained locally on a 16 GB M2. It does not establish a reliable general political-slant classifier.

## Coverage limits

| Partition | Tweets | Synthetic | Manifesto | Left-label share |
|---|---:|---:|---:|---:|
| Adapter fit | 6,398 | 1,200 | 54 | 64.4% |
| Calibration | 1,567 | 400 | 0 | 71.5% |
| Router fit | 1,795 | 400 | 0 | 42.4% |
| Validation | 1,256 | 400 | 243 | 40.2% |

The 54 fit Manifesto passages occupy one group; the 243 validation passages occupy one different group. Synthetic validation occupies one group, while the tweet validation occupies 19 groups. These source/group concentrations help explain the wide group-bootstrap intervals. They also limit conclusions drawn from aggregate passage counts. The fit objective uses class balancing, but calibration and router fitting still encounter different class and source mixtures.

The conservative assembly preserved historical split roles and excluded leakage conflicts. Its resulting Manifesto coverage is much smaller than the downloaded candidate pool. Source availability is not the same as source representation in each admitted partition. Existing run inputs and protected test identities remain fixed.

## What the results support

The selected full router improves development log loss over equal averaging, but loses macro-F1. Its average weight is approximately 81% Mistral, and its group-bootstrap advantage is uncertain. The self contribution is small and no longer dominates, but its incremental benefit is also uncertain. An equal-average ablation without GLiNER has stronger aggregate macro-F1, while GLiNER is relatively more useful on Manifesto. A single global ranking is insufficient for dropping an expert.

Human-text transfer remains weak on the previously inspected public IBC sample: selected-router log loss is 0.739, compared with 0.627 for equal averaging and 0.693 for a constant half-probability prediction. IBC contains only 97 labeled sentences, lacks original book/author grouping and has already informed prior development. It cannot serve as a fresh release benchmark.

## Next development priorities

1. Construct a separately versioned development split with adequate source and class coverage in fit, calibration and router-fit partitions, preserving the locked test closure. Require a coverage audit before spending compute. Reassigning eligible development groups would invalidate direct comparisons with earlier validation scores and must be recorded as a new protocol.
2. Add passage-level human labels across political topics and domains, including ambiguous and nonpolitical passages. Keep a newly collected, author/document-disjoint evaluation set untouched by tuning.
3. Compare the adapted recipes against frozen baselines on the same current split before attributing gains to LoRA. Report domain-level effects and uncertainty alongside aggregate scores.
4. Evaluate cheaper expert subsets using development-only fitting. Preserve the original export and describe any later subset search as exploratory. Validate the chosen cost/quality tradeoff on new human data.
5. Validate abstention and an explicit neutrality/politicalness policy before deployment. A left/right confidence threshold alone has not been shown to handle these cases.

[Development evaluation](../reports/adapter-development-evaluation.json) · [Group uncertainty](../reports/adapter-validation-uncertainty.json) · [IBC diagnostic](../reports/adapter-ibc-diagnostic.json) · [Coverage audit](../reports/adapter-domain-coverage.json)
