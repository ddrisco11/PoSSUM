# Historical PoSSUM evaluations and workflows

These records describe earlier datasets and model versions. Their scores are not directly comparable with the current human-facet development and frozen topic audit. Use the [current model card](../README.md) for the selected research release and inference command. Commands below reproduce historical pipelines; they do not describe the current frozen model.

**Data revision (October 17):** Manifesto policy-category labels are excluded from new development because isolated passages can lack a self-contained ideological stance. The derived `assembled-v2-no-manifesto` corpus contains 13,416 development passages: 7,598 fit, 1,967 calibration, 2,195 router-fit and 1,656 validation. No model has yet been trained on this revision. Historical results below retain their original Manifesto-containing inputs and are not evidence that Manifesto caused the observed transfer regression. [Exclusion verification](../reports/manifesto-exclusion.json) · [Dataset review guide](dataset-review.md).

**Curated dataset v1.0:** A new local unified CSV contains **3,250 reviewed passages**: 1,106 MITweet examples mapped from human ideology facets, 2,113 synthetic examples and 31 previously seen IBC diagnostic sentences. Manifesto and party-only proxy labels are excluded. All records retain source provenance, original labels, review reasons and permitted-use roles. MITweet still needs a group-aware split; IBC must never enter training. This collection is 65% synthetic and its MITweet subset is heavily abortion-focused, so it is not a broad human gold benchmark. No model has been trained on this assembled version. [Dataset documentation](possum-dataset-v1.md) · [Validation manifest](../reports/possum-dataset-v1.json).

## Historical adapted-model evaluation (October 17)

The expanded run completed one pass over **7,652 expert-fit examples**, followed by calibration on 1,967, router fitting on 2,195 and validation on **1,899 examples across 21 groups**. Validation comprises 1,256 party-labeled tweets, 400 synthetic statements and 243 policy-mapped Manifesto passages. All labels are weak or synthetic. Locked test data were not used for fitting or model selection. Subsequent source auditing exposed 13 reserved synthetic passages; they remain excluded and are not an unseen benchmark.

| Configuration | Validation macro-F1 ↑ | Log loss ↓ |
|---|---:|---:|
| BERT LoRA | 0.816 | 0.396 |
| GLiNER, frozen | 0.669 | 0.628 |
| Mistral LoRA | 0.837 | 0.381 |
| Qwen, frozen | 0.779 | 0.445 |
| Four-expert equal average | 0.851 | 0.385 |
| Learned fixed mixture | 0.852 | 0.353 |
| Neural router | 0.840 | 0.331 |
| **Router + capped self contribution** | **0.839** | **0.329** |
| Equal average without GLiNER, exploratory ablation | 0.857 | 0.336 |
| BERT/GLiNER equal average | 0.819 | 0.461 |

The predeclared selection metric for this historical run was validation log loss, so `router_self_43` was its local export. It assigns average weights of 80.6% to Mistral, 9.9% to BERT, 4.6% to self prediction, 3.0% to Qwen and 1.9% to GLiNER. Self weight is capped at 25%; its classifier is fitted on expert-fit data and frozen during routing. Three router seeds give nearly identical results on the same split, not independent replications.

The paired whole-group bootstrap interval for selected-router minus equal-average log loss is **−0.148 to +0.039**, crossing zero. This is a descriptive development interval without selection correction. The small aggregate self-branch gain is also uncertain. These results support further testing, not a statistically established routing advantage or a decision to drop an expert.

The admitted split has a substantial coverage gap: Manifesto contributes only **54 fit examples in one group**, no calibration or router-fit examples, and 243 validation examples in one group. Tweets supply 83.6% of adapter-fit data. This is a plausible contributor to weak policy-text transfer, not a causal finding. See the [domain coverage audit](../reports/adapter-domain-coverage.json) and [development findings](development-findings.md).

Source differences matter: selected-router macro-F1 is **0.837 on tweets**, **0.995 on synthetic statements**, and **0.561 on Manifesto**. Equal averaging scores 0.625 on Manifesto and has lower Manifesto log loss. The latest scores cannot be treated as direct improvements over the earlier pilot because the data and partitions changed. The matched frozen comparison is complete: BERT improves from **0.790 to 0.816 macro-F1**, and Mistral from **0.796 to 0.837** on the same validation split. Four-expert equal averaging improves from **0.836 / 0.415 log loss** with frozen BERT/Mistral to **0.851 / 0.385** with adapters. Gains are concentrated in tweets; the adapted ensemble is slightly worse on Manifesto. BERT compares a frozen mean-pooled logistic probe against its adapted sequence classifier, so that comparison includes the head recipe. [Matched comparison](../reports/frozen-adapter-comparison.json). On the previously inspected 97-example human-labeled IBC sample, transfer remains weak: the selected router scores **0.580 macro-F1 / 0.739 log loss**, versus **0.603 / 0.627** for equal averaging. Frozen Qwen has the highest individual macro-F1, 0.649. The selected router is worse than a constant 0.5 prediction on log loss (0.693), reinforcing that this export is not production-ready. These 97 examples are a diagnostic, not a fresh final test.

The matched IBC comparison exposes a regression: frozen Mistral scores **0.711 macro-F1 / 0.719 log loss**, versus **0.585 / 0.816** after adaptation. The full frozen equal average scores **0.751 / 0.581**, versus **0.603 / 0.627** for the adapted equal average. BERT improves from 0.423 to 0.495 macro-F1 but remains weak. This small, previously inspected diagnostic shows a domain tradeoff, not a basis for declaring either recipe generally superior. [Matched IBC comparison](../reports/frozen-adapter-ibc-comparison.json).

A post-hoc subset audit finds that **BERT + Mistral equal averaging** gives 0.848 validation macro-F1 and 0.330 log loss, close to the full router with about 57% less measured scoring compute. Its IBC macro-F1 falls to 0.557, while **GLiNER + Qwen** reaches 0.701 on that small diagnostic. These contrasting results are exploratory and do not justify a universal subset choice. Timing excludes model loading and swapping. [Subset audit](../reports/adapter-expert-subsets.json).

[Full adapted evaluation](../reports/adapter-development-evaluation.json) · [Group uncertainty](../reports/adapter-validation-uncertainty.json) · [Human-text diagnostic](../reports/adapter-ibc-diagnostic.json) · [Adapter training record](../reports/adapter-training.json)

## Earlier frozen four-expert evaluation

The U.S. Manifesto pilot sampled 1,200 candidate passages; filtering and deduplication retained 1,171. Document groups stay separate across probe fitting, temperature calibration, routing-head training, validation and the reserved test partition.

| Partition | Passages |
|---|---:|
| Probe fitting | 317 |
| Temperature calibration | 116 |
| Routing-head training | 205 |
| Validation | 159 |
| Reserved test, untouched | 374 |

The validation set contains only **two document groups**. Labels map human-annotated policy categories through RILE rules to left/right; they are **weak ideology labels**, not direct human judgments of passage-level slant. Holding out 20% of document groups produces a 31.9% test share by passage count because documents vary in size. The target for broader evaluation remains approximately 80/20 without splitting documents across partitions.

| Model | Validation macro-F1 ↑ | Log loss ↓ |
|---|---:|---:|
| BERT + fitted probe | 0.431 | 0.579 |
| GLiNER 2.5 | 0.627 | 0.648 |
| Mistral 7B | 0.629 | 0.705 |
| Qwen3.5 9B | 0.618 | 0.601 |
| **Equal averaging** | **0.700** | **0.573** |
| Learned fixed mixture | 0.626 | 0.618 |
| Neural router, experts only | 0.665–0.682 | 0.601–0.606 |
| Neural router + self prediction | 0.522–0.584 | 2.549–2.931 |

Ranges cover initialization seeds 42–44 on the **same split and cached expert scores**. They are not confidence intervals or independent dataset repetitions. Equal averaging leads on both macro-F1 and log loss in this comparison. The self-prediction branch takes 65.9–99.8% of average mixture weight and makes confident errors, consistent with overfitting on the small routing-training partition. Low learned weight is not sufficient evidence to remove an expert.

Mistral and Qwen use exact teacher-forced scores for ` left` and ` right` after a prompt ending in `This statement reflects a`. Scores are normalized and temperature-calibrated on the calibration partition; they do not come from sampled completions. Calibration temperatures were approximately 1.343 for Mistral and 2.200 for Qwen.

On an M2 Mac with 16 GB unified memory, sequential 4-bit MLX scoring of the 797 development passages took **9.3 minutes for Mistral** and **15.5 minutes for Qwen**. Routing fits and ablations took **2.8 seconds**. These measurements exclude model downloads and are specific to this workload and machine.

[Complete four-expert results](../reports/frozen-four-pilot.json) · [Training protocol](evaluation.md) · [Model registry](models.md)

## Data coverage

Multi-source adapter training, scoring and development evaluation are complete. The table below describes source availability before conservative admission; final assembled counts follow below.

| Source | Available eligible candidates | Label / access status |
|---|---:|---|
| IBC public sample | 97 | Human-labeled binary examples; already used for external diagnostics. Full corpus unavailable locally. |
| U.S. Manifesto | 4,289 | Human policy codes mapped to weak binary labels across 12 documents; only a subset used in the latest pilot. |
| Party Platform Topics | 8,535 | Party-derived labels; original document provenance needs recovery or conservative whole-source placement. |
| GPT-4 Political Ideologies | 3,200 | Synthetic; retained as a separately identifiable source. |
| Political Tweets | 114,183 | Party-derived candidates across 481 author identities; 11,850 pass a conservative politicalness/stance heuristic before assembly. |
| GPT-4 Political Bias | 0 under the current length rule | Non-neutral examples fall below the 20-word minimum. |
| Qbias | Pending | Raw source acquisition remains outstanding. |
| SCCC | Pending text access | Public metadata and partisan scores do not supply the full message text. |

Candidate counts are not final training sizes. See the [corpus status](corpus-status.md) and [dataset manifest](datasets.md) for provenance and preprocessing details.

## Earlier transfer diagnostics

An earlier synthetic-trained pilot achieved 0.987 BERT macro-F1 on one synthetic validation topic, but transferred poorly to external text. On 97 human-labeled IBC sentences, BERT scored 0.514, GLiNER 0.566 and the hybrid gate 0.514–0.531 macro-F1. On 300 party-labeled platform passages, the corresponding scores were 0.483, 0.602 and 0.495–0.499. All evaluated models had external log loss worse than a constant 0.5 predictor.

These results belong to the earlier synthetic-trained models, **not the latest four-expert Manifesto configuration**. They demonstrate why synthetic validation alone is insufficient. The public IBC sample lacks original document IDs and is now a previously inspected diagnostic set, not a fresh final benchmark. The separate synthetic test partition was not used for these reported scores; its group-constrained split is 75/25. Later source auditing exposed 13 reserved passages, which remain excluded from the new dataset and cannot be treated as unseen.

[External evaluation report](../reports/external-evaluation.md) · [Initial synthetic pilot](../reports/initial-pilot.md) · [External evaluation protocol](external-evaluation.md)

## Expanded data preparation and local adaptation

A conservative multi-source assembly is now available locally: **13,713 development passages**, partitioned into 7,652 expert-fit, 1,967 calibration, 2,195 router-fit and 1,899 validation examples. They consist of 11,016 heuristic-screened party-labeled tweets, 2,400 synthetic statements and 297 policy-mapped Manifesto passages. This is a weak/synthetic research dataset, not new human ground truth. Original test identities remain protected; conflicts, missing provenance and unverified candidates are quarantined. The larger model-training run started on October 17, 2026. Both rank-8 adapters completed the full fit partition, while GLiNER and Qwen remain frozen. Measured training compute was 137.2 minutes for Mistral and 2.9 minutes for BERT, excluding loading and checkpoint writes. Mistral peaked at 4.20 GiB of MLX memory. See the [adapter training record](../reports/adapter-training.json). Development validation is reported above; it does not establish performance on new human-authored text.

Short M2 backward-pass probes confirmed narrow rank-8 QLoRA compatibility for Mistral and Qwen. Their updates were discarded; no quality claim follows. See the [ten-hour training options report](overnight-training-options.md), [assembled corpus report](../reports/assembled-corpus.json), and [independent provenance audit](dataset-admission-audit.md) for counts, measured speed/memory, limitations and overnight choices.

## Earlier pipeline setup

After preparing `data/manifesto/candidates.jsonl` as described in the [corpus documentation](corpus-status.md), launch the frozen four-expert pipeline from the repository root:

```bash
mkdir -p runs/frozen-four-42
python -u scripts/run_frozen_four.py >> runs/frozen-four-42/pipeline.log 2>&1
```

The pipeline first produces compact-model diagnostics, then scores Mistral and Qwen. Its **final** report, `runs/frozen-four-42/four-expert-report.json`, compares exactly BERT, GLiNER, Mistral and Qwen, with an optional self-prediction branch. Intermediate compact-model results use a different set of components.

Use the same command to resume only after the previous process has stopped. Expert scoring checkpoints save every ten passages using atomic replacement and disk synchronization. Completed expert caches are reused before loading weights. Interrupted routing fits restart; their cached expert scores remain available. Checkpoints live in ignored `runs/`, not on GitHub.

## Adapter training and inference

The [local adaptation protocol](local-adaptation.md) describes immutable partitions, recoverable checkpoints and export selection. Derive the new corpus once from the historical assembly, then use a new run directory:

```bash
python scripts/exclude_manifesto.py
python -u scripts/run_option_a.py --dataset data/assembled-v2-no-manifesto --run runs/option-a-no-manifesto --hours 8
```

The pipeline trains Mistral and BERT sequentially, caches the five components, then calibrates and compares nine mixture fits plus equal averaging and ablations. The self-prediction branch is fitted only on the expert-fit partition and frozen during router training, with its mixture weight capped at 25%. A fixed absolute cutoff can be supplied with `--deadline` as a timezone-qualified ISO timestamp. Resuming preserves the original cutoff.

Once evaluation completes, use the selected local export:

```bash
python scripts/predict_option_a.py --export runs/option-a-20261017/export --text "A political passage of approximately twenty to one hundred and fifty words goes here, expressing the viewpoint to be classified."
```

An inference-only bundle can be assembled locally:

```bash
python scripts/package_adapter_export.py --run runs/option-a-20261017
python scripts/predict_option_a.py --export runs/option-a-20261017/bundle/export --text "A political passage of approximately twenty to one hundred and fifty words goes here, expressing the viewpoint to be classified."
```

The approximately 3 MB bundle includes adapter weights, calibration, saved heads, aggregate validation and a SHA-256 manifest. It omits optimizer checkpoints, base weights and raw data. It requires the source checkout, dependencies and separately cached pinned base models. Both the relocated adapter bundle and the frozen reference passed raw-text prediction equivalence checks; both adapter optimizer-recovery checks passed. [Verification record](../reports/export-verification.json).

The matched frozen reference is independently reloadable after running the comparison:

```bash
python scripts/predict_option_a.py --export runs/option-a-20261017/frozen-export --text "A political passage of approximately twenty to one hundred and fifty words goes here, expressing the viewpoint to be classified."
```

Adapter weights and score caches remain local. The exported configuration requires its sibling adapter directories and the pinned base models. The end-to-end smoke test verified checkpoint recovery, all four expert scoring paths, router export reloads and fresh-text inference; it is a functionality test, not an accuracy benchmark.

## Live training monitor

```bash
.venv/bin/python scripts/monitor_training.py
```

The read-only Tkinter dashboard refreshes every two seconds with component progress, scoring speed, current-stage ETA and the live log. Use `--run /path/to/run` for another run directory. Closing the window leaves training running. Python must include Tkinter. Stage ETA excludes downloads and queued models; download file counts do not indicate byte-level progress.

