<img src="assets/possum-model-card.png" alt="Cartoon red and blue possum wearing an Uncle Sam hat beside the PoSSUM title" width="100%">

# PoSSUM

**Political Slant Scoring Using Mixtures**

PoSSUM studies the political viewpoint expressed in short English passages about U.S. politics. It combines calibrated predictions from multiple models and compares equal averaging with learned mixture weights. Neural routing uses 384-dimensional BGE-small embeddings; an optional self-prediction branch lets the routing head contribute its own answer.

**Current status:** The selected research model is a calibrated fixed mixture of human-adapted Mistral (88.6%) and frozen, chat-formatted Qwen (11.4%). It scored **0.890 macro-F1 / 0.872 balanced accuracy / 0.241 log loss** on a previously sealed 159-passage topic stress test. Mistral alone has higher F1 and substantially lower inference cost, while the mixture has lower log loss on this sample. Neural routing was weaker during development. All three evaluated models have verified local inference exports; broader validation and neutrality handling remain incomplete.

## Frozen topic stress evaluation

All models, calibration, mixture weights and decision rules were fixed before opening **159 MITweet passages**: 111 left and 48 right, across Energy Crisis, Mass Shootings and Political Parties. No test result was used to select or retune the candidates.

| Frozen candidate | Macro-F1 ↑ | Balanced accuracy ↑ | Log loss ↓ | Left recall | Right recall |
|---|---:|---:|---:|---:|---:|
| Development-selected Mistral–Qwen mixture | 0.890 | 0.872 | 0.241 | 0.973 | 0.771 |
| Mistral adapter alone | 0.907 | 0.893 | 0.264 | 0.973 | 0.813 |
| DeBERTa compact alternative | 0.766 | 0.744 | 0.488 | 0.946 | 0.542 |

The mixture's descriptive 95% row-bootstrap intervals are **0.826–0.939 for macro-F1** and **0.150–0.357 for log loss**. They do not capture dependence between passages, author effects or label uncertainty. This is a narrow topic stress test using mapped human ideology facets, not a broad U.S. political-text benchmark; related policy themes can occur in synthetic training data. Mistral alone makes 12 classification errors versus 14 for the mixture, with no established universal advantage for either model.

Topic performance varies. The mixture's macro-F1 is 0.834 on Energy Crisis (61 passages), 0.714 on Mass Shootings (14), and 0.934 on Political Parties (84). DeBERTa's Mass Shootings F1 is 0.417, illustrating the transfer gap despite its strong development score. At decision thresholds fixed earlier on gate data, the mixture and single adapter both reach 0.906 balanced accuracy; the table retains the default 0.5 threshold. [All candidates, topics, uncertainty, confusion matrices and fixed-prior sensitivity](reports/human-topic-stress.json).

## Development comparison

**Human-text development (October 18, completed):** The revised corpus contains **2,199 MITweet passages mapped from human ideology facets** and **2,113 synthetic passages**. Broader admission recovers meaningful short statements and additional topics while keeping ambiguous and conflicting labels out. The human validation subset is strongly left-skewed (**366 left / 42 right**), and **83.8%** of its passages belong to Abortion or Women’s Right. An earlier mixture scored 0.898 macro-F1 / 0.131 log loss on the remaining 66 passages, but that small, already-used subset is not broad validation. [Topic coverage audit](reports/human-topic-coverage.json). These are exploratory development results, with checkpoint and architecture selection on the same validation set.

| Configuration | Human validation macro-F1 ↑ | Log loss ↓ |
|---|---:|---:|
| Lexical baseline | 0.727 | 0.210 |
| BERT full fine-tune, mixed data, seed 42 | 0.776 | 0.200 |
| BERT full fine-tune, human data only | 0.732 | 0.221 |
| DeBERTa full fine-tune, mixed data | 0.892 | 0.144 |
| DeBERTa full fine-tune, human data only | 0.786 | 0.192 |
| BERT distilled from Mistral, alpha 0.25 | 0.802 | 0.206 |
| BERT distilled from Mistral, alpha 0.5 | 0.758 | 0.209 |
| Frozen Mistral | 0.852 | 0.145 |
| Human-only Mistral LoRA, epoch 1 | 0.865 | 0.122 |
| Human-only Mistral LoRA, epoch 2 | 0.891 | 0.103 |
| Frozen Qwen, raw-completion prompt | 0.604 | 0.265 |
| Frozen Qwen, nonthinking chat template | 0.764 | 0.191 |
| **Best verified fixed mixture** | **0.885** | **0.0985** |

Both human-only Mistral adapter epochs are complete. Epoch two improves human validation loss from 0.145 to 0.103 and balanced accuracy from 0.863 to 0.903. The selected two-model mixture lowers loss further to 0.0985, but has slightly lower macro-F1 than the single adapter; its paired loss difference is −0.0043 with a descriptive 95% interval of [−0.0201, +0.0054]. This does not establish a useful ensemble advantage. Synthetic macro-F1 improves from 0.977 to 0.987, while synthetic log loss worsens from 0.062 to 0.072 for the adapter and 0.099 for the mixture. The selected export reproduces cached calibration predictions within 1.2e−16. [Completed matched comparison, both epochs and uncertainty](reports/human-mistral-adapter-completed.json).

Both distillation settings finished four epochs and selected epoch 3. Neither improves BERT's primary metric, calibrated log loss; the alpha-0.25 F1 increase does not change that selection rule. Paired loss intervals include zero. Before the chat-template comparison, the mixture refit with distilled candidates reached 0.126 log loss and did not replace the earlier 0.124 mixture. [Distillation results, including all epochs](reports/human-distillation-development.json).

Both ModernBERT trials finished behind BERT (human-only log loss 0.228; mixed-data 0.230). BERT seed repetitions give human log loss from 0.200 to 0.229; equal averaging gives 0.197, with an uncertain improvement over seed 42. Qwen's nonthinking chat-template comparison is complete: macro-F1 improves from 0.604 to 0.764 and log loss from 0.265 to 0.191. Adding it to the mixture changes loss from 0.12385 to 0.12288 with unchanged F1; the paired loss interval spans zero, and the added inference cost still needs measurement. [Chat-template and mixture comparison](reports/human-qwen-chat-development.json). Human-only Mistral LoRA training and evaluation are complete. Both DeBERTa-v3-base recipes completed four epochs. Mixed-source training selects epoch 3 (0.892 macro-F1 / 0.144 log loss); human-only selects epoch 4 (0.786 / 0.192). This improves the single BERT reference, while the adapted Mistral and selected mixture still have lower probability loss. Its initial half-precision optimizer update failed; training restarted from pretrained weights in float32. [Completed DeBERTa comparison, all eight epochs](reports/human-deberta-completed.json). Adding DeBERTa to the candidate pool leaves the selected mixture effectively unchanged. [Seed sensitivity](reports/human-seed-sensitivity.json) · [Completed compact comparison](reports/human-compact-completed.json) · [Qwen raw-prompt results](reports/human-qwen-development.json) · [Mistral and mixture comparison](reports/human-mistral-development.json).

These figures use different inputs from the historical evaluations below and cannot be compared directly. [Protocol](docs/human-text-development.md) · [Aggregate data audit](reports/human-development-data.json) · [Initial baselines](reports/human-development-baselines.json).

**Data revision (October 17):** Manifesto policy-category labels are excluded from new development because isolated passages can lack a self-contained ideological stance. The derived `assembled-v2-no-manifesto` corpus contains 13,416 development passages: 7,598 fit, 1,967 calibration, 2,195 router-fit and 1,656 validation. No model has yet been trained on this revision. Historical results below retain their original Manifesto-containing inputs and are not evidence that Manifesto caused the observed transfer regression. [Exclusion verification](reports/manifesto-exclusion.json) · [Dataset review guide](docs/dataset-review.md).

**Curated dataset v1.0:** A new local unified CSV contains **3,250 reviewed passages**: 1,106 MITweet examples mapped from human ideology facets, 2,113 synthetic examples and 31 previously seen IBC diagnostic sentences. Manifesto and party-only proxy labels are excluded. All records retain source provenance, original labels, review reasons and permitted-use roles. MITweet still needs a group-aware split; IBC must never enter training. This collection is 65% synthetic and its MITweet subset is heavily abortion-focused, so it is not a broad human gold benchmark. No model has been trained on this assembled version. [Dataset documentation](docs/possum-dataset-v1.md) · [Validation manifest](reports/possum-dataset-v1.json).

| Property | Specification |
|---|---|
| Task | Binary political slant classification |
| Labels | `0 = Left`, `1 = Right` |
| Input | Short English passages about U.S. politics; current development admission uses 12–150 words |
| Evaluated components | Lexical and embedding baselines; BERT, ModernBERT and DeBERTa fine-tunes; BERT distillation; Mistral LoRA; frozen GLiNER and Qwen |
| Routing input | BGE-small, 384-dimensional embeddings |
| Current development selection | Two-model fixed mixture, selected by human validation log loss |
| Training status | Planned compact, frozen-expert, distillation and human-only Mistral adapter comparisons complete |
| Deployment status | Verified local exports and completed 159-passage topic audit; broader human evaluation and abstention remain incomplete |

Every enabled expert runs on every passage. This is a dense prediction ensemble, not sparse routing that avoids expert computation. Component weights and source datasets are downloaded separately.

## Verified research release

[PoSSUM-2026-10-18-research](reports/human-model-release.json) records the three frozen variants, exact manifest hashes, evaluation results, decision rules and deployment checks. The selected mixture remains the primary development-selected model; the single adapter and compact encoder are documented alternatives.

DeBERTa's local bundle is approximately **712 MiB**, including its encoder weights. The mixture and single-Mistral bundles are approximately **1.6 MiB each**, containing adapters/calibration rather than base weights: they require two and one separately cached quantized base models, respectively. These bundle sizes are not total download or memory requirements. Weights remain local.

## Run the current local models

With the local exports and cached base models available, run from the repository directory:

```bash
HF_HOME=cache/huggingface HF_HUB_OFFLINE=1 .venv/bin/python -m possum.predict_development --run runs/improvement-20261018 --variant selected --text "The government should expand affordable healthcare coverage through public funding and ensure that access to essential treatment does not depend on income."
```

Use `--variant compact` for the selected DeBERTa encoder or `--variant single_expert` for the two-epoch Mistral adapter. The command resolves the verified version automatically and uses the immutable candidate record after final evaluation is frozen. Exports and base weights are local artifacts, not included in the Git checkout. All three exports also passed relocation into a different directory and inference in fresh Python processes, with maximum probability difference below 1.2e−8 on three calibration passages. This uses the existing runtime and cached base weights; clean-machine installation is untested. [Portability checks](reports/human-export-portability.json).

Measured local inference on 12 calibration passages:

| Export | Median single request, seconds | Batch seconds per passage |
|---|---:|---:|
| Selected Mistral–Qwen mixture | 6.38 | 2.33 |
| DeBERTa compact alternative | 1.34 | 0.138 |
| Mistral adapter alone | 1.36 | 0.820 |

Each request includes model loading, integrity checks and cleanup; these are not persistent warm-server measurements. The batch contains the same 12 passages, and all three exports passed single/batch prediction parity. The mixture's small validation-loss improvement over Mistral alone comes with substantially higher measured cost. [Timing protocol and complete measurements](reports/human-inference-latency.json).

The output includes left/right probabilities, a default binary label, the gate-set decision threshold and a separate equal-prior score. These are probabilities under a binary task, not a measure of ideological intensity. Neutral, ambiguous or nonpolitical input can still receive a confident binary answer; a validated neutrality/abstention model remains outstanding. Inputs over the recorded token limit are rejected rather than truncated. The Python batch API expects a list or tuple of strings; empty passages and invalid model probabilities produce explicit errors. Encoder resources are released on input-length errors.

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

The admitted split has a substantial coverage gap: Manifesto contributes only **54 fit examples in one group**, no calibration or router-fit examples, and 243 validation examples in one group. Tweets supply 83.6% of adapter-fit data. This is a plausible contributor to weak policy-text transfer, not a causal finding. See the [domain coverage audit](reports/adapter-domain-coverage.json) and [development findings](docs/development-findings.md).

Source differences matter: selected-router macro-F1 is **0.837 on tweets**, **0.995 on synthetic statements**, and **0.561 on Manifesto**. Equal averaging scores 0.625 on Manifesto and has lower Manifesto log loss. The latest scores cannot be treated as direct improvements over the earlier pilot because the data and partitions changed. The matched frozen comparison is complete: BERT improves from **0.790 to 0.816 macro-F1**, and Mistral from **0.796 to 0.837** on the same validation split. Four-expert equal averaging improves from **0.836 / 0.415 log loss** with frozen BERT/Mistral to **0.851 / 0.385** with adapters. Gains are concentrated in tweets; the adapted ensemble is slightly worse on Manifesto. BERT compares a frozen mean-pooled logistic probe against its adapted sequence classifier, so that comparison includes the head recipe. [Matched comparison](reports/frozen-adapter-comparison.json). On the previously inspected 97-example human-labeled IBC sample, transfer remains weak: the selected router scores **0.580 macro-F1 / 0.739 log loss**, versus **0.603 / 0.627** for equal averaging. Frozen Qwen has the highest individual macro-F1, 0.649. The selected router is worse than a constant 0.5 prediction on log loss (0.693), reinforcing that this export is not production-ready. These 97 examples are a diagnostic, not a fresh final test.

The matched IBC comparison exposes a regression: frozen Mistral scores **0.711 macro-F1 / 0.719 log loss**, versus **0.585 / 0.816** after adaptation. The full frozen equal average scores **0.751 / 0.581**, versus **0.603 / 0.627** for the adapted equal average. BERT improves from 0.423 to 0.495 macro-F1 but remains weak. This small, previously inspected diagnostic shows a domain tradeoff, not a basis for declaring either recipe generally superior. [Matched IBC comparison](reports/frozen-adapter-ibc-comparison.json).

A post-hoc subset audit finds that **BERT + Mistral equal averaging** gives 0.848 validation macro-F1 and 0.330 log loss, close to the full router with about 57% less measured scoring compute. Its IBC macro-F1 falls to 0.557, while **GLiNER + Qwen** reaches 0.701 on that small diagnostic. These contrasting results are exploratory and do not justify a universal subset choice. Timing excludes model loading and swapping. [Subset audit](reports/adapter-expert-subsets.json).

[Full adapted evaluation](reports/adapter-development-evaluation.json) · [Group uncertainty](reports/adapter-validation-uncertainty.json) · [Human-text diagnostic](reports/adapter-ibc-diagnostic.json) · [Adapter training record](reports/adapter-training.json)

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

[Complete four-expert results](reports/frozen-four-pilot.json) · [Training protocol](docs/evaluation.md) · [Model registry](docs/models.md)

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

Candidate counts are not final training sizes. See the [corpus status](docs/corpus-status.md) and [dataset manifest](docs/datasets.md) for provenance and preprocessing details.

## Earlier transfer diagnostics

An earlier synthetic-trained pilot achieved 0.987 BERT macro-F1 on one synthetic validation topic, but transferred poorly to external text. On 97 human-labeled IBC sentences, BERT scored 0.514, GLiNER 0.566 and the hybrid gate 0.514–0.531 macro-F1. On 300 party-labeled platform passages, the corresponding scores were 0.483, 0.602 and 0.495–0.499. All evaluated models had external log loss worse than a constant 0.5 predictor.

These results belong to the earlier synthetic-trained models, **not the latest four-expert Manifesto configuration**. They demonstrate why synthetic validation alone is insufficient. The public IBC sample lacks original document IDs and is now a previously inspected diagnostic set, not a fresh final benchmark. The separate synthetic test partition was not used for these reported scores; its group-constrained split is 75/25. Later source auditing exposed 13 reserved passages, which remain excluded from the new dataset and cannot be treated as unseen.

[External evaluation report](reports/external-evaluation.md) · [Initial synthetic pilot](reports/initial-pilot.md) · [External evaluation protocol](docs/external-evaluation.md)

## Expanded data preparation and local adaptation

A conservative multi-source assembly is now available locally: **13,713 development passages**, partitioned into 7,652 expert-fit, 1,967 calibration, 2,195 router-fit and 1,899 validation examples. They consist of 11,016 heuristic-screened party-labeled tweets, 2,400 synthetic statements and 297 policy-mapped Manifesto passages. This is a weak/synthetic research dataset, not new human ground truth. Original test identities remain protected; conflicts, missing provenance and unverified candidates are quarantined. The larger model-training run started on October 17, 2026. Both rank-8 adapters completed the full fit partition, while GLiNER and Qwen remain frozen. Measured training compute was 137.2 minutes for Mistral and 2.9 minutes for BERT, excluding loading and checkpoint writes. Mistral peaked at 4.20 GiB of MLX memory. See the [adapter training record](reports/adapter-training.json). Development validation is reported above; it does not establish performance on new human-authored text.

Short M2 backward-pass probes confirmed narrow rank-8 QLoRA compatibility for Mistral and Qwen. Their updates were discarded; no quality claim follows. See the [ten-hour training options report](docs/overnight-training-options.md), [assembled corpus report](reports/assembled-corpus.json), and [independent provenance audit](docs/dataset-admission-audit.md) for counts, measured speed/memory, limitations and overnight choices.

## Next development milestones

Further development will use the completed equal-average, router and frozen-baseline comparisons. Priorities include:

- Acquire and audit self-contained human-labeled viewpoint passages; exclude context-dependent Manifesto policy mappings from new development.
- Evaluate final ensemble calibration and an uncertainty-based abstention policy on separate data.
- Compare cheaper expert subsets to quantify the accuracy–latency tradeoff.
- Evaluate on fresh human-labeled political, neutral, ambiguous and nonpolitical passages.
- Compare the adapted experts and regularized routing against fixed and equal-weight baselines on the expanded development split.

These milestones are not established release capabilities. A confidence threshold alone does not detect neutrality or nonpolitical content.

## Intended use and limitations

PoSSUM is intended for research into expressed political viewpoint in text. It does not infer an individual's political beliefs. Party or publisher labels are noisy proxies for passage-level ideology. Left/right categories depend on political context, topic and time; the present scope is U.S. politics.

Representative human-text evaluation, robust abstention and broader domain testing remain incomplete. The current validation set is too small and concentrated to establish reliable real-world performance or statistically supported component rankings. OpenJev and ModernBERT adapters exist but were not evaluated in the four-expert run. Training data and component weights retain their upstream licenses; review the model registry and dataset manifest before redistribution or deployment.

## Local setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,gliner,mlx,data,adapters]'
pytest
possum --help
```

The MLX extra requires Apple Silicon. Other component workflows can omit it. Tested package versions are recorded in [requirements-tested.txt](requirements-tested.txt). Use one backbone at a time and cache predictions on memory-constrained machines. OpenJev is excluded from the current 16 GB workflow.

After preparing `data/manifesto/candidates.jsonl` as described in the [corpus documentation](docs/corpus-status.md), launch the frozen four-expert pipeline from the repository root:

```bash
mkdir -p runs/frozen-four-42
python -u scripts/run_frozen_four.py >> runs/frozen-four-42/pipeline.log 2>&1
```

The pipeline first produces compact-model diagnostics, then scores Mistral and Qwen. Its **final** report, `runs/frozen-four-42/four-expert-report.json`, compares exactly BERT, GLiNER, Mistral and Qwen, with an optional self-prediction branch. Intermediate compact-model results use a different set of components.

Use the same command to resume only after the previous process has stopped. Expert scoring checkpoints save every ten passages using atomic replacement and disk synchronization. Completed expert caches are reused before loading weights. Interrupted routing fits restart; their cached expert scores remain available. Checkpoints live in ignored `runs/`, not on GitHub.

## Adapter training and inference

The [local adaptation protocol](docs/local-adaptation.md) describes immutable partitions, recoverable checkpoints and export selection. Derive the new corpus once from the historical assembly, then use a new run directory:

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

The approximately 3 MB bundle includes adapter weights, calibration, saved heads, aggregate validation and a SHA-256 manifest. It omits optimizer checkpoints, base weights and raw data. It requires the source checkout, dependencies and separately cached pinned base models. Both the relocated adapter bundle and the frozen reference passed raw-text prediction equivalence checks; both adapter optimizer-recovery checks passed. [Verification record](reports/export-verification.json).

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

## Modular components and reproducibility

```bash
# Emit features or two-column scores without training a backbone.
possum score bge data/development.jsonl --output runs/bge-features.npz --device mps
possum score gliner data/development.jsonl --output runs/gliner.npz
possum score mlx data/development.jsonl --model-id /path/to/pinned-mlx-model --output runs/causal.npz
```

`--model-id` selects a checkpoint within a component module. Encoder outputs are features, not untrained left/right classification logits. Frozen zero-shot score caches can be integrated through `possum pilot --expert-scores name=file.npz`; learned external heads require fold provenance and are rejected by that interface.

Record checkpoint revisions, dataset hashes, split membership, dependency versions, seeds, prompts and device for every run. Selection uses development partitions only. Raw corpora, model caches, trained local checkpoints and sample-level outputs stay ignored; published reports contain aggregate metrics. An optional test workflow is provided in [docs/ci-example.yml](docs/ci-example.yml); automated CI is not enabled.
