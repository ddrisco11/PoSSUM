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

Both ModernBERT trials finished behind BERT (human-only log loss 0.228; mixed-data 0.230). BERT seed repetitions give human log loss from 0.200 to 0.229; equal averaging gives 0.197, with an uncertain improvement over seed 42. Qwen's nonthinking chat-template comparison is complete: macro-F1 improves from 0.604 to 0.764 and log loss from 0.265 to 0.191. Adding it to the mixture changes loss from 0.12385 to 0.12288 with unchanged F1; the paired loss interval spans zero. The final two-model configuration's measured cost is reported below. [Chat-template and mixture comparison](reports/human-qwen-chat-development.json). Human-only Mistral LoRA training and evaluation are complete. Both DeBERTa-v3-base recipes completed four epochs. Mixed-source training selects epoch 3 (0.892 macro-F1 / 0.144 log loss); human-only selects epoch 4 (0.786 / 0.192). This improves the single BERT reference, while the adapted Mistral and selected mixture still have lower probability loss. Its initial half-precision optimizer update failed; training restarted from pretrained weights in float32. [Completed DeBERTa comparison, all eight epochs](reports/human-deberta-completed.json). Adding DeBERTa to the candidate pool leaves the selected mixture effectively unchanged. [Seed sensitivity](reports/human-seed-sensitivity.json) · [Completed compact comparison](reports/human-compact-completed.json) · [Qwen raw-prompt results](reports/human-qwen-development.json) · [Mistral and mixture comparison](reports/human-mistral-development.json).

These figures use different inputs from the historical evaluations below and cannot be compared directly. [Protocol](docs/human-text-development.md) · [Aggregate data audit](reports/human-development-data.json) · [Initial baselines](reports/human-development-baselines.json).

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

## Remaining validation needs

- Collect broader, independently adjudicated U.S. viewpoint passages, with balanced class coverage and document/author metadata.
- Validate neutrality, ambiguity, nonpolitical inputs and abstention using separate labeled data.
- Confirm the mixture-versus-single-model tradeoff on a new benchmark and in a persistent inference service.
- Verify a clean installation and deployment packaging beyond the existing local runtime.

The frozen topic test is now permanently reserved evaluation data. Confidence alone does not establish neutrality. Earlier weak-label datasets, experiments and training commands remain available in the [historical record](docs/historical-results.md).

## Intended use and limitations

PoSSUM is intended for research into expressed political viewpoint in text. It does not infer an individual's political beliefs. Party or publisher labels are noisy proxies for passage-level ideology. Left/right categories depend on political context, topic and time; the present scope is U.S. politics.

Representative human-text evaluation, robust abstention and broader domain testing remain incomplete. The current validation set is too small and concentrated to establish reliable real-world performance or statistically supported component rankings. ModernBERT was evaluated in the current development comparison; OpenJev was not included in this local run. Training data and component weights retain their upstream licenses; review the model registry and dataset manifest before redistribution or deployment.

## Local setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,gliner,mlx,data,adapters]'
pytest
possum --help
```

The MLX extra requires Apple Silicon. Other component workflows can omit it. The actual current runtime is recorded in the [dependency snapshot](reports/human-development-runtime.json); this is not a verified fresh-install lockfile. Use one backbone at a time and cache predictions on memory-constrained machines. OpenJev is excluded from the current 16 GB workflow.

## Modular components and reproducibility

```bash
# Emit features or two-column scores without training a backbone.
possum score bge data/development.jsonl --output runs/bge-features.npz --device mps
possum score gliner data/development.jsonl --output runs/gliner.npz
possum score mlx data/development.jsonl --model-id /path/to/pinned-mlx-model --output runs/causal.npz
```

`--model-id` selects a checkpoint within a component module. Encoder outputs are features, not untrained left/right classification logits. Frozen zero-shot score caches can be integrated through `possum pilot --expert-scores name=file.npz`; learned external heads require fold provenance and are rejected by that interface.

Record checkpoint revisions, dataset hashes, split membership, dependency versions, seeds, prompts and device for every run. Selection uses development partitions only. Raw corpora, model caches, trained local checkpoints and sample-level outputs stay ignored; published reports contain aggregate metrics. An optional test workflow is provided in [docs/ci-example.yml](docs/ci-example.yml); automated CI is not enabled.
