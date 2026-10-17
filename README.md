<img src="assets/possum-model-card.png" alt="Cartoon red and blue possum wearing an Uncle Sam hat beside the PoSSUM title" width="100%">

# PoSSUM

**Political Slant Scoring Using Mixtures**

PoSSUM studies the political viewpoint expressed in short English passages about U.S. politics. It combines calibrated predictions from multiple models and compares equal averaging with learned mixture weights. Neural routing uses 384-dimensional BGE-small embeddings; an optional self-prediction branch lets the routing head contribute its own answer.

**Current status:** BERT and Mistral LoRA adapters have completed one full pass over 7,652 fit examples from the assembled multi-source development corpus. Expert scoring and validation are in progress; new quality results are pending. The frozen four-expert pilot is complete. Equal averaging is the strongest evaluated four-expert configuration. Learned routing has not demonstrated an advantage, and the self-prediction branch shows signs of overfitting. There is no validated production release.

| Property | Specification |
|---|---|
| Task | Binary political slant classification |
| Labels | `0 = Left`, `1 = Right` |
| Input | Approximately 20–150 words; English, U.S. politics |
| Evaluated experts | BERT + fitted classification probe, GLiNER 2.5 base, Mistral 7B v0.3, Qwen3.5 9B |
| Routing input | BGE-small, 384-dimensional embeddings |
| Best current baseline | Equal average of the four experts' temperature-calibrated probabilities |
| Training status | Previous pilot: frozen backbones. Current run: BERT and Mistral LoRA; frozen GLiNER and Qwen |
| Deployment status | Development code and aggregate evaluation reports; no validated release checkpoint |

Every enabled expert runs on every passage. This is a dense prediction ensemble, not sparse routing that avoids expert computation. Component weights and source datasets are downloaded separately.

## Latest evaluation

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

Multi-source adapter training is complete; scoring and evaluation are in progress. The table below describes source availability before conservative admission; final assembled counts follow below.

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

These results belong to the earlier synthetic-trained models, **not the latest four-expert Manifesto configuration**. They demonstrate why synthetic validation alone is insufficient. The public IBC sample lacks original document IDs and is now a previously inspected diagnostic set, not a fresh final benchmark. The separate synthetic test partition remains untouched; its group-constrained split is 75/25.

[External evaluation report](reports/external-evaluation.md) · [Initial synthetic pilot](reports/initial-pilot.md) · [External evaluation protocol](docs/external-evaluation.md)

## Expanded data preparation and local adaptation

A conservative multi-source assembly is now available locally: **13,713 development passages**, partitioned into 7,652 expert-fit, 1,967 calibration, 2,195 router-fit and 1,899 validation examples. They consist of 11,016 heuristic-screened party-labeled tweets, 2,400 synthetic statements and 297 policy-mapped Manifesto passages. This is a weak/synthetic research dataset, not new human ground truth. Original test identities remain protected; conflicts, missing provenance and unverified candidates are quarantined. The larger model-training run started on October 17, 2026. Both rank-8 adapters completed the full fit partition, while GLiNER and Qwen remain frozen. Measured training compute was 137.2 minutes for Mistral and 2.9 minutes for BERT, excluding loading and checkpoint writes. Mistral peaked at 4.20 GiB of MLX memory. See the [adapter training record](reports/adapter-training.json). New quality measurements are pending.

Short M2 backward-pass probes confirmed narrow rank-8 QLoRA compatibility for Mistral and Qwen. Their updates were discarded; no quality claim follows. See the [ten-hour training options report](docs/overnight-training-options.md), [assembled corpus report](reports/assembled-corpus.json), and [independent provenance audit](docs/dataset-admission-audit.md) for counts, measured speed/memory, limitations and overnight choices.

## Next development milestones

The next release candidate will be evaluated against the equal-average baseline. Planned work includes:

- Expand Manifesto development coverage while preserving held-out documents and existing test boundaries.
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

The [local adaptation protocol](docs/local-adaptation.md) describes immutable partitions, recoverable checkpoints and export selection. After assembling `data/assembled-v1`, run:

```bash
python -u scripts/run_option_a.py --run runs/option-a-20261017 --hours 8
```

The pipeline trains Mistral and BERT sequentially, caches the five components, then calibrates and compares nine mixture fits plus equal averaging and ablations. The self-prediction branch is fitted only on the expert-fit partition and frozen during router training, with its mixture weight capped at 25%. A fixed absolute cutoff can be supplied with `--deadline` as a timezone-qualified ISO timestamp. Resuming preserves the original cutoff.

Once evaluation completes, use the selected local export:

```bash
python scripts/predict_option_a.py --export runs/option-a-20261017/export --text "A political passage of approximately twenty to one hundred and fifty words goes here, expressing the viewpoint to be classified."
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
