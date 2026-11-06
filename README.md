# PoSSUM — Political Slant Scoring Using Mixtures

**An applied machine learning project exploring when calibrated model ensembles improve political viewpoint classification—and when a single model is the better tradeoff.**

PoSSUM classifies the viewpoint expressed in short English passages about U.S. politics as left or right. The project spans data curation, transformer fine-tuning, probability calibration, constrained mixture optimization, and reproducible local inference.

The central question: **does combining models produce better probability estimates than using the strongest model alone?** On a frozen 159-passage topic stress test, the selected mixture achieved **0.890 macro-F1** and **0.241 log loss**. Mistral alone achieved higher F1 and lower inference cost; the mixture had lower log loss. That tradeoff is the main result, rather than a claim that a larger ensemble always wins.

**Stack:** Python · PyTorch · Hugging Face Transformers · scikit-learn · SciPy · MLX · LoRA

[Results](#results) · [Method](#method) · [Run locally](#run-locally) · [Technical documentation](#technical-documentation)

<img src="assets/possum-model-card.png" alt="PoSSUM project illustration: a red and blue possum wearing an Uncle Sam hat" width="640">

## Project highlights

- **Data and evaluation design:** Curated 2,199 human-annotated MITweet passages and 2,113 synthetic passages, with separate fitting, calibration, mixture-fitting, validation, and topic-holdout partitions.
- **Model experimentation:** Compared lexical and embedding baselines, BERT/ModernBERT/DeBERTa fine-tuning, Mistral LoRA adaptation, frozen language models, and knowledge distillation.
- **Applied optimization:** Learned nonnegative mixture weights constrained to sum to one, optimizing log loss; compared fixed mixtures with equal averaging and embedding-based neural routing.
- **Research engineering:** Recorded dataset and model hashes, pinned model revisions, exported three inference variants, and checked prediction consistency after reload, relocation, and batch inference.
- **Practical model selection:** Evaluated classification quality, probability quality, uncertainty, and inference cost together.

## Why this problem?

Political viewpoint depends on what a passage says, its topic, and its context. Publisher or party labels are imperfect substitutes for passage-level annotations, and high classification accuracy does not guarantee reliable probabilities.

PoSSUM investigates these issues through a binary research task. It scores **expressed viewpoint in text**, not an individual's political beliefs. Its outputs are class probabilities, not ideological intensity scores or neutrality judgments.

## Method

```text
Curated passages → Separate fit / calibration / mixture-fit / validation splits
                              ↓
                 Baselines + fine-tuned / frozen models
                              ↓
                    Calibrated model probabilities
                              ↓
                 Fixed mixtures vs. neural routing
                              ↓
               Freeze candidates → Topic stress test
                              ↓
                 Verified exports + inference timing
```

The fixed mixture combines calibrated right-class probabilities:

$$
p_{\mathrm{mix}}(y=\mathrm{Right}\mid x)
= \sum_{k=1}^{K} w_k p_k(y=\mathrm{Right}\mid x),
\qquad w_k \geq 0,\quad \sum_k w_k = 1.
$$

Weights are fitted on the mixture-fit partition using constrained optimization (SciPy SLSQP). Candidate selection uses human-text validation log loss, which penalizes confident wrong predictions. Calibration, mixture fitting, and validation use distinct partitions; the final topic test stays sealed until candidates and decision rules are fixed.

The development-selected model combines **88.6% human-adapted Mistral** with **11.4% frozen, chat-formatted Qwen**. Neural routing based on 384-dimensional BGE-small embeddings was weaker during development. Every enabled expert runs on every passage: this is a dense prediction ensemble, so adding experts increases computation.

[Evaluation protocol](docs/human-text-development.md) · [Mixture implementation](src/possum/improvement_mixtures.py)

## Results

### Frozen topic stress test

The test contains **159 MITweet passages** across Energy Crisis, Mass Shootings, and Political Parties, with 111 left and 48 right labels. Models, calibration, weights, and decision rules were frozen before evaluation. No candidate was selected or retuned using these results.

| Frozen candidate | Macro-F1 ↑ | Balanced accuracy ↑ | Log loss ↓ |
|---|---:|---:|---:|
| Development-selected Mistral–Qwen mixture | 0.890 | 0.872 | **0.241** |
| Mistral LoRA adapter alone | **0.907** | **0.893** | 0.264 |
| DeBERTa compact alternative | 0.766 | 0.744 | 0.488 |

**Interpretation:** The mixture has lower probability loss on this sample, while Mistral alone makes fewer classification errors (12 versus 14). The evidence does not establish a universal ensemble advantage. DeBERTa's weaker topic-test performance, despite strong development results, shows why held-out evaluation matters.

This is a narrow stress test, not a broad benchmark of U.S. political text. Labels are mapped from human ideology facets rather than newly adjudicated overall U.S. left/right labels, and there is no author-disjoint guarantee. The mixture's descriptive 95% row-bootstrap interval for macro-F1 is **0.826–0.939**; it does not capture author/topic dependence or label uncertainty.

[Full results, per-topic scores, and uncertainty](reports/human-topic-stress.json)

### Accuracy–cost tradeoff

| Export | Median single request | Batch time per passage |
|---|---:|---:|
| Mistral–Qwen mixture | 6.38 s | 2.33 s |
| Mistral adapter alone | 1.36 s | 0.820 s |
| DeBERTa compact alternative | 1.34 s | 0.138 s |

Measured locally on 12 calibration passages. Requests include model loading, integrity checks, and cleanup; these are not warm-server latency measurements. All three variants passed single/batch prediction consistency checks. Mistral alone offers a compelling quality–cost tradeoff within this evaluation.

[Timing protocol and measurements](reports/human-inference-latency.json) · [Detailed development comparisons](docs/research-release-notes.md#development-comparison)

## Run locally

**Availability:** The repository includes source code, tests, and aggregate evaluation reports. Trained exports, raw corpora, and base-model caches are not included. Inference requires the corresponding local artifacts; cloning the repository alone does not provide a ready-to-run trained model.

### Install the source package

```bash
git clone https://github.com/ddrisco11/PoSSUM.git
cd PoSSUM
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
possum --help
pytest
```

For the full Apple Silicon research workflow:

```bash
pip install -e '.[dev,gliner,mlx,data,adapters]'
```

The Mistral/Qwen exports use MLX on Apple Silicon. DeBERTa has a verified CPU inference path. The [runtime snapshot](reports/human-development-runtime.json) records the evaluated environment; a clean-machine installation has not yet been verified.

### Predict with an existing local export

With the research run and cached base models available:

```bash
HF_HOME=cache/huggingface HF_HUB_OFFLINE=1 \
python -m possum.predict_development \
  --run runs/improvement-20261018 \
  --variant selected \
  --text "The government should expand affordable healthcare coverage through public funding."
```

Use `--variant single_expert` for Mistral alone, or `--variant compact --device cpu` for DeBERTa on CPU. Output includes left/right probabilities, a binary prediction, the gate-set decision threshold, and a separate equal-prior score. Inputs exceeding the recorded token limit are rejected rather than silently truncated.

[Release manifests and export checks](reports/human-model-release.json) · [Training and evaluation protocol](docs/human-text-development.md)

## Repository guide

| Path | Contents |
|---|---|
| [`src/possum/`](src/possum/) | Data processing, models, calibration, mixture fitting, evaluation, and inference |
| [`tests/`](tests/) | Tests for core behavior, checkpoints, scoring, and evaluation |
| [`scripts/`](scripts/) | Experiment, dataset, audit, and export utilities |
| [`reports/`](reports/) | Aggregate metrics, uncertainty estimates, runtime records, and release evidence |
| [`docs/`](docs/) | Methodology, dataset documentation, and detailed experiment findings |

## Scope and next steps

PoSSUM is a **research prototype** with verified local exports. Its main open questions are broader generalization and handling text outside the binary task:

- Validate on broader, independently adjudicated passages with balanced classes and document/author metadata. The human development validation set is left-skewed (366 left / 42 right), and 83.8% concerns Abortion or Women’s Right.
- Add and evaluate neutrality, ambiguity, nonpolitical-input handling, and abstention. Such inputs can currently receive confident binary predictions.
- Reassess the mixture-versus-single-model tradeoff on a new benchmark and in a persistent inference service.
- Verify clean installation and distributable model packaging. Data and component weights retain their upstream licenses.

## Technical documentation

- [Detailed research release notes](docs/research-release-notes.md) — full development comparisons, export details, and component commands.
- [Human-text evaluation protocol](docs/human-text-development.md) — admission rules, partitions, selection, and uncertainty methods.
- [Research release record](reports/human-model-release.json) — frozen variants, manifest hashes, metrics, and verification evidence.
- [Model registry](docs/models.md) and [dataset documentation](docs/datasets.md) — component and data provenance.
- [Historical experiments](docs/historical-results.md) — earlier evaluations and training commands.
