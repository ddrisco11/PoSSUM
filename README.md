<img src="assets/possum-model-card.png" alt="Cartoon red and blue possum wearing an Uncle Sam hat beside the PoSSUM title" width="100%">

# PoSSUM
**Political Slant Scoring Using Mixtures**

## Model overview

PoSSUM classifies the political viewpoint expressed in short English passages about U.S. politics. It combines component predictions with a neural gate conditioned on 384-dimensional BGE-small embeddings. The hybrid gate can also assign weight to its own prediction.

| Property | Specification |
|---|---|
| Task | Binary political slant classification |
| Labels | `0 = Left`, `1 = Right` |
| Input | Approximately 20–150 words |
| Architecture | Dense mixture of calibrated predictions with an optional self-prediction branch |
| Components | BERT, BGE-small, GLiNER; optional ModernBERT, OpenJev, Mistral and Qwen adapters |
| Status | Development implementation with synthetic training and external human-text diagnostics; no validated release checkpoint |
| Language and domain | English, U.S. politics |

Model weights and source datasets are downloaded separately. Every enabled expert runs; the gate does not provide sparse inference routing.

## Training and evaluation

The compact pilot uses frozen encoders with fitted classification heads. Expert fitting, temperature calibration, gate training and validation use disjoint groups. Comparisons include single models, equal averaging, a fixed mixture, an experts-only gate and a hybrid gate with its own classifier.

The first pilot used 480 synthetic development passages and achieved 0.987 BERT macro-F1 on a single synthetic validation topic. A subsequent **source-held-out evaluation** used 97 human-labeled IBC sentences and 300 weakly labeled party-platform passages, without retraining or recalibrating on either source.

| Model | Human IBC macro-F1 | Platform macro-F1 (weak labels) |
|---|---:|---:|
| BERT + fitted head | 0.514 | 0.483 |
| GLiNER 2.5 | 0.566 | 0.602 |
| Hybrid gate, seeds 42–44 | 0.514–0.531 | 0.495–0.499 |

**The synthetic-trained mixture does not yet demonstrate reliable transfer or an advantage over the strongest external expert.** All evaluated models also had external log loss worse than a constant 0.5 predictor. Human-data training and calibration remain necessary. The IBC sample is small and lacks original document IDs; these are diagnostic results, not a definitive benchmark.

The reserved synthetic test remains unopened. Its group-constrained split is 75/25; the broader evaluation target remains 80/20. External source-held-out diagnostics are separate from that split.

See the [human-text evaluation report](reports/external-evaluation.md), [initial synthetic pilot](reports/initial-pilot.md), [training protocol](docs/evaluation.md), [external evaluation protocol](docs/external-evaluation.md), [model registry](docs/models.md), and [dataset manifest](docs/datasets.md).

A frozen four-expert pilot is configured for BERT, GLiNER 2.5, Mistral 7B v0.3 and Qwen3.5 9B, with 4-bit MLX causal scoring. Only the BERT classification probe and mixture heads learn parameters; all backbones remain frozen. A bounded U.S. Manifesto subset uses document-separated calibration, routing training and validation. See [corpus status](docs/corpus-status.md).

## Intended use and limitations

PoSSUM is a development model for studying expressed political viewpoint in text. It does not infer an individual's beliefs. Party and publisher metadata provide noisy supervision rather than passage-level ground truth. Neutral, ambiguous and nonpolitical text is outside the binary training target; abstention and politicalness detection need separate evaluation.

A small human-annotated sample has been evaluated; representative human-text evaluation and broader domain testing remain incomplete. OpenJev, Mistral, Qwen and ModernBERT have not completed checkpoint-level inference in the pilot. Training data and component weights retain their upstream licenses; see the model registry and dataset manifest.

## Local setup (Apple Silicon)
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
possum --help
```

Use one component at a time and cache scores. The default pilot avoids 7–27B weight downloads. OpenJev is disabled on a 16 GB Mac; its smallest official MLX checkpoint alone is approximately 15 GB. Optional MLX causal audits require `pip install -e '.[mlx]'` and explicit model selection.

## Live training monitor

Open the read-only Tkinter dashboard from the repository root:

```bash
.venv/bin/python scripts/monitor_training.py
```

It refreshes every two seconds with component progress, measured scoring speed, current-stage ETA and the live pipeline log. Use `--run /path/to/run` to inspect another run. Closing the window leaves training running. The Python installation must include Tkinter.

Scoring checkpoints are saved every ten passages using atomic replacement and disk synchronization. Resume an interrupted frozen four-expert run with `python -u scripts/run_frozen_four.py`; completed expert caches are reused before loading weights. Resume only after the previous process has stopped. The small routing fits restart if interrupted; expert scoring remains cached. Checkpoints live in ignored `runs/`, not on GitHub.

## Reproducibility
Record checkpoint revisions, dataset hashes, split membership, dependency versions, seeds, score prompts, and device in every run. Raw corpora, model caches and sample-level outputs stay ignored. Public reports contain aggregate metrics only.

## Run an expert independently
```bash
# These commands emit features or two-column scores; they do not train a model.
possum score bge data/development.jsonl --output runs/bge-features.npz --device mps
possum score gliner data/development.jsonl --output runs/gliner.npz
possum score openjev data/development.jsonl --output runs/openjev.npz
# Run a local 4-bit checkpoint:
possum score mlx data/development.jsonl --model-id /path/to/pinned-mlx-model --output runs/causal.npz
# Integrate frozen zero-shot caches into the same calibrated mixture:
possum pilot data/corpus.jsonl --expert-scores gliner=runs/gliner.npz --output runs/with-gliner
```

Score only development rows during selection; use `split.json` to construct that input. External caches must cover every selected development ID, and include their JSON metadata sidecars. Learned external heads are rejected because fold provenance is required. Encoder `score` commands emit features rather than untrained classification logits. `--model-id` swaps the checkpoint within a module.

An optional GitHub Actions test workflow is provided in `docs/ci-example.yml`; automated CI is not enabled.

## The gate can also answer
The **hybrid gate** assigns probability mass to its own BGE-conditioned prediction as well as to external experts. Its K+1 weights sum to one. The self classifier and routing head train jointly, while experts stay frozen. Reports compare this with an experts-only gate and expose the average self weight.

```bash
python scripts/replay_mixtures.py runs/synthetic-42 --output reports/hybrid-pilot.json
```
