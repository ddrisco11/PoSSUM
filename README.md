<img src="assets/cover.svg" alt="Red and blue vector possum beside PoSSUM title" width="100%">

# PoSSUM
**Political Orientation and Slant Scoring Using Mixtures**

A short experiment: does a small, embedding-conditioned mixture beat a strong single model at classifying the political viewpoint expressed in a short U.S. political passage?

**Labels:** `0 = Left`, `1 = Right`. **Status:** compact M2 pilot completed; [results and limitations](reports/initial-pilot.md). No scientific conclusions yet. Model weights and source datasets are downloaded separately and never committed.

## Experiment in three steps
1. Freeze encoders; run a small grouped pilot with BGE-small, BERT, GLiNER, and a lexical baseline.
2. Calibrate component scores; compare best single, equal average, fixed mixture, and a 384 → 32 → K neural gate. Run leave-one-component-out ablations.
3. Only if validation evidence supports it, expand the data, audit Mistral/Qwen logits, and fine-tune selected encoders. Open the reserved test set once the recipe is fixed.

This is a dense mixture of predictions, not a sparsely routed language model: every enabled expert runs. A gate does not automatically reduce inference cost.

See [experiment protocol](docs/experiment.md), [model registry](docs/models.md), and [dataset manifest](docs/datasets.md).

## Local setup (Apple Silicon)
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
possum --help
```

Use one component at a time and cache scores. The default pilot avoids 7–27B weight downloads. OpenJev is disabled on a 16 GB Mac; its smallest official MLX checkpoint alone is approximately 15 GB. Optional MLX causal audits require `pip install -e '.[mlx]'` and explicit model selection.

## Reproducibility
Record checkpoint revisions, dataset hashes, split membership, dependency versions, seeds, score prompts, and device in every run. Raw corpora, model caches and sample-level outputs stay ignored. Public reports contain aggregate metrics only.

This experiment measures expressed viewpoint, not an individual's beliefs. Party/source labels are noisy supervision; they are not human verification of a passage's viewpoint. Neutral and ambiguous input is outside the binary training target. Production abstention and politicalness detection remain separate evaluation tasks.

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
