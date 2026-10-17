# Ten-hour development options on Apple M2, 16 GB

Assessment date: 2026-10-16. This report covers local research development with the currently downloaded corpora and model snapshots. It does not assume cloud compute, new data access, or a production-ready result. Dataset assembly and short backward-pass probes are complete; the overnight training schedule below has not been launched.

## Recommendation

Use the ten-hour window for **one complete Mistral QLoRA adapter, BERT adaptation, a regularized routing head, and an evaluated export**, retaining frozen Qwen and GLiNER as experts. GLiNER adaptation is a stretch task only after a short compatibility/timing check. This gives a realistic path to a complete comparison rather than unfinished adapters with no evaluation.

Two full large-model adapters plus all downstream scoring are unlikely to finish in ten hours. A smaller Mistral+Qwen adapter comparison is feasible if each is deliberately trained on a bounded subset. All-layer LoRA, repeated seeds for every backbone, multiple epochs over the literal downloaded candidate inventory, and OpenJev adaptation are outside this overnight scope.

Adapter training is technically feasible on this Mac: both Mistral and Qwen completed real backward/update steps during profiling. Those timing-probe updates were discarded; they are not trained release adapters or evidence of better classification.

## What data is actually available

The existing manifest has eight sources, five with nonempty local eligible text collections. There are **130,304 candidate occurrences**, reduced to **129,351 unique normalized texts**. That inventory is not an approved supervised training set.

A conservative assembly, preserving earlier test boundaries and development roles, produces:

| Role | Passages | Permitted use |
|---|---:|---|
| Expert fitting | 7,652 | Backbone adapters and fitted classification heads |
| Calibration | 1,967 | Expert temperatures and separately planned probability calibration |
| Router fitting | 2,195 | Mixture weights and controlled self-prediction experiments |
| Validation | 1,899 | Development comparison; not a fresh human benchmark |
| **Admitted development total** | **13,713** | Union for feature/score caching; never re-split blindly |
| Protected holdout file | 3,983 | Contains prior test-connected data and newly held-out tweet authors; not one interchangeable final benchmark |
| IBC diagnostic | 97 | Previously inspected human-labeled diagnostic only |
| Quarantine | 111,558 | Retained locally with reasons; excluded from this training release |

Development supervision comprises **11,016 heuristic-screened, party-labeled tweets**, **2,400 synthetic statements**, and **297 policy-mapped Manifesto passages**. Only **54 Manifesto passages remain in the fit partition**; 243 are validation. Therefore this training release is principally a weak tweet/synthetic model, not a broadly human-supervised ideology model.

Reasons for the reduction:

- The tweet screen retains 11,850 of 114,183 candidates before split/joint admission. It requires explicit policy vocabulary and a stance cue and excludes truncated, ceremonial and service-template material. Its precision/recall have not been human measured; a rejected row is not a verified nonpolitical example. The filter may discard useful political text.
- Manifesto has 4,289 occurrences but 3,827 normalized unique texts. Full document/duplicate closure protects 2,417 from development. Preserving earlier development assignments additionally quarantines a 1,084-row component that connects previously separate fitting/calibration/routing roles.
- Manifesto and the platform dataset share 491 exact texts, with 158 label conflicts. Party identity and policy-derived ideology are different targets. Contradictions are quarantined rather than resolved by arbitrary source precedence.
- Platform rows lack document identities and overlap a previously inspected diagnostic set; the source remains quarantined pending provenance recovery.
- Full IBC, Qbias raw text and SCCC message text are not locally available. The supplementary synthetic bias set has no nonneutral rows meeting the 20-word minimum.

All **1,174 originally reserved IDs** remain outside development: 1,159 appear in the assembled protected file and 15 in quarantine. The protected file also contains additional document-connected buffer examples and 832 new author-held-out tweets. These must be reported separately; the buffer is not a newly selected final test. Full provenance protection also applies to quarantined rows.

This is a deliberately conservative research dataset. A future broader version could recover original platform documents and review excluded tweets. That would be a new audited dataset version, not a reason to erase current test protection or silently train on all 129,351 candidates.

Artifacts: [assembly counts](../reports/assembled-corpus.json), [tweet filter audit](../reports/tweet-admission.json), [provenance audit](dataset-admission-audit.md). Raw text remains under ignored `data/assembled-v1/`.

## Measured local compute

| Operation | Observed speed / memory | Evidence and limits |
|---|---|---|
| Mistral 7B 4-bit QLoRA | Mean 0.80 s/example; peak MLX 4.13 GiB | 12 short steps, first two excluded from timing mean; rank 8, q/v projections in the last four layers, batch 1, checkpointed activations |
| Qwen3.5 9B 4-bit QLoRA | Mean 2.92 s/example; peak MLX 6.73 GiB | Eight short steps, first two excluded; rank 8, selected full/linear attention projections within the last four layers, batch 1 |
| BERT full-parameter backward/update | About 10 examples/s | Earlier 12-step MPS benchmark, batch 8, fixed 256 tokens, float32; not a LoRA measurement |
| Frozen Mistral scoring | 0.696 s/example | Measured across 797 passages |
| Frozen Qwen scoring | 1.169 s/example | Measured across 797 passages |
| Frozen GLiNER scoring | About 0.168 s/example | 134 s across 797 passages, including that stage's load overhead |
| Routing fits and ablations | 2.8 s at pilot size | Cached scores; larger datasets and tuning add time, but large-model scoring is the bottleneck |

These are short profiles and one pilot's timings, not ten-hour sustained-throughput guarantees. They exclude some or all loading, tokenization, checkpointing, implementation, memory pressure and thermal effects. MLX peak allocation is not whole-system RAM use. The machine has 16 GB shared with macOS and other apps; models must run sequentially.

The Mistral probe inputs were 72–132 tokens; Qwen inputs were 61–114. The assembled fit set averages about 106 Mistral tokens and 90 Qwen tokens, with maxima 174 and 152. All fit examples fit within 256 tokens using the current prompt without truncation. A separate three-step Qwen stress probe at 256 tokens took 11.7–13.4 seconds per step and peaked at 8.88 GiB of MLX allocation. Those inputs repeat token sequences solely to test length/memory; they are not training data or an accuracy experiment. The requested MLX allocation limit did not bound observed peak allocation, so it must not be treated as a hard whole-system memory guarantee. This result reinforces using actual token lengths and memory checks rather than extrapolating the short-passage rate to long inputs. See the [length-stress report](../reports/qwen-lora-length-stress.json).

Sources: [Mistral timing](../reports/mistral-lora-throughput.json), [Qwen timing](../reports/qwen-lora-throughput.json), [BERT timing](../reports/bert-training-throughput.json), [fit token lengths](../reports/assembled-fit-token-lengths.json), [completed pilot](../reports/frozen-four-pilot.json).

## Why all stages do not fit automatically

Using the measured short-step means, one pass over the **7,652 fit examples** costs approximately:

- Mistral adapter: **1.7 hours** before overhead.
- Qwen adapter: **6.2 hours** before overhead.
- BERT full fine-tune: **13 minutes** before overhead; actual adapter training must be timed separately.

After changing an expert, its old predictions and fitted temperature are stale. It must score the calibration, routing and validation partitions again. Their union is **6,061 passages**. Frozen Mistral and Qwen together would need **3.1 hours** just to score that union at the measured rate. Adapted inference may differ. GLiNER adds roughly 17 minutes, and BERT/BGE add further time.

Thus both full large-model adapters plus these scoring passes already imply **about 11 hours**, before compact-model training, implementation, selection or checkpoints. A practical allowance is **14–18 hours or more**, particularly if longer inputs or sustained throttling slow Qwen. Two or three epochs multiply the training part.

For comparison, scoring **100,000 passages** through the two frozen LLMs alone takes about **52 hours** on this machine. One narrow-adapter pass over that many passages extrapolates to roughly 22 hours for Mistral and 81 hours for Qwen. These are scale illustrations, not approval to train quarantined data.

“Use the full dataset” should mean all admitted examples have a documented role. It must not mean the same examples train the expert, calibrator and routing head, which would produce optimistic stacking results.

## Options

| Option | Ten-hour deliverable | Tradeoff |
|---|---|---|
| **A — Complete one strong adapter, recommended** | Mistral QLoRA over the full fit partition for one pass; BERT adaptation if its smoke test succeeds; frozen Qwen/GLiNER; recalibrated ensemble and regularized routing comparison | Most likely to complete an evaluated end-to-end model. A second Mistral epoch or GLiNER adaptation is conditional on remaining time. |
| **B — Adapters for both large models** | Mistral and Qwen QLoRA on a fixed, source/label-balanced 2,000–3,000-example fit subset each; full admitted compact-feature baseline; common held-out scoring and routing | Demonstrates adaptation in both LLMs, but cannot honestly claim full-fit training for them. Subset manifests must be saved. |
| **C — Broad compact-model development** | BERT and, if supported on MPS, GLiNER adaptation over the full fit partition; full BGE representation cache; careful calibration/routing and cheap deployment comparisons | Best emphasis on inference cost and robustness. Large LLMs stay frozen, optionally scored only on the disjoint routing/evaluation partitions. |
| **D — All four adapted on the full fit partition** | Longer run, likely 14–18+ hours for one conservative pass with downstream evaluation | Not a dependable ten-hour promise. GLiNER training still needs local profiling; extra epochs and all-layer LoRA cost more. |
| **E — Literal candidate inventory or OpenJev** | Requires revised data admission and substantially more compute | Multi-day local work or external GPUs; not an overnight plan. OpenJev remains outside the 16 GB workflow. |

Option B's 3,000-example Mistral and Qwen runs imply about 0.7 and 2.4 hours of training before overhead. Their combined scoring of the 6,061-row development evaluation union adds about 3.1 hours. Implementation, compact-model work and contingency consume the rest. It is feasible but less forgiving than Option A.

## Recommended ten-hour allocation

A deadline-aware run should save a useful result before trying optional work:

| Window | Work | Stop / fallback condition |
|---|---|---|
| Hour 0–1.5 | Final dataset validation; implement adapter serialization/loading, resume and model-specific loss; BERT/GLiNER smoke tests; start caches | Preserve the existing baseline; abort only a failing component, not the entire study |
| Hour 1.5–4 | Full-fit Mistral narrow QLoRA pass and BERT adaptation; checkpoints during training | Stop adapter updates at a saved boundary if time or memory budget is exceeded |
| Hour 4–8 | Score calibrated experts sequentially on calibration/gate/validation; BGE features; reuse only caches with identical model+adapter hashes | No optional second large-model training run until required scores are complete |
| Hour 8–9 | Temperature fitting, equal-average/fixed/regularized-router/self-branch comparisons; source/group metrics and ablations | Prefer the strongest validated baseline; do not automatically select the most complex model |
| Hour 9–10 | Reload check, inference example, packaging, aggregate report, budget buffer | Save a coherent partial result with explicit missing stages if something overruns |

This is a work allocation, not a claim that the current scripts already implement the full unattended schedule. The existing pipeline handles frozen experts; adapter-aware training, loading, cache invalidation and optimizer-state resume require integration. The current short probes establish feasibility, not that all production plumbing is finished.

GLiNER adaptation can replace optional second-epoch work only if its official classification training path works on this MPS setup and has a measured budget. Its installed package has PEFT LoRA support, but a successful inference run does not prove the training path is ready. BERT can use standard PEFT q/v LoRA with its classification head trainable, or full fine-tuning if that is faster to validate locally. The prior BERT throughput measurement must not be relabeled a LoRA measurement.

## Model and evaluation design

- **Adapter objective:** preserve the current prompt and label mapping; train the correct left/right continuation with loss only at the decision token(s). Do not use prompt-language-modeling loss as a substitute for the classification objective. Verify tokenizer boundaries and label order for each model.
- **Conservative adapters:** 4-bit frozen backbone, rank 8, selected attention projections within the final four layers, batch 1 with optional accumulation. These are genuine learned LoRA matrices, but a narrower adaptation than all-layer LoRA. Archive the exact target-module list and trainable parameter count.
- **BGE:** keep the 384-dimensional encoder frozen initially so changes in expert predictions and routing can be interpreted separately.
- **Routing:** train only on the dedicated 2,195 gate examples. Start with a linear or smaller regularized gate, compare against equal averaging, and tune stopping/regularization inside development groups. Repeated initializations do not replace independent document/author splits.
- **Self prediction:** retain as an explicit ablation. Prefer training a BGE classifier on the expert-fit partition, calibrating it separately, then treating it as a controlled additional expert; compare with the jointly trained branch rather than letting it absorb all weight unchecked.
- **Calibration:** refit temperatures after adaptation. If final mixture calibration or abstention thresholds are fitted, reserve a disjoint calibration subdivision or use a documented cross-fitting design. Do not choose thresholds on final-test performance.
- **Evaluation:** per-source macro-F1, balanced accuracy, log loss, Brier score, calibration and coverage/accuracy. Report author/document group counts and class imbalance. Preserve existing test IDs and report any new tweet holdout separately.
- **Data balance:** source-aware sampling/weights, label balance and a synthetic contribution cap are needed. Overall accuracy on a tweet-dominated pool would not demonstrate political viewpoint understanding across genres.

No new human labels are created by this work. The 97 IBC examples are already inspected diagnostics, and the newly held-out tweets still have party-proxy labels. A shippable claim needs independent human annotation, nonpolitical/neutral/ambiguous coverage and an agreed accuracy/coverage target. The overnight result can be an installable research prototype, not an established reliable political-slant service.

## Checkpoint and export requirements

Frozen score caches already save in ten-passage atomic chunks. LoRA training requires more: adapter weights, optimizer state, step/epoch, random/sampler state, model revision, dataset/split hashes, and atomic versioned manifests. A weights-only MLX `--resume-adapter-file` is a warm restart, not necessarily an exact optimizer-state continuation. Verify resume by comparing a short uninterrupted run with a checkpoint/reload continuation before a long run.

Keep the last two valid adapter checkpoints and a separate best development checkpoint. Budget around short chunks and reload checks. All model/adapter-specific score caches need hashes so a frozen cache cannot be silently reused for an adapted expert. The exported configuration must specify base revisions, adapter locations, calibration parameters, expert order, BGE revision, router architecture and decision policy.

Run only one GPU-heavy process at a time. Power and wakefulness must be confirmed at launch; no overnight run is active from this assessment. A process-scoped sleep inhibitor can cover the chosen run while plugged in. Preserve checkpointing even if sleep prevention is enabled. Raw corpora and local weights remain outside Git; publish code, configuration, hashes and aggregate metrics.

## Reproduce the preparation

Run from the repository root with the existing downloaded files and historical split manifests present:

```bash
.venv/bin/python scripts/audit_tweet_admission.py
.venv/bin/python scripts/assemble_corpus.py
.venv/bin/python scripts/check_assembled_corpus.py data/assembled-v1
```

Use the preassigned `fit.jsonl`, `calibration.jsonl`, `gate.jsonl` and `validation.jsonl` directly. Do **not** pass their union to the old pilot CLI to create a new split; that would discard the audited assignments. The prior split manifests and source files are local prerequisites, not bundled public datasets.

Training references: [MLX-LM LoRA/QLoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md), [GLiNER training guide](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/9-training.md). Published library support is separate from locally measured compatibility and model quality.
