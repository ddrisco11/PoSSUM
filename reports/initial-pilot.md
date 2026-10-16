# Initial M2 pilot — 2026-10-16

**Decision: keep BERT as the reference; MoE value remains unproven.** No full training or large-model inference was run.

## What actually ran

- 3,200 synthetic source rows downloaded; upstream labels inverted explicitly.
- 480 development excerpts: 240 expert-fit, 80 calibration, 80 gate-fit, 80 validation.
- Eight issue groups; two reserved for test (800 rows, 25%). This is an explicit group-constrained deviation from the intended 80/20 target, not an exact 80/20 split.
- Final-test texts were not encoded or scored. The 80 validation excerpts belong to one issue group, so confidence intervals are withheld.
- Frozen BGE-small and BERT on MPS, fitted logistic heads, lexical baseline, standalone BGE neural head; GLiNER 2.5 on CPU.
- Replayed fixed, expert-only and self-augmented gates across three initialization seeds using the same frozen scores and fixed data split. These are not three independent data replications.

## Validation observations

| Model | Macro-F1 | Log loss |
|---|---:|---:|
| tfidf | 0.947 | 0.1284 |
| bge | 0.934 | 0.2820 |
| bert | 0.987 | 0.0950 |
| bge_neural | 0.817 | 0.3878 |
| GLiNER 2.5 | 0.644 | 0.5371 |
| expert_gate, seed 42 | 0.987 | 0.0930 |
| hybrid_gate, seed 42 | 0.987 | 0.0942 |
| expert_gate, seed 43 | 0.987 | 0.0936 |
| hybrid_gate, seed 43 | 0.987 | 0.0921 |
| expert_gate, seed 44 | 0.987 | 0.0940 |
| hybrid_gate, seed 44 | 0.987 | 0.0922 |

The hybrid self branch received roughly 0.02–0.03% average routing weight. This only describes this small fit; it does not imply the branch is intrinsically useless. BERT dominated the routing. GLiNER and the standalone BGE neural expert are candidates for further ablation, not justified permanent exclusions. The tiny mixture gain over BERT is not persuasive evidence of benefit.

## Verification and limits

Eight local tests cover provenance failures, conflicting labels, transitive grouping, disjoint splits, temperature calibration, convex gate behavior, multi-token candidate boundaries, and gradient flow into the self branch. GLiNER, BGE and BERT completed real inference. Mistral, Qwen, OpenJev, MLX and ModernBERT adapters have not completed checkpoint-level inference in the pilot environment. Their readiness is not implied by unit-test success.

`requirements-tested.txt` records installed package versions. Model revisions are pinned in the package registry. Compact encoder inference used normalized pooled features, eager attention, batch size 8 and a 256-token cap; overlength inputs may be truncated. Runtime figures in compact-pilot.json include loading/feature extraction/head fitting and are not isolated inference benchmarks. Peak memory was not measured.

The first synthetic pilot does not satisfy the final human-text evaluation requirement. IBC/Manifesto/platform/tweet admission still needs original grouping provenance and verified label mappings; optional datasets remain manifest entries. No raw corpora or model weights were published.

## Further evaluation

Recover human-text document/author groups and assemble several independent validation groups. Compare BERT, lexical baseline, GLiNER, expert gate and hybrid gate on those before deciding to drop experts or fine-tune. OpenJev is deferred for memory; Mistral/Qwen require a quantized backend and further runtime validation.
