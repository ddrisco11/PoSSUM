# Model decisions — checked 2026-10-16

| Module | Checkpoint | Role |
|---|---|---|
| BERT | `google-bert/bert-base-uncased` | Exact original BERT family baseline; frozen encoder plus fitted head |
| ModernBERT | `answerdotai/ModernBERT-base` | Modern BERT replacement, separately swappable |
| BGE | `BAAI/bge-small-en-v1.5` | 384-dimensional normalized CLS embeddings, gate input and standalone baseline |
| GLiNER | `fastino/gliner2.5-base-v1` | Current English family release; optional extra |
| GLiNER 2.0 | `fastino/gliner2-base-v1` | Explicit original requested checkpoint, swappable |
| OpenJev | `openjev/openjev` | 27B typed-decision expert, official helper API |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` | Latest identified Mistral 7B revision, candidate log-probability audit |
| Qwen | `Qwen/Qwen3.5-9B` | Recent approximately 7B-class general model; newer 27B releases exceed scope |

Official sources: [BERT](https://huggingface.co/google-bert/bert-base-uncased), [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base), [BGE](https://huggingface.co/BAAI/bge-small-en-v1.5), [GLiNER 2.5](https://huggingface.co/fastino/gliner2.5-base-v1), [GLiNER 2.0](https://huggingface.co/fastino/gliner2-base-v1), [OpenJev](https://huggingface.co/openjev/openjev), [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3), [Qwen](https://huggingface.co/Qwen/Qwen3.5-9B).

On 16 GB unified memory, OpenJev is deferred: even its official 4-bit MLX weights are about 15 GB, before runtime overhead. The adapter connects to a separately launched official local helper and records server version. No paid endpoint is assumed. OpenJev weights are CC BY-NC 4.0; upstream model/data licenses remain separate from repository code.

Mistral/Qwen should use explicit MLX 4-bit conversions on this machine, one at a time. A conversion must be recorded as its own checkpoint and revision; quantized results must not be presented as full-precision measurements. The generic Transformers causal runner is for adequately provisioned hardware; it is not the default M2 path. Qwen3.5 uses a multimodal model class even for text input.
