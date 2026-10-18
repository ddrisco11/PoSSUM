# Human-text development benchmark

The October 18 revision centers model selection on political viewpoints expressed in text. It excludes Manifesto policy mappings and party-only labels from fitting. Human MITweet annotations are mapped to a binary axis only when related, non-central labels among facets I1, I3, I4, I5 and I6 agree. Automated semantic screening checks whether the passage expresses a reasonably self-contained stance. This screening is not an additional human annotation, and agreement among facets is not equivalent to a direct overall U.S. left/right label.

The revision admits 12–150 words, preserving meaningful short tweets. All source topics are eligible. Explicit partisan framing can qualify without a policy proposal; typos, thread markers and missing handles do not independently disqualify understandable statements. Factual news without attributable ideological framing, unresolved context and conflicting facet labels remain excluded. Original labels and occurrence-level provenance are retained locally.

| Partition | MITweet | Synthetic | Total |
|---|---:|---:|---:|
| Fit | 1,224 | 956 | 2,180 |
| Calibration | 204 | 400 | 604 |
| Mixture fit | 204 | 359 | 563 |
| Development validation | 408 | 398 | 806 |
| Topic holdout | 159 | 0 | 159 |

The topic holdout contains all admitted MITweet Energy Crisis, Mass Shootings and Political Parties passages, including linked duplicate components. It remains sealed during development selection. This holds out human-source topic groups; it does not establish that the same policy themes are absent from synthetic training statements. Other human rows are split 60/10/10/20 across fit, calibration, mixture fit and validation. Exact and near duplicates are connected transitively before partitioning. Synthetic examples preserve their historical assignments. Protected historical examples and previously inspected diagnostics cannot enter fitting.

Source author and thread identifiers are unavailable for MITweet. Partitioning therefore prevents detected lexical-duplicate leakage but does not establish author-disjoint evaluation. Human validation contains 366 left and 42 right examples, with substantial abortion and women's-rights coverage. Broad political generalization remains an open evaluation requirement.

## Comparisons

Word/character TF-IDF, frozen BGE and frozen BERT logistic classifiers establish low-cost baselines. Each compares human-only fitting against mixed human/synthetic fitting. Fit weights balance source–class cells. L2-regularized Platt calibration uses only the human calibration partition. The initial best lexical baseline has human validation macro-F1 0.727, balanced accuracy 0.690 and log loss 0.210. Its pooled human-plus-synthetic macro-F1 is 0.869; that pooled number must not replace the human result.

BERT LoRA and full fine-tuning compare the same two fitting regimes. Frozen GLiNER, Mistral and Qwen use the same development rows. A newer encoder comparison uses [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base), a 149-million-parameter Apache-2.0 backbone. Candidate training and scoring are checkpointed with immutable input hashes and model revisions.

Primary selection uses human development-validation log loss. Macro-F1, balanced accuracy, source/topic breakdowns, calibration error and class-specific behavior qualify that selection. A constant human-calibration-prior predictor checks whether apparent performance exceeds a no-text baseline; it also provides an explicit shrinkage component without running a large model. Mixture fitting is restricted to the designated gate partition. An optional self contribution is supplied by a BGE classifier fitted only on the fit partition. The topic holdout is scored only after candidate selection is frozen. Repeated validation comparisons are exploratory development, not independent final tests.

The calibrated output estimates a binary class probability under the calibration distribution; it does not measure ideological intensity. A probability near one-half does not by itself establish neutrality or lack of political content. Abstention and out-of-domain behavior require separate validation.

## Calibration robustness

An additional comparison chooses among unconstrained Platt calibration, nonnegative-slope Platt calibration, positive temperature scaling, and temperature scaling with a calibration-prior intercept. Five-fold stratified cross-validation uses only the 204 human calibration examples; a log-loss tolerance of 0.00001 favors the existing recipe when differences are numerical. The chosen method is then fitted on all human calibration examples. These alternatives are compared alongside the original calibrated experts during model and mixture development selection.

GLiNER illustrates the uncertainty from only 21 right-labeled calibration examples: its original fitted slope was slightly negative, reversing a raw validation AUC of 0.591 into 0.409. The calibration-only comparison chooses a nonnegative slope that is effectively constant. This avoids reversed ordering without establishing useful discrimination. Temperature scaling gives only a small lexical-model change; neither finding establishes a general accuracy improvement.

Each completed mixture comparison and each calibration artifact version is retained locally. Export candidates must reproduce cached calibration predictions after loading from disk. The active export points to the lowest human-validation-log-loss candidate among verified bundles for the exact same dataset, so a newer candidate cannot silently replace a better incumbent. This remains development selection, not final-test evidence.

## Compact-model distillation comparison

Frozen Mistral improves human development macro-F1 to 0.852 and log loss to 0.145; the partial fixed-mixture comparison reaches 0.873 and 0.124. Neural routing is weaker on this comparison. These findings motivate a compact distillation trial rather than assuming that greater architectural complexity helps.

After frozen expert and encoder comparisons, the lower-human-validation-log-loss frozen Mistral/Qwen model supplies soft left/right targets for the 2,180 fit passages only. These targets remain explicitly model-generated. The compact backbone is chosen from the initial BERT/ModernBERT recipe comparison and initialized from its pretrained base. Two four-epoch mixed-data trials use teacher-loss weights 0.25 and 0.50, temperature 2, learning rate 0.00002 and seed 42. Their objective combines labeled cross-entropy with temperature-scaled KL divergence; source–class weights apply to both terms. Raw teacher logits use the existing forced left/right prompt. Calibration, mixture fitting, validation and the sealed topic holdout retain their separate roles.

The teacher plan, model revision, dataset hash, target identities and completed target-cache hash are recorded. Teacher scoring saves every ten passages; student checkpoints include optimizer/RNG state and the target-cache hash. The comparison is prospective: no distillation improvement is claimed before training and evaluation finish.
