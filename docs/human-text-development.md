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


## Calibration variants and mixture cost

Calibration variants of one predictor share the same underlying forward pass. New exports copy each predictor artifact once, cache its raw scores within a prediction batch, and apply each calibrator separately. Batch caches cannot carry predictions into a different request. Aggregate prediction weights still identify each calibrated component.

A grouped fixed-mixture comparison regularizes the total weight assigned to each underlying predictor, so creating an additional calibration variant does not independently change that predictor's regularization target. The legacy per-component penalty remains reported. In the partial Mistral/compact comparison, the stronger grouped penalty increases macro-F1 to 0.883 but worsens log loss to 0.129, versus 0.873 and 0.124 for the selected legacy mixture. The predeclared log-loss criterion therefore retains the incumbent. [Grouped comparison](../reports/human-grouped-mixture-audit.json).

## Small mixtures and alternative exports

A predefined small-mixture family checks whether two predictors can retain the full ensemble's performance at lower inference cost. It compares pairs from the original-calibration lexical model, a compact model, frozen Mistral and frozen Qwen. The lexical/compact representatives are selected by human gate log loss; each pair's nonnegative convex weights are fitted on the same gate partition. Human validation still selects among the resulting models. This adds exploratory comparisons and does not create an independent test.

A named candidate can be exported with `python -m possum.improvement_inference export RUN --candidate NAME`. Each export must pass reload parity. Exporting a smaller alternative does not replace a better verified incumbent; the bundle ledger retains both for later accuracy/latency comparisons.

## Qwen prompt-format comparison

Frozen Qwen3.5-9B scores substantially below Mistral with the shared raw-completion prompt: human macro-F1 0.604 and log loss 0.265. Its local tokenizer artifact provides a conversation template with an explicit `enable_thinking=False` assistant prefix. A separate, versioned `qwen_chat` comparison places the same classification task in the user turn and forces the assistant completion `This statement reflects a` followed by ` left` or ` right`. Both continuations are single tokens in the pinned tokenizer. No reasoning or completion is sampled.

The raw-completion baseline remains unchanged. Chat-template scores have their own cache, calibration and provenance, and use the same calibration/gate/validation partitions. This tests a plausible formatting mismatch; it does not assume Qwen will improve or attribute the observed gap to formatting before the comparison finishes.

Default export selection compares the selected winners from the current comparison and all archived comparisons with the same frozen dataset hash. Adding experts can change fitted mixture weights and make a new winner worse, so an earlier winner remains eligible even if it had not yet been exported. Historical export reads the historical score arrays and router parameters, recomputes decision rules without modifying the archive, and records the comparison hash in the bundle. Reload parity is still required. Explicit named-candidate exports use the current comparison unless `--comparison-dir` selects an archive. This preserves the development selection criterion; it does not make repeated validation selection an independent test.

## Human-only Mistral adapter comparison

A separate Mistral LoRA trial starts from the pinned frozen base and uses only the 1,224 MITweet fit passages. Synthetic data and all calibration, gate, validation and holdout rows are excluded from adapter fitting. The recipe uses rank 8 on query/value projections in the last four layers, scale 16, zero dropout, Adam at 0.00001, four-example gradient accumulation, gradient clipping at norm 1 and class-balanced binary next-token cross-entropy. Two epochs are scored and calibrated separately on the unchanged development partitions, retaining the frozen expert as a comparison.

Recovery checkpoints contain adapter tensors, optimizer state, deterministic epoch/order position, dataset/recipe hashes and cumulative training time. Before sustained fitting, an actual accumulated update must reproduce after optimizer/adapter reload. Zero dropout and deterministic ordering are explicit assumptions of that check. Completed epoch adapters are immutable and scoring verifies their weight hashes. This is a prospective trial; no improvement is claimed before results exist.

## Final topic-stress evaluation

The final workflow first exports and verifies the selected model and, optionally, a compact alternative. `python -m possum.final_evaluation freeze RUN --compact-bundle BUNDLE` then fixes their manifests, decision rules and the dataset/holdout hashes without reading holdout passages. Normally all planned comparisons must be complete; a partial-run freeze requires an explicit reason. A freeze cannot be replaced, and subsequent training, calibration/scoring development, mixture fitting and export selection are disabled for that run.

`python -m possum.final_evaluation evaluate RUN` checks the frozen artifacts, durably records exposure before opening holdout text, and saves predictions every ten examples. Interrupted scoring resumes only with the same selection and hashes. Completed evaluation returns its existing report instead of predicting again. Thresholds remain those fitted on the gate partition. Reports include per-source/topic results, class counts and end-to-end batch timing; those timings include loading and integrity checks and are not warm-service latency. No candidate may be retuned or selected using these holdout results.

## Initialization sensitivity

The selected mixed-data BERT recipe was repeated with seeds 43 and 44 alongside seed 42. Their best development checkpoints have human log loss 0.2004, 0.2290 and 0.2196, respectively (mean 0.2163; sample standard deviation 0.0146). Predictions disagree at the 0.5 decision threshold on 28 of 408 human validation passages. These are three initializations on one split, with checkpoint selection on that same validation set, rather than independent data replications.

An equal-weight average of the three calibrated predictions is now included as a fixed candidate without additional weight fitting. It reaches human log loss 0.1972 and macro-F1 0.748. Its paired loss difference against seed 42 is −0.0033, with a descriptive 95% row-bootstrap interval of −0.0181 to +0.0115. This small, uncertain change does not establish superiority; the Mistral-based mixture remains the strongest verified development candidate. [Seed results](../reports/human-seed-sensitivity.json).
