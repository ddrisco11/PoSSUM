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

The teacher plan, model revision, dataset hash, target identities and completed target-cache hash are recorded. Teacher scoring saves every ten passages; student checkpoints include optimizer/RNG state and the target-cache hash. Both four-epoch comparisons are complete. With alpha 0.25, epoch 3 reaches human macro-F1 0.802 and log loss 0.206; alpha 0.5 selects epoch 3 at 0.758 and 0.209. The original seed-42 BERT reaches 0.776 and 0.200. Neither distilled model improves the primary log-loss criterion. Paired loss differences versus the original BERT are +0.0058 (descriptive 95% interval −0.0089 to +0.0207) and +0.0084 (−0.0083 to +0.0250), respectively. Both intervals include zero and do not establish a difference. The mixture refit after adding the students also fails to improve the prior best verified mixture. [All epochs and source breakdowns](../reports/human-distillation-development.json).


## Calibration variants and mixture cost

Calibration variants of one predictor share the same underlying forward pass. New exports copy each predictor artifact once, cache its raw scores within a prediction batch, and apply each calibrator separately. Batch caches cannot carry predictions into a different request. Aggregate prediction weights still identify each calibrated component.

A grouped fixed-mixture comparison regularizes the total weight assigned to each underlying predictor, so creating an additional calibration variant does not independently change that predictor's regularization target. The legacy per-component penalty remains reported. In the partial Mistral/compact comparison, the stronger grouped penalty increases macro-F1 to 0.883 but worsens log loss to 0.129, versus 0.873 and 0.124 for the selected legacy mixture. The predeclared log-loss criterion therefore retains the incumbent. [Grouped comparison](../reports/human-grouped-mixture-audit.json).

## Small mixtures and alternative exports

A predefined small-mixture family checks whether two predictors can retain the full ensemble's performance at lower inference cost. It compares pairs from the original-calibration lexical model, a compact model, frozen Mistral and frozen Qwen. The lexical/compact representatives are selected by human gate log loss; each pair's nonnegative convex weights are fitted on the same gate partition. Human validation still selects among the resulting models. This adds exploratory comparisons and does not create an independent test.

An additional efficiency comparison removes underlying predictors with less than 1% total weight from the fixed mixture with penalty 0.1, then renormalizes the remaining weights without refitting. All calibration variants of a predictor are grouped before applying that cutoff. A variant with little individual weight therefore survives if its shared predictor exceeds the cutoff. The original mixture remains available. The 1% rule is fixed before evaluating this candidate; a smaller model is not assumed to preserve accuracy, and predictor counts are not substitutes for measured latency.

A named candidate can be exported with `python -m possum.improvement_inference export RUN --candidate NAME`. Each export must pass reload parity. Exporting a smaller alternative does not replace a better verified incumbent; the bundle ledger retains both for later accuracy/latency comparisons.

For the compact alternative, `python -m possum.improvement_inference export RUN --best-compact` selects the single compact predictor with the lowest human validation log loss across all completed comparisons for the same frozen dataset. Calibration variants remain eligible; multiple-model averages and noncompact experts are excluded from this one-model category. Archived candidates can win, so adding later models does not silently discard an earlier compact result. The export still requires actual reload parity and does not replace a better overall incumbent. This selection uses development data only and must precede the final holdout freeze.

## Qwen prompt-format comparison

Frozen Qwen3.5-9B scores substantially below Mistral with the shared raw-completion prompt: human macro-F1 0.604 and log loss 0.265. Its local tokenizer artifact provides a conversation template with an explicit `enable_thinking=False` assistant prefix. A separate, versioned `qwen_chat` comparison places the same classification task in the user turn and forces the assistant completion `This statement reflects a` followed by ` left` or ` right`. Both continuations are single tokens in the pinned tokenizer. No reasoning or completion is sampled.

The completed chat-template comparison reaches human macro-F1 0.764 and log loss 0.191, versus 0.604 and 0.265 for the raw-completion baseline. Both use unchanged calibration/gate/validation partitions and separately fitted human calibration. The paired loss difference is −0.0743 (descriptive 95% interval −0.1075 to −0.0401). This supports the format change for this frozen Qwen checkpoint on this development split; it is not an independent generalization test.

The new fixed mixture reaches human log loss 0.12288 versus 0.12385 for the previous winner, with unchanged macro-F1 0.873. Its paired difference interval spans −0.0091 to +0.0061: the small aggregate gain is uncertain. The mixture now includes Qwen and several BERT variants, so latency matters before interpreting the added complexity as useful. Actual reload parity passed with maximum probability error 2.38e−10. [Source/topic results and weights](../reports/human-qwen-chat-development.json).

Default export selection compares the selected winners from the current comparison and all archived comparisons with the same frozen dataset hash. Adding experts can change fitted mixture weights and make a new winner worse, so an earlier winner remains eligible even if it had not yet been exported. Historical export reads the historical score arrays and router parameters, recomputes decision rules without modifying the archive, and records the comparison hash in the bundle. Reload parity is still required. Explicit named-candidate exports use the current comparison unless `--comparison-dir` selects an archive. This preserves the development selection criterion; it does not make repeated validation selection an independent test.

## Human-only Mistral adapter comparison

A separate Mistral LoRA trial starts from the pinned frozen base and uses only the 1,224 MITweet fit passages. Synthetic data and all calibration, gate, validation and holdout rows are excluded from adapter fitting. The recipe uses rank 8 on query/value projections in the last four layers, scale 16, zero dropout, Adam at 0.00001, four-example gradient accumulation, gradient clipping at norm 1 and class-balanced binary next-token cross-entropy. Two epochs are scored and calibrated separately on the unchanged development partitions, retaining the frozen expert as a comparison.

Recovery checkpoints contain adapter tensors, optimizer state, deterministic epoch/order position, dataset/recipe hashes and cumulative training time. Before sustained fitting, an actual accumulated update must reproduce after optimizer/adapter reload. Zero dropout and deterministic ordering are explicit assumptions of that check. Completed epoch adapters are immutable and scoring verifies their weight hashes. The first epoch is complete: 306 accumulated updates, 425,984 trainable parameters and 19.4 minutes of fitting. Its 1,973 development scores took 27.0 minutes. Human validation macro-F1/log loss improve from the frozen model’s 0.852/0.145 to 0.865/0.122; the paired log-loss change is −0.0231 with a descriptive 95% interval of −0.0366 to −0.0108. Relative to the larger verified mixture, the difference is only −0.00055 with an interval spanning zero. Synthetic macro-F1 rises from 0.977 to 0.985 while log loss worsens from 0.062 to 0.112. These are interim results: epoch two and final adapter export verification are pending. [Matched source/topic comparison](../reports/human-mistral-adapter-interim.json).

## Final topic-stress evaluation

The final workflow first exports and verifies the selected model and, optionally, a compact alternative. `python -m possum.final_evaluation freeze RUN --compact-bundle BUNDLE` then fixes their manifests, decision rules and the dataset/holdout hashes without reading holdout passages. Normally all planned comparisons must be complete; a partial-run freeze requires an explicit reason. A freeze cannot be replaced, and subsequent training, calibration/scoring development, mixture fitting and export selection are disabled for that run.

`python -m possum.final_evaluation evaluate RUN` checks the frozen artifacts, durably records exposure before opening holdout text, and saves predictions every ten examples. Interrupted scoring resumes only with the same selection and hashes. Completed evaluation returns its existing report instead of predicting again. Thresholds remain those fitted on the gate partition. Reports include per-source/topic results, class counts and end-to-end batch timing; those timings include loading and integrity checks and are not warm-service latency. No candidate may be retuned or selected using these holdout results.

The final report also includes confusion matrices and per-class recall with Wilson 95% intervals, both overall and by source/topic. Absent-class recall and two-class balanced accuracy are explicitly undefined for single-class groups. Overall log loss, Brier score, macro-F1 and balanced accuracy receive descriptive 95% intervals from 2,000 class-stratified row-bootstrap draws with fixed seed 42 and observed class counts. These intervals condition on the evaluated passages: they do not account for topic/author dependence, label uncertainty or population shift. Three held-out topic groups are insufficient to establish broad topic generalization. Reporting methods are specified before opening the holdout.

## Initialization sensitivity

The selected mixed-data BERT recipe was repeated with seeds 43 and 44 alongside seed 42. Their best development checkpoints have human log loss 0.2004, 0.2290 and 0.2196, respectively (mean 0.2163; sample standard deviation 0.0146). Predictions disagree at the 0.5 decision threshold on 28 of 408 human validation passages. These are three initializations on one split, with checkpoint selection on that same validation set, rather than independent data replications.

An equal-weight average of the three calibrated predictions is now included as a fixed candidate without additional weight fitting. It reaches human log loss 0.1972 and macro-F1 0.748. Its paired loss difference against seed 42 is −0.0033, with a descriptive 95% row-bootstrap interval of −0.0181 to +0.0115. This small, uncertain change does not establish superiority; the Mistral-based mixture remains the strongest verified development candidate. [Seed results](../reports/human-seed-sensitivity.json).

## Inference timing and batch consistency

`python -m possum.inference_benchmark RUN --bundle BUNDLE --device mps` measures a verified export on the shortest, median-length and longest calibration passage in each source/class cell. It checks single-request predictions against a batch of the same passages and requires maximum probability error below 0.0002. It uses the shared pipeline lock to avoid competing with training, checks artifact hashes, and caches completed measurements by bundle manifest and sampling plan. It never opens the topic holdout.

The report separates the first request, later single-request latency, and batch time per passage. Each call includes integrity checks, model loading, prediction and cleanup; later calls can benefit from filesystem and runtime caches. These measurements describe the reloadable predictor, not a persistent warm inference service. They do not constitute another accuracy evaluation. Actual timing results will be reported after measurement.

## Runtime provenance

The [runtime snapshot](../reports/human-development-runtime.json) records Python, macOS, architecture and installed versions of the libraries used for the development run. Compact models use PyTorch MPS and quantized causal models use MLX. Model revisions and dataset hashes are recorded separately in training and export metadata. This package snapshot documents the environment that produced the results; it is not a tested fresh-install recipe or a portable dependency lock.

## Additional encoder comparison

A queued comparison uses [Microsoft DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base), pinned to revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`. The MIT-licensed model has 12 layers, 768 hidden dimensions, an 86-million-parameter backbone and 98 million embedding parameters. Its different pretraining recipe makes it a useful encoder comparison; published general-language results do not establish political-slant performance.

Two full fine-tunes use the same frozen partitions and calibrated evaluation as BERT: mixed-source and human-only fitting, four epochs, learning rate 0.00002 and seed 42. The plan records the dataset hash, model revision and recipes before training. Tokenizer-only preflight passed without truncation on all development partitions. Actual MPS training compatibility, accuracy and speed remain to be measured. Training shares the existing pipeline lock, keeps optimizer/RNG recovery checkpoints and has a bounded compute budget that reserves time for final evaluation. A planned incomplete comparison must be disclosed if the final selection is frozen before it finishes.


## Development topic coverage

Abortion and Women’s Right account for 342 of 408 human validation passages (83.8%; 314 left and 28 right). The other eight topics contribute 66 passages (52 left and 14 right). For the verified mixture after the Qwen chat comparison, macro-F1/log loss are 0.861/0.121 on the two largest topics and 0.898/0.131 on the remainder. The smaller subset does not show an aggregate collapse, but its size and label coverage cannot establish broad political-text performance. These slices were defined by topic counts, not errors, and they do not alter model selection. They are diagnostics on development data already used for selection. [Per-partition topic counts and model snapshot](../reports/human-topic-coverage.json).
