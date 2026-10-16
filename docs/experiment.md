# Short experiment protocol

## Question and scope
Predict left/right viewpoint in English U.S. political text of 20–150 whitespace-separated words. A disagreement with a politician or a quotation need not express the author's own view. Exclude neutral, ambiguous and nonpolitical examples only when the dataset supplies reliable labels or human review; do not pretend party metadata performs that filtering.

## Splits
Reserve approximately 20% of connected source groups for a final test. Group by original document and author, including shared identities across datasets; join near-duplicate clusters before splitting. Remaining 80% is development. Within development allocate disjoint expert-fit, calibration, gate-fit and validation groups. This held-out stacking design avoids fitting a gate on in-sample expert scores; it trades some efficiency for an easily audited pilot. Group proportions will not be exactly 80/20 at row level.

Pilot: cap at 800 accepted rows from development, with both labels and sufficient groups. Never inspect final-test metrics to choose components. Primary evaluation should ultimately use human-verified, unseen text; party-platform and synthetic pilot metrics are diagnostic only. IBC passages shorter than 20 words are excluded in the strict experiment and can be reported in a separately identified sensitivity run.

## Comparisons
TF–IDF logistic regression; frozen BERT with a fitted logistic head; BGE-small with logistic head; BGE-small neural classifier; GLiNER zero-shot scores; optional OpenJev and causal Mistral/Qwen scores. Base BERT has no trained political classification head: a randomly initialized classification layer is not a valid expert.

Learn per-component temperature on calibration groups only, then freeze it. Gate weights are softmax(MLP(BGE-384)); combine calibrated class probabilities, not unrelated raw logit scales. Fit fixed and input-dependent mixtures on gate-fit groups only. Refit each leave-one-out mixture on the same gate-fit partition. Compare on validation with paired group bootstrap intervals, macro-F1, balanced accuracy, log loss, Brier score, ECE, latency and memory. Include error correlation and per-source metrics.

## Decision rule, specified before results
A component is a candidate to drop if its leave-one-out ablation does not worsen validation log loss and it adds material runtime, or if it is consistently below the lexical baseline. Low standalone accuracy alone is insufficient: complementary errors may help. A neural gate merits expansion only if it improves log loss over both the strongest single and fixed mixture across seeds, with a paired group bootstrap interval excluding zero, without a material macro-F1 regression. Otherwise retain the simpler model. A small or weak-label pilot may be inconclusive; report that rather than selecting a winner from noise.

## Causal audit
Use a fixed prefix ending `This statement reflects a` and candidate continuations ` left` / ` right`. Read full-vocabulary logits directly; for multi-token candidates sum teacher-forced conditional log probabilities. Binary softmax is conditional on this candidate set. Record label-token IDs, raw scores, prompt and temperature. Compare alternate prompt wording only within development. Keep quoted text delimited and use an explicit U.S. frame. Do not use generation temperature or top-k logprobs as substitutes for calibrated full candidate scores.

## Later work
Grouped cross-fitting can replace the conservative held-out stacking design. Fine-tune BERT only inside training folds, regenerate out-of-fold scores, recalibrate, and rerun the same ablations. Domain/source holdouts and human annotation are required before a claim about viewpoint generalization.
