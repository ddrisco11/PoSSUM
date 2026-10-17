# Passage-level label review

The target is ideology expressed by the passage itself. Party affiliation, a policy-category code, patriotic language, or a missing referent is not sufficient evidence of a left/right stance. Neutral, ambiguous, nonpolitical and context-dependent passages need separate review outcomes rather than a forced binary label.

## Manifesto exclusion

New adapter runs default to `data/assembled-v2-no-manifesto`. This derived corpus quarantines all 297 Manifesto-bearing development examples (54 fit and 243 validation). It preserves all remaining role assignments and the original protected inventory byte for byte. Calibration and gate files are unchanged. Existing adapters and metrics still reflect the historical training corpus; they have not been retrained or relabeled. Source-specific evaluation scores against policy mappings measure agreement with those mappings, not verified standalone ideology.

`scripts/exclude_manifesto.py` creates the revision without overwriting the parent and runs the independent assembly checker. New runs reject Manifesto-bearing development inputs. Historical run snapshots remain immutable for reproducibility. The exclusion does not establish a causal explanation for the human-text regression; tweets dominate fitting and also use proxy labels.

## Local review samples

`scripts/prepare_dataset_review.py` prepares 50 deterministic examples from each of the six locally available text sources. Selection uses SHA-256 ranks of a fixed seed, pool name and normalized content identity, with no hand selection or class balancing. Main pools are the normalized candidates, except Manifesto uses historical admitted development and Political Bias uses raw records because none passed admission. An additional 50 admitted fit tweets and 50 raw Manifesto fragments distinguish actual fitting inputs from broader source quality.

Local review files live under ignored `data/review-20261017/`. Source excerpts and human review notes are not public artifacts. CSV columns retain original text, assigned label, supervision basis, word count, previous role, next-run status, provenance, and blank review fields. Use `review_label` values such as left, right, neutral, ambiguous, nonpolitical or needs_context. Use `self_contained_stance` to distinguish an explicit viewpoint from dependence on surrounding text. Existing labels are hypotheses for review, not prefilled human judgments.

Protected content IDs and known test-connected document/author groups are excluded from sampling. The platform source-wide placeholder is not a document identity: known protected overlaps are removed, but missing document metadata prevents a complete document-level guarantee. These samples are inspected development material, never a new final test. The short Political Bias records and raw Manifesto fragments are deliberately displayed even when they fail the 20-word rule.

## Available but unused material

- Party Platform Topics: 8,535 candidate passages, no overnight fitting. Missing document/year identities and cross-source label conflicts keep them quarantined. A 300-example subset was used in earlier diagnostics, so the source is not wholly untouched.
- GPT-4 Political Bias: 657 local raw rows, of which 201 are neutral and all 456 nonneutral rows fall below 20 words. None entered model fitting or evaluation. A short-text protocol could reconsider them, but they remain synthetic.
- Political Tweets: 114,183 normalized candidates. The initial stance/policy screen retained 11,850 and rejected 102,333; joint assembly admitted 11,016 to development, including 6,398 for adapter fitting. Excluded tweets are not automatically useful training data.
- IBC: only 97 eligible public-sample passages; used for diagnostics, never training. Full-corpus text and book/author identities are unavailable locally.
- GPT-4 Political Ideologies: 2,400 development statements and 800 protected holdout statements. The holdout is intentionally not available for fitting or this audit.
- Qbias and SCCC: no local raw text collections. There are no excerpts to sample and no hidden pool ready for training.

The sampling report contains aggregate counts and methodology only. All eight sources are listed in the local inventory CSV, including the two with no acquired text.
