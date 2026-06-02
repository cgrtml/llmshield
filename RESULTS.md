# Evaluation

This is the full evaluation behind the numbers in the README. Everything here is
reproducible from the scripts in `eval/`; the model is trained from public datasets,
not shipped pre-baked, so a reviewer can re-run and check.

## Data

Three public datasets, merged and de-duplicated:

| Source | Examples | Notes |
|---|---:|---|
| `deepset/prompt-injections` | 662 | injection vs benign |
| `jackhhao/jailbreak-classification` | 1,306 | jailbreak vs benign |
| `xTRam1/safe-guard-prompt-injection` | ~3,948 | held back as the OOD set |

Labels are binary (1 = attack, 0 = benign). Exact-duplicate texts are removed before
splitting, so no prompt appears in both train and test.

## Method

- **Features:** each prompt is embedded with VoyageAI (`voyage-3`, 1024-dim). The
  embedding *is* the feature vector — an earlier version used a handful of
  hand-engineered features and a similarity-to-known-attacks score, which was much weaker.
- **Model:** a soft decision tree (`neural-trees`), compared against logistic
  regression and a scikit-learn decision tree.
- **Threshold:** tuned on a validation split to hold recall ≥ 95% (security-first),
  then frozen and measured on the test split.
- **Significance:** Alpaydın's combined 5×2cv F-test and McNemar's test (both from
  `neural-trees`).

## Headline numbers

Combined real data, 60/20/20 train/val/test, threshold set on validation:

| Model | Recall | FPR | F1 |
|---|---:|---:|---:|
| **Soft decision tree** | **96.1%** | **0.3%** | **0.978** |
| Logistic regression | 95.7% | 3.1% | 0.960 |

5-fold cross-validation (soft tree, default threshold) — to check the result holds
across splits rather than on one lucky one:

```
recall 95.5% ± 0.8   FPR 2.5% ± 1.3   F1 0.963 ± 0.010
```

The ~1% standard deviation is the point: it's stable.

## Out-of-distribution

The honest test of generalization. A model trained only on
`deepset` + `jackhhao` was evaluated on `xTRam1`, which it never saw:

```
recall 87.6%   FPR 10.9%   F1 0.882
```

It degrades from 0.97 → 0.88 but does not collapse — there's real transferable signal,
not memorization. The jump in false positives (1% → 11%) is the weak spot and the main
reason more diverse training data helps.

## Sanity checks

These exist because the first version of this project fooled itself, and the checks
are how it got caught.

- **Leakage:** 0 duplicate prompts across splits.
- **Trivial baselines (5-fold F1):** majority-class 0.00, length-only 0.68 — both well
  below the real models (~0.96), so the model isn't just exploiting length.
- **Artifact ablation:** on an early *synthetic* dataset, punctuation + casing features
  alone reached F1 0.96 — i.e. the model was reading how the data was generated, not the
  attack. On real data the same ablation drops to F1 0.49, confirming the real data is clean.
- **Significance:** soft tree vs logistic regression, 5×2cv F-test, p = 0.015.

## vs an existing model

Against ProtectAI's `deberta-v3-base-prompt-injection-v2`, on our held-out set:

| Model | Recall | FPR | F1 |
|---|---:|---:|---:|
| this project | 95.1% | 2.4% | 0.961 |
| ProtectAI deberta (default) | 70.9% | 1.0% | 0.824 |

**Caveat, and it's a big one:** this is our distribution, which our model trained on
and theirs did not. It's a home-field result, not evidence of being better in general —
a fair comparison needs a neutral set both models are blind to. ProtectAI is tuned more
conservatively (higher precision, lower recall).

## What would make this stronger

- A fourth, genuinely unseen dataset to re-measure OOD for the current (larger) model.
- A fine-tuned encoder baseline (needs a GPU) to compare against the embedding+tree approach.
- Adversarial loop: collect the misses, add them to training, repeat.
