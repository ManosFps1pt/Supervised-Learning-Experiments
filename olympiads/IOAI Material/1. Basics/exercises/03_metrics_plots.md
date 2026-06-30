# Sprint 03: Metrics, Validation, and Leakage

## Source

Use `../sources/L01.ipynb`, section 2.5, diagnostic plots, leakage, adversarial validation, and cross-validation sections.

## Time Box

Target: **45-60 minutes**.

This exercise replaces the separate metrics, leakage, and CV drills. The goal is
to avoid trusting bad scores under time pressure.

## One Dense Drill

For each scenario, choose the metric, validation split, and first leakage check:

- predicting whether a rare event happens,
- predicting house prices,
- ranking samples by risk,
- classifying balanced labels,
- multiple rows per same user/player/image source,
- time-ordered events.

Then take one binary-classification toy example with true labels and predicted
probabilities. Evaluate thresholds such as `0.2`, `0.5`, and `0.8`.

Finally, write a five-item "do not trust this score yet" checklist.

## Required Self-Checks

- Your answer should mention what the metric rewards and what it ignores.
- At least one scenario should reject plain accuracy.
- Accuracy and F1 should not necessarily prefer the same threshold.
- You should be able to explain one false positive and one false negative tradeoff.
- Your validation split names what must not cross folds.
- Your leakage checklist includes columns/IDs, split procedure, duplicates, preprocessing, and target-derived features.

## Hints

1. Choose the metric before training.
2. Grouped data needs grouped validation.
3. Time-like data often needs time-aware validation.
4. Any preprocessing fitted on all rows before splitting is suspicious.
5. A plot is useful only if it answers a question about target, outliers, missingness, or shift.

## Stop Condition

Stop when you have a compact metric card:

- target type,
- metric,
- split strategy,
- threshold rule if relevant,
- top leakage trap.
