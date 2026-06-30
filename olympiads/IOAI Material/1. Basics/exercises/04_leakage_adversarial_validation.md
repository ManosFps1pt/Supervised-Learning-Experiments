# Problem Set 04: Leakage and Adversarial Validation

## Source

Use `../sources/L01.ipynb`, section 3 on data leakage and adversarial validation.

## Concepts Covered

- Detecting leakage from suspicious validation scores.
- Understanding how preprocessing can leak information.
- Separating train, validation, and test logic.
- Adversarial validation for train/test distribution shift.
- Interpreting high AUC in a train-versus-test classifier.

## Problems

### 1. Suspicious Score Investigation

You train a quick classifier and get a near-perfect validation score on noisy-looking data. List the first five leakage checks you would run before trusting the result.

Self-check:

- At least two checks should inspect columns or IDs.
- At least one check should inspect the split procedure.
- At least one check should inspect preprocessing.

Hints:

1. Ask whether any feature was created using the target.
2. Ask whether duplicate or near-duplicate samples cross the split.
3. Ask whether preprocessing was fit before splitting.

### 2. Leakage Column Hunt

Given a table description with features such as `sample_id`, `timestamp`, `fold`, `target_mean_by_user`, `post_event_status`, and `image_width`, mark which columns are suspicious and explain why.

Self-check:

- You should distinguish "always leakage" from "needs investigation".
- You should explain how each suspicious column could encode the answer.

Hints:

1. Anything computed after the event is dangerous.
2. IDs can leak if they encode source, target, or duplicates.
3. Aggregates are safe only if computed inside the training fold.

### 3. Adversarial Validation

Create a binary label that marks rows as `train` or `test`. Train a simple model to predict this label from features.

Self-check:

- If the adversarial model performs near random, train/test distributions look similar under those features.
- If it performs well, inspect the most informative features.

Hints:

1. Do not include the original target.
2. Interpret the adversarial model as a distribution-shift detector.
3. High adversarial performance does not automatically tell you the fix; it tells you where to inspect.

### 4. Fix Plan

For a detected shift or leakage risk, propose a validation plan that would be harder to fool.

Self-check:

- The plan should name a splitter or grouping principle.
- The plan should say which preprocessing must happen inside each fold.

Hints:

1. Grouped data needs grouped validation.
2. Time-like data often needs time-aware validation.
3. Fold-safe preprocessing usually belongs in a pipeline.

### Stretch

Write a one-page "do not trust this score yet" checklist for future contests.
