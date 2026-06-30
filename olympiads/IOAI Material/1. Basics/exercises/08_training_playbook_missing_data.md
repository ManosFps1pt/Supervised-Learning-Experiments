# Problem Set 08: Training Playbook and Missing Data

## Source

Use `../sources/L01.ipynb`, sections 7 and 8 on ML training tips, sanity checks, missing data, missingness correlation, strategy comparison, and contest playbook.

## Concepts Covered

- Fast contest baseline workflow.
- Shape, dtype, missing-value, and target checks.
- Train/validation discipline before improvement.
- Missing-data visualization.
- Missingness correlated with target.
- Comparing imputation strategies fairly.
- Practical contest reflexes.

## Problems

### 1. First Ten Minutes Plan

Write the first ten minutes of your workflow for a new tabular ML task.

Self-check:

- The plan should inspect shape, dtypes, missing values, target distribution, and metric.
- The plan should include a simple baseline before advanced ideas.
- The plan should say how you will avoid leakage.

Hints:

1. Baseline first, improvement second.
2. Do not train before knowing the metric.
3. A quick sanity function saves time under pressure.

### 2. Missingness Report

Given a table with missing numeric values, produce a report that answers:

- which columns are missing,
- how much is missing,
- whether missingness is related to the target,
- which missing indicators might be useful features.

Self-check:

- Missing counts alone are not enough.
- At least one check should compare target behavior by missing/not-missing.

Hints:

1. Missingness can be signal.
2. Missingness can also be an artifact of data collection.
3. Do not use test labels to study missingness.

### 3. Strategy Comparison

Compare at least three missing-data strategies under the same validation setup.

Examples:

- simple median or most-frequent imputation,
- imputation plus missing indicators,
- model or pipeline that handles missing values differently.

Self-check:

- Every strategy should use the same folds.
- Preprocessing should be fitted inside each fold.
- The comparison should use the task metric.

Hints:

1. Changing the split while changing the strategy makes the comparison noisy.
2. Missing indicators often help when missingness is informative.
3. More complex imputation is not automatically better.

### 4. Contest Reflex Card

Create a one-page card for yourself with:

- before-training checks,
- baseline checklist,
- validation checklist,
- debugging checklist,
- final-submission checklist.

Self-check:

- The card should be usable during a timed practice round.
- Every item should trigger a concrete action, not vague advice.

Hints:

1. Prefer "print shape and dtype" over "inspect data".
2. Prefer "verify metric direction" over "check metric".
3. Keep it short enough that you would actually use it.

### Stretch

After finishing one full exercise, add one real mistake to the error journal using the repo protocol.
