# Sprint 04: Fast Baseline Playbook

## Source

Use `../sources/L01.ipynb`, sections on sklearn pipelines, missing data, sanity
checks, and the contest playbook.

## Time Box

Target: **30 minutes once**, then **10-15 minutes reused** at the start of later
lessons.

## Purpose

This is the reusable workflow for any new tabular or small ML task. It should be
short enough to run during competition without thinking.

## Baseline Checklist

Before modeling:

- print shape, dtypes, missing counts, and target distribution,
- identify task type and metric,
- identify ID, group, time, or duplicate columns,
- choose validation split before feature engineering,
- decide which preprocessing must happen inside the fold.

First baseline:

- simple preprocessing only,
- one simple model,
- correct metric,
- fold scores with mean and variability,
- one saved note about the biggest current weakness.

Missing data:

- compare plain imputation against imputation plus missing indicators,
- check whether missingness relates to the target using training labels only,
- keep the same validation folds when comparing strategies.

Debugging:

- if score is too good, suspect leakage first,
- if score is too bad, check shapes, dtypes, target encoding, metric direction, and split,
- if model training fails, overfit a tiny batch or tiny subset before changing architecture.

## Output

Create a one-page personal checklist in your notes or notebook. Every line should
be an action, not vague advice.

Good:

- `print X_train.shape, X_val.shape, y_train distribution`
- `verify metric direction: higher is better or lower is better`
- `check duplicate IDs across train and validation`

Weak:

- `inspect data`
- `check metric`
- `avoid leakage`

## Stop Condition

Stop when the checklist fits on one screen and you would actually use it before
starting a new exercise.
