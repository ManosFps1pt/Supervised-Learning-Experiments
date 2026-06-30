# Problem Set 02: Pandas Features

## Source

Use `../sources/L01.ipynb`, section 2, the groupby/transform examples, merge validation, and the raw-table-to-features mini-task.

## Concepts Covered

- DataFrame inspection.
- `groupby`, `agg`, and `transform`.
- Per-row features based on group statistics.
- Safe joins with `merge(validate=...)`.
- Missing values and outlier-aware feature construction.
- Turning raw tables into model-ready features.

## Problems

### 1. Team-Relative Features

Create or load a small table with player rows, a team column, minutes, shots, and goals. Add features that compare each player to their own team average.

Self-check:

- The new columns should have the same number of rows as the original table.
- A team-level aggregate should have one row per team.
- A `transform` result should align row-by-row with the original table.

Hints:

1. Use `agg` when you want a smaller summary table.
2. Use `transform` when you need a same-length result.
3. Check that no row order assumptions silently enter your logic.

### 2. Merge That Should Fail

Build two small tables where one table has duplicate keys. Try to merge them as if the relationship were one-to-one, then explain what the validation error tells you.

Self-check:

- You should be able to identify which key is duplicated.
- You should be able to state whether the intended relationship is one-to-one, one-to-many, or many-to-one.

Hints:

1. Use `value_counts` on the key before merging.
2. Pick the `validate` argument that matches your intended relationship.
3. A merge that runs is not necessarily a correct merge.

### 3. Raw Table to Features

Given a raw table with `sample_id`, numeric columns, categorical columns, and missing values, produce a clean feature table.

Include:

- missing-value indicators for important numeric columns,
- a simple outlier flag,
- at least one ratio or rate feature,
- a categorical count or frequency feature.

Self-check:

- The final table should preserve `sample_id`.
- Feature rows should match raw rows.
- Every engineered feature should have a short reason.

Hints:

1. First inspect shape, dtypes, missing counts, and suspicious ranges.
2. Build features that could plausibly help a model, not random transformations.
3. Keep target information out of the feature table unless the task explicitly allows it.

### Stretch

Write a short checklist for deciding whether a pandas operation is row-level, group-level, or dataset-level.
