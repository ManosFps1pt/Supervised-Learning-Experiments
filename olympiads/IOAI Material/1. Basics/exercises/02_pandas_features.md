# Sprint 02: Pandas Feature Table

## Source

Use `../sources/L01.ipynb`, section 2, the groupby/transform examples, merge validation, and the raw-table-to-features mini-task.

## Time Box

Target: **50-60 minutes**.

This is the most important Lesson 1 exercise. Data processing will repeat in
almost every later lesson.

## One Dense Drill

Create one small raw table with:

- `sample_id`
- one group column such as `team` or `user_id`
- at least two numeric columns
- one categorical column
- some missing values
- one suspicious outlier

Produce a clean feature table with:

- original `sample_id`,
- missing-value indicators for important numeric columns,
- one outlier flag,
- one ratio or rate feature,
- one group-relative feature using `groupby(...).transform(...)`,
- one group summary table using `groupby(...).agg(...)`,
- one safe merge that uses `validate=...`.

## Required Self-Checks

- The new columns should have the same number of rows as the original table.
- A team-level aggregate should have one row per team.
- A `transform` result should align row-by-row with the original table.
- Feature rows should match raw rows.
- Every engineered feature should have a short reason.
- You can explain whether each operation is row-level, group-level, or dataset-level.

## Hints

1. First inspect shape, dtypes, missing counts, and suspicious ranges.
2. Use `agg` when you want a smaller summary table.
3. Use `transform` when you need a same-length result.
4. Use `value_counts` on keys before merging.
5. Keep target information out of features unless the task explicitly allows it.

## Stop Condition

Stop when you have one model-ready feature table and a short note explaining
each feature. Do not spend time making the table realistic or large.
