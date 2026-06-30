# Problem Set 05: Pipelines and Cross-Validation

## Source

Use `../sources/L01.ipynb`, section 4 on correct cross-validation, `Pipeline`, `ColumnTransformer`, and splitter choice.

## Concepts Covered

- Train/validation split discipline.
- Fold-safe preprocessing.
- Numeric and categorical preprocessing with `ColumnTransformer`.
- Pipelines for imputation, scaling, encoding, and modeling.
- `KFold`, `StratifiedKFold`, and `GroupKFold`.
- Choosing the validation splitter that matches the data.

## Problems

### 1. Pipeline Blueprint

Design a pipeline for a table with numeric columns, categorical columns, missing values, and a classification target.

Self-check:

- Numeric preprocessing should handle missing values and scaling if needed.
- Categorical preprocessing should handle missing or unknown categories.
- The model should only see transformed features.

Hints:

1. Keep preprocessing inside the pipeline.
2. Use a column transformer when column types need different treatment.
3. The validation fold must not influence fitted preprocessors.

### 2. Splitter Choice Drill

For each dataset, choose the splitter:

- balanced tabular classification,
- imbalanced binary classification,
- multiple rows per player,
- time-ordered events,
- image augmentations derived from the same original image.

Self-check:

- Your choice should explain what must not cross folds.
- At least one answer should use grouping.

Hints:

1. Stratification preserves label proportions.
2. Grouping prevents entity leakage.
3. Time order may matter more than random mixing.

### 3. Manual Leakage Audit

Take a preprocessing plan and mark whether each step is safe outside CV or must be inside CV:

- filling missing values with a global median,
- standard scaling,
- one-hot category fitting,
- dropping constant columns,
- selecting top features by target correlation.

Self-check:

- Any step using feature distributions from all rows is suspicious.
- Any step using the target is especially dangerous.

Hints:

1. Ask whether validation rows helped choose a number, category set, or feature.
2. If yes, the step belongs inside the fold.
3. Feature selection is model training in disguise.

### 4. Cross-Validation Report

Run or design a CV evaluation that reports fold scores, mean, and variability.

Self-check:

- A single mean score is not enough.
- You should inspect whether one fold is much worse than the others.

Hints:

1. Fold variability can reveal shift, leakage, or unstable models.
2. Keep the metric aligned with the task.
3. Do not tune repeatedly on the same validation feedback without tracking decisions.

### Stretch

Create a validation decision tree: random split, stratified split, group split, or time split.
