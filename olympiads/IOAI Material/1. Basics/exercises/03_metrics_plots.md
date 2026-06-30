# Problem Set 03: Metrics and Diagnostic Plots

## Source

Use `../sources/L01.ipynb`, section 2.5 and the Matplotlib/Seaborn diagnostic plots section.

## Concepts Covered

- Choosing a metric before training.
- Accuracy, F1, MAE, MSE, and RMSE.
- Probability thresholds for binary classification.
- Confusion matrices.
- Basic diagnostic plots for feature/target relationships.
- Plotting to find data problems before modeling.

## Problems

### 1. Metric Decision Drill

For each scenario, choose the metric you would optimize and explain why:

- predicting whether a rare event happens,
- predicting house prices,
- ranking samples by risk,
- classifying balanced image labels,
- predicting a count-like target with large outliers.

Self-check:

- Your answer should mention what the metric rewards and what it ignores.
- At least one scenario should reject plain accuracy.

Hints:

1. Think about class imbalance.
2. Think about whether the output is a label, probability, score, or number.
3. Think about whether large errors should be punished strongly.

### 2. Threshold Sweep

Given true binary labels and predicted probabilities, evaluate several thresholds.

Self-check:

- Accuracy and F1 should not necessarily prefer the same threshold.
- You should be able to explain one false positive and one false negative tradeoff.

Hints:

1. Start with thresholds such as `0.2`, `0.5`, and `0.8`.
2. Use the confusion matrix to interpret what changed.
3. Do not tune the threshold on test data.

### 3. Regression Error Inspection

Given true and predicted numeric values, compute MAE, MSE, and RMSE. Then inspect which samples contribute the largest errors.

Self-check:

- RMSE should react more strongly to large errors than MAE.
- The largest-error rows should be inspectable by index or ID.

Hints:

1. Keep per-sample absolute errors, not only the average.
2. Sort by error before drawing conclusions.
3. Look for outliers or impossible target values.

### 4. Diagnostic Plot Task

Make two plots before training a model:

- one distribution plot for a numeric feature,
- one plot comparing a feature against the target or label.

Self-check:

- Each plot should answer a specific question.
- You should write down one possible data issue revealed by the plot.

Hints:

1. A plot without a question is decoration.
2. Look for outliers, separability, missingness, and suspicious clusters.
3. Keep plots simple enough to interpret under time pressure.

### Stretch

Create a short "metric card" for one competition task: target type, metric, validation plan, and one metric-specific trap.
