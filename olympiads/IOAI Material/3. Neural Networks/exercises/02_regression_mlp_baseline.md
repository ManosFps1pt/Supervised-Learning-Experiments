# Sprint 02: Regression MLP Baseline

## Source

Use `../sources/NNs_Regression.ipynb`.

## Time Box

Target: **35-45 minutes**.

Do not tune architecture for a long time. The useful skill is building a clean
baseline and reading whether it generalizes.

## Goal

Train a small PyTorch MLP to approximate noisy `sin(x)` data and verify that the
model learns the pattern without only memorizing noise.

## Task

Create a regression experiment with:

- training inputs sampled from `[-10, 10]`,
- targets `sin(x) + noise`,
- a separate test grid,
- an MLP with input shape `(N, 1)` and output shape `(N, 1)`,
- `MSELoss`,
- Adam or AdamW.

Produce:

- printed tensor shapes for `X_train`, `y_train`, `X_test`, `y_test`,
- a training-loss curve,
- a final test MSE,
- a plot of noisy test points and model predictions,
- one sentence explaining whether the model is underfitting, reasonable, or
  overfitting.

## Required Self-Checks

- Model output shape matches target shape exactly.
- Loss decreases for most of training.
- Test MSE is in the same rough range as the source notebook, not orders of
  magnitude worse.
- The prediction curve is smooth and follows `sin(x)` rather than every noisy
  point.

## Debugging Probes

- Overfit 20 points first if the full run does not learn.
- Print `next(model.parameters()).dtype` and input tensor dtype if training
  fails.
- If the plot is flat, check learning rate, activation placement, and whether
  `optimizer.step()` is called.
- If the model output has shape `(N,)`, fix the target/output shape before
  changing the architecture.

## Stop Condition

Stop when you have a baseline MSE, a plot, and one clear explanation of the
main failure mode if the model is not good.
