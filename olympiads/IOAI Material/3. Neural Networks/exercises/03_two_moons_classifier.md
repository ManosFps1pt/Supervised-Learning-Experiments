# Sprint 03: Two-Moons Classifier and Learning Rate

## Source

Use `../sources/NNs Classification.ipynb` and the learning-rate slides in
`../sources/Neural Networks.pptx`.

## Time Box

Target: **40-50 minutes**.

This is the highest-yield coding exercise in Lesson 3. Keep the experiment
small and compare only a few learning-rate choices.

## Goal

Train a binary classifier on non-linear two-moons data, choose the correct
loss/output setup, and explain how learning rate changes convergence.

## Task A: Clean Baseline

Build a PyTorch classifier for `make_moons` with:

- train/test split,
- input shape `(N, 2)`,
- target shape `(N, 1)`,
- final layer outputting one logit,
- `BCEWithLogitsLoss`,
- thresholded sigmoid probabilities for accuracy.

Produce:

- train loss curve,
- final test accuracy,
- decision-boundary plot on test data,
- printed examples of logits, probabilities, and predicted labels for 5 samples.

## Task B: Learning-Rate Comparison

Run the same model setup with three learning-rate strategies:

- too small,
- reasonable,
- too large or scheduled down from high to low.

Use the same data split. Compare:

- final test accuracy,
- final training loss,
- whether the loss curve is slow, stable, unstable, or oscillating.

## Required Self-Checks

- You do not apply `sigmoid` before `BCEWithLogitsLoss`.
- Accuracy uses probabilities or logits after training, not raw loss values.
- Decision-boundary grid has shape `(grid_points, 2)`.
- The model can reach high test accuracy on the source-style setup.
- You can explain why a scheduled high learning rate may improve convergence
  but can also destabilize training.

## Hints

1. Use a fixed random seed while comparing learning rates.
2. Change one variable at a time.
3. If accuracy stays near 0.5, inspect target shape and thresholding first.
4. If loss is `nan`, lower the learning rate before changing the model.

## Stop Condition

Stop when you can show one good boundary plot and explain which learning-rate
run you would trust in a contest.
