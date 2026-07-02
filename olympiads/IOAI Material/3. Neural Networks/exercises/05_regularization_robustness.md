# Stretch 05: Regularization and Robustness

## Source

Use `../sources/Neural Networks.pptx`, plus either
`../sources/NNs_Regression.ipynb` or `../sources/NNs Classification.ipynb`.

## Time Box

Target: **30-45 minutes**.

Only do this after the core path. It is useful, but less urgent than getting
the baseline and probing workflow correct.

## Goal

See how regularization and small input perturbations change generalization and
confidence.

## Choose One Track

### Track A: Regularization

Use the two-moons classifier or sine regressor. Compare baseline training
against two of:

- dropout,
- weight decay,
- early stopping,
- batch normalization.

Keep the train/test split fixed and report:

- final train loss,
- final validation/test metric,
- whether the gap between train and validation/test improved.

### Track B: Simple Adversarial Perturbation

Use the two-moons classifier. Choose correctly classified test points near the
decision boundary and add small input perturbations.

Report:

- original probability and predicted class,
- perturbed probability and predicted class,
- perturbation size,
- whether the decision changed.

You do not need a full Fast Gradient Method implementation unless you already
finished Track A.

## Required Self-Checks

- Dropout is active during training and inactive during evaluation.
- Early stopping uses validation behavior, not test-set tuning.
- Weight decay is compared against the same optimizer and split.
- Perturbations are small enough that the input still looks like the same
  two-moons point.

## Hints

1. Regularization helps only if there is a real overfitting gap to reduce.
2. Batch normalization usually goes after a linear layer and before activation.
3. For adversarial-style tests, start near the boundary; far-away points may be
   too confident to flip with a small perturbation.

## Stop Condition

Stop when you can state one practical rule, such as when you would add weight
decay or why decision-boundary points are fragile.
