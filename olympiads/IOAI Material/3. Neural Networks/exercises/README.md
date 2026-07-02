# Exercises: 3. Neural Networks

Goal: build a competition-useful neural-network workflow, not a full deep
learning course. After this lesson, you should be able to set up a small MLP,
choose the right output/loss pair, debug shapes, read training curves, and run a
simple interpretability check.

Source material is in `../sources/`.

Main sources:

- `Neural Networks.pptx`: MLP concepts, activations, loss, perceptron update,
  gradient descent, learning rate, batch normalization, dropout, early stopping,
  weight decay, adversarial attacks.
- `NNs_Regression.ipynb`: noisy `sin(x)` regression with a PyTorch MLP.
- `NNs Classification.ipynb`: two-moons binary classification, decision
  boundary, and learning-rate schedule.
- `NNs_Structure.ipynb`: multi-label text model, hidden activations, linear vs
  non-linear probes, and feature interactions.
- `train.jsonl`, `test.jsonl`, `feature_names.json`, `model.pt`: data and model
  assets for the probing exercise.

## Core Path

Total target: **2.5-3 hours**.

1. `01_manual_forward_backprop.md` - 25-35 minutes.
2. `02_regression_mlp_baseline.md` - 35-45 minutes.
3. `03_two_moons_classifier.md` - 40-50 minutes.
4. `04_multilabel_probe_puzzle.md` - 45-55 minutes.

If time is tight, do exercises 1, 3, and 4 first. Exercise 2 is useful, but
classification/probing are higher yield for IOAI-style tasks.

## Optional Stretch

- `05_regularization_robustness.md` - 30-45 minutes if the core path is done.

## What To Save

Create your own work as a notebook or a `.py` file with `#%%` cells. Do not save
solutions in this folder unless you explicitly decide to archive them later.

For each exercise, save only:

- the final metric or observation,
- one screenshot/plot if it helped you debug,
- one reusable contest reflex.

## Stop Condition

Stop Lesson 3 when you can do these without looking at a worked solution:

- choose `MSELoss` vs `BCEWithLogitsLoss` and explain the output shape,
- overfit a tiny batch to debug a PyTorch model,
- compare train vs validation/test behavior,
- explain why a learning rate can be too small or too large,
- identify when a hidden feature is linearly available vs tangled.

No solution code should be generated here unless explicitly requested.

