# Sprint 04: Multi-Label Probe Puzzle

## Source

Use:

- `../sources/NNs_Structure.ipynb`,
- `../sources/train.jsonl`,
- `../sources/test.jsonl`,
- `../sources/feature_names.json`,
- `../sources/model.pt`.

## Time Box

Target: **45-55 minutes**.

This is the most IOAI-like Lesson 3 exercise: inspect an existing model and
turn hidden activations into evidence.

## Goal

Given a model that predicts 8 independent yes/no sentence features, identify
which feature is not linearly represented at a hidden layer and explain the
interaction pattern.

## Context

The labels are independent binary features:

- `number`,
- `question`,
- `color`,
- `food`,
- `sentiment`,
- `country`,
- `person`,
- `body_part`.

The source notebook uses a sentence encoder, a trained MLP head, and hidden
activations from a 64-dimensional layer. Most labels are easy for a linear
probe. One is not.

## Task A: Data and Output Sanity

Before probing, inspect:

- number of train/test examples,
- feature names and label matrix shape,
- positive rate for each feature,
- whether the output should use sigmoid or softmax.

Write one sentence explaining why the 8 outputs are not required to sum to 1.

## Task B: Linear vs Non-Linear Probe

At the chosen hidden layer, compare for each label:

- a linear probe,
- a small non-linear probe,
- AUC or another threshold-independent metric.

Produce a table:

```text
feature   linear_auc   nonlinear_auc   gap
```

Identify the feature with the largest non-linear gap.

## Task C: Explain the Geometry

For the hard feature, test at least two hypotheses:

- adding squared features helps,
- splitting by another label makes a linear direction work,
- the direction changes between subsets.

Write a short explanation of the representation in plain language.

## Required Self-Checks

- You compare train-fitted probes on test data.
- AUC is computed from scores/probabilities, not hard labels.
- You do not treat the 8-label problem as single-label softmax
  classification.
- You can name the hard feature and the label it interacts with most strongly.
- Your explanation distinguishes "not present" from "present but non-linearly
  tangled".

## Hints

1. Start by reproducing the source notebook's shapes before changing anything.
2. Cache embeddings or activations if you run the encoder.
3. A below-chance linear AUC can mean the feature is encoded in a symmetric or
   interaction-heavy way, not that the network knows nothing.
4. If a split makes a feature linearly easy in both subsets, compare the
   learned directions with cosine similarity.

## Stop Condition

Stop when you can explain the result as a contest answer: what you tested, what
you found, and why it supports your conclusion.
