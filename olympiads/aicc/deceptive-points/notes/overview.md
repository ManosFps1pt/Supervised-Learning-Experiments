# Deceptive Points Overview

## Source

- AICC/problem URL: https://aicc-official.org/solutions/round-0/deceptive-points
- Kaggle URL: https://www.kaggle.com/competitions/deceptive-points-aicc-round-0
- Platform: Kaggle
- Contest: AICC Round 0, October 2025

## Task Statement

A math teacher created a tabular dataset to show the relationship between study effort and exam scores. The true relationship is mostly linear: higher effort should generally lead to higher expected scores.

Some students corrupted part of the training data by adding conflicting examples that make the relationship look misleading. Build a model that predicts the teacher's true expected score from the provided features, using only the noisy training data.

The test set is teacher-only data, so the goal is to learn the valid underlying relationship while avoiding the corrupted/deceptive training entries.

## Input Format

`train.csv` contains:

- `feature1`
- `feature2`
- `feature3`
- `feature4`
- `target`

`test.csv` contains:

- `ID`
- `feature1`
- `feature2`
- `feature3`
- `feature4`

The hidden `solution.csv` contains the teacher-only target values and is not available during competition solving.

## Evaluation

Submissions are evaluated with mean squared error between predicted values and hidden teacher-only target values. Lower MSE is better.

## Submission Format

Create `submission.csv` with:

- `ID`: copied from `test.csv`
- `Target`: predicted numerical value

Example shape:

```csv
ID,Target
0,74.2
1,81.6
```

## Contest Restrictions

- Do not use the hidden solution file for training or fitting.
- Any model family is allowed, including linear models, tree ensembles, gradient boosting, and neural networks.
- You may use all four provided features and any preprocessing you choose.
