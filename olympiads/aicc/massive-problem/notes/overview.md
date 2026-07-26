# Massive Problem Overview

## Source

- AICC contests URL: https://aicc-official.org/contests
- Kaggle URL: https://www.kaggle.com/competitions/massive-problem-aicc-round-6
- Platform: Kaggle
- Contest: AICC Round 6, April 2026

## Task Statement

Train a classifier that assigns each cell to one of 12 cell types from RNA-seq gene-expression features.
Each sample has expression levels for `Gene1` through `Gene1434`, plus a `batch` column identifying
which patient the row came from.

The dataset is large enough that memory management matters. The official baseline explicitly sets CSV
dtypes and deletes unused variables to reduce RAM pressure.

## Input Format

The competition provides 10 files:

- `RNA_seq_patient_0.csv` through `RNA_seq_patient_8.csv`: training data from 9 separate patients.
  Each row contains `Gene1` through `Gene1434`, `batch`, and `label`.
- `test.csv`: test data from a separate 10th patient. It contains `Gene1` through `Gene1434` and `batch`,
  but no `label`.

## Evaluation

Submissions are evaluated using macro-averaged F1 score. Higher is better.

The official baseline notebook reports:

- random validation macro F1: about `0.5697`
- Kaggle baseline score: `0.2413`

This gap is a useful warning: random row splits do not match the held-out-patient test setting.

## Submission Format

Create a CSV file named `submission.csv` with two columns:

- `id`: the row index in `test.csv`
- `label`: the predicted cell-type label for that row

Example:

```csv
id,label
0,3
1,9
```

## Contest Restrictions

- Standard AICC no-LLM contest rule.
- No architecture restrictions beyond the no-LLM rule.
- Do not republish the task or competition data without permission from the authors.
