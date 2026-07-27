# Essay Gap Overview

## Source

- Kaggle: https://www.kaggle.com/competitions/essay-gap-aicc-round-2
- AICC editorial: https://aicc-official.org/solutions/round-2/essay-gap
- Baseline notebook: https://www.kaggle.com/code/nikolatesla13/baseline-essay-gap-aicc-round-2

## Task Statement

For each essay with a missing middle sentence, choose which of four candidate options best fits the surrounding `before` and `after` context.

## Data

`train.csv` columns:

- `sampleID`
- `before`
- `after`
- `opt_0`, `opt_1`, `opt_2`, `opt_3`
- `label`

`test.csv` has the same input columns without `label`.

## Evaluation

Macro F1 over classes `0..3`.

## Submission Format

```csv
sampleID,answer
100,0
101,1
```

## Import Status

Dataset downloaded and extracted. Baseline and reference artifacts downloaded.

