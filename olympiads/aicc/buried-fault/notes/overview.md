# Buried Fault Overview

## Source

- Kaggle competition: https://www.kaggle.com/competitions/buried-fault-aicc-round-9
- Baseline notebook: https://www.kaggle.com/code/antoningorokva/baseline-buried-fault-aicc-round-9
- Platform: Kaggle
- Contest: AICC Round 9

## Task Statement

Each recording is six vibration-sensor channels sampled at 64 Hz for 32 seconds, giving 2048 time steps per channel. Every recording contains exactly one fault event. Training labels give only the fault type, not the event start/end location.

For each test recording, predict:

- fault type label in `0..5`
- event start index
- event end index

The test machines and sites are unseen in training.

## Data

Kaggle lists:

```text
train.npy              float32, shape (2400, 6, 2048), contains NaN
train_meta.csv         recording_id, machine_id, site_id, label
test.npy               float32, shape (1800, 6, 2048), contains NaN
test_meta.csv          recording_id, machine_id, site_id
sample_submission.csv  recording_id, label, start, end
```

`train.npy[i]` corresponds to row `i` of `train_meta.csv`; same for test files.

## Evaluation

Metric:

```text
score = 0.5 * macro_F1 + 0.5 * mean_IoU
```

IoU is zero whenever the predicted fault type is wrong. Kaggle page reports baseline solution `0.10` and reference solution `0.70`.

## Submission Format

Submit `submission.csv`:

```csv
recording_id,label,start,end
test_00000,3,812,948
test_00001,0,1503,1622
```

Constraints: `label` is `0..5`, `0 <= start < end <= 2048`.

## Import Status

Baseline downloaded. Dataset downloaded and extracted under `data/`.
