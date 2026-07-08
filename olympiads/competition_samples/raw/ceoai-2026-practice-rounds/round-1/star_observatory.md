# Star Observatory

- Source: https://judge.nitro-ai.org/competitions/ceoai/ceoai-2026-practice-1/3/view
- Competition: CEOAI 2026 - Practice Round 1
- Local status: public statement mirrored; train data, test data, sample output, and starter kit are listed on Nitro but not mirrored here.
- Related local notebook: `raw/romania-roai-solved/international-contests/ceoai-practice/round-1-2026/star-observatory.ipynb`
- CEOAI tags: `5(a)`, `2(a)`, `3(c)`
- Priority: very high

## Task Type

Computer vision plus regression. Predict star center coordinates and calibrated target flux from star images affected by atmospheric distortion.

## Dataset Contract

Public statement lists:

- `train_images/00000.png` through `00999.png`;
- `train.csv` with `image_id`, `fried_parameter`, `airmass`, `target_flux`;
- `test_images/00000.png` through `00299.png`;
- `test.csv` with `image_id`.

Each image is `128 x 128` and contains one star.

## Subtasks

1. Star Center Prediction: output `(x, y)` coordinates. Scored by Manhattan-distance MAE. Worth 20 points.
2. Flux Prediction: output true flux as a continuous value. Scored by RMSE. Worth 80 points.

Constraints include a 45-minute notebook runtime and no pretrained models.

## Output Contract

Submit one CSV with exactly 600 rows:

```text
subtaskID,datapointID,answer
1,00001.png,"(69.0, 66.0)"
2,00001.png,1542.3
```

- 300 center predictions.
- 300 flux predictions.

## Baseline Route

1. Load images and metadata.
2. For center prediction, use intensity-weighted centroid or brightest-region centroid.
3. For flux prediction, start with image intensity features plus metadata where available.
4. Train a simple regression baseline.
5. Validate exactly 600 output rows and tuple formatting.

## Completion Evidence

Save a 600-row submission CSV with center tuples and flux predictions, plus local MAE/RMSE notes.
