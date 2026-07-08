# Too Easy Fairy

- Source: https://judge.nitro-ai.org/competitions/roai-2025/lot-2-2026/2/view
- Competition: ROAI Selection Camp - CPU Practical Round
- Local status: public statement mirrored; train data, test data, sample output, and starter kit are listed on Nitro but not mirrored here.
- CEOAI tags: `5(a)`, `5(c)`, `3(c)`
- Priority: very high

## Task Type

Computer vision / one-shot semantic segmentation from frozen foundation-model features.

For each image, the task provides:

- a feature tensor of shape `(16, 16, 384)`, extracted from DINOv2;
- one foreground seed coordinate;
- one background seed coordinate.

The goal is to output a binary foreground/background mask over the `16 x 16 = 256` patch grid.

## Dataset Contract

Expected public files on Nitro:

- `features/`: `.npy` files with shape `(16, 16, 384)`;
- `seeds.csv`: columns such as `image_id`, `fg_x`, `fg_y`, `bg_x`, `bg_y`.

Seed coordinates are in `224 x 224` image space, so a baseline must convert them to the `16 x 16` patch grid.

## Output Contract

Submit one CSV with columns:

```text
subtaskID,datapointID,answer
```

- `subtaskID`: `1`.
- `datapointID`: image id.
- `answer`: comma-separated binary predictions for the 256 patches.

## Scoring

Scoring is based on Dice score. The starter kit contains the exact evaluator. Dice below `0.40` receives only a small base score; Dice at or above `0.78` receives full points.

## Baseline Route

1. Load one feature map and confirm shape `(16, 16, 384)`.
2. Convert foreground/background seed pixel coordinates into patch coordinates.
3. Compare every patch feature to the foreground and background seed features with cosine similarity.
4. Predict foreground where similarity to the foreground seed exceeds similarity to the background seed.
5. Save the mask CSV and validate each answer has exactly 256 comma-separated `0`/`1` values.

## Completion Evidence

Save a binary mask CSV and a short note with:

- feature tensor shape check;
- seed-to-patch conversion rule;
- local Dice score if labels/evaluator are available;
- at least one inspected mask or patch-grid sanity check.
