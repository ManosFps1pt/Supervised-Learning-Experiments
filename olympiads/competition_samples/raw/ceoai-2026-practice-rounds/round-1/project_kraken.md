# Project KRAKEN

- Source: https://judge.nitro-ai.org/competitions/ceoai/ceoai-2026-practice-1/2/view
- Competition: CEOAI 2026 - Practice Round 1
- Local status: public statement mirrored; starter kit is listed on Nitro but not mirrored here.
- CEOAI tags: `3(c)`, `5(a)`, `2(a)`
- Priority: very high

## Task Type

Multimodal deep-learning and classical-model workflow with image-like tensors, time series, text/glyph features, and three prediction subtasks.

## Dataset Contract

Public statement lists:

- `train_slices.npy`, `test_slices.npy`: arrays with shape `(N, 3, 128, 128)`.
- `train_echoes.npy`, `test_echoes.npy`: time-series arrays with shape `(N, 1024, 2)`.
- `train_glyphs.csv`, `test_glyphs.csv`: encoded telemetry sequences.
- `train_targets.csv`: target labels/values for all three subtasks.

## Subtasks

1. Geodesic Rectification: predict 10 spline coefficients, scored by MSE. Worth 30 points.
2. Non-Euclidean Topological Classification: classify into `0..49` or anomaly class `-1`, scored by modified macro F1. Worth 35 points.
3. Heisenberg Stability Limit: predict a stability float in `[0, 1]`, scored by RMSE. Worth 35 points.

## Output Contract

Submit one CSV containing predictions for every test datapoint and all three subtasks:

```text
subtaskID,datapointID,answer
1,test_00000,"0.123;0.456;...;0.999"
2,test_00000,-1
3,test_00000,0.884
```

- Subtask 1 answer: 10 floats separated by semicolons.
- Subtask 2 answer: one integer in `0..49` or `-1`.
- Subtask 3 answer: one float in `[0, 1]`.

## Baseline Route

1. Inspect each modality shape and target columns.
2. Start with simple flattened/statistical features per modality.
3. Train separate baseline models per subtask.
4. Validate output formatting before model improvement.
5. Improve only the weakest subtask after the first valid CSV exists.

## Completion Evidence

Save a single submission CSV with all three subtask outputs, plus metric notes for MSE, modified macro F1, and RMSE.
