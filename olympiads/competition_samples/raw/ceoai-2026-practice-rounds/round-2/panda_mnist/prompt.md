# Panda MNIST

- Source: https://judge.nitro-ai.org/competitions/ceoai/ceoai-2026-practice-2/1/view
- Competition: EUROAI (CEOAI) 2026 - Practice Round 2
- Local status: public statement mirrored; train data, test data, starter kit, and pre-judging script are listed on Nitro but not mirrored here.
- CEOAI tags: `5(a)`, `3(b)`, `3(c)`
- Priority: very high

## Task Type

Computer vision digit classification under scanner/domain shift, with a model-size penalty and TorchScript submission.

## Dataset Contract

Subtask 1 uses `train_data.zip/subtask1/`:

- `scanner1_X.npy`, `scanner1_y.npy`;
- `scanner2_X.npy`, `scanner2_y.npy`;
- `scanner3_X.npy`, `scanner3_y.npy`.

Images have shape `(N, 1, 28, 28)` with uint8 pixels in `[0, 255]`.

Subtask 2 uses `train_data.zip/subtask2/` with scanners `1..8`. Images have shape `(N, 3, 28, 28)`. Scanners 1-3 are the subtask 1 grayscale signal triplicated to three channels.

## Submission Contract

Submit `submission.zip` containing two TorchScript models:

```text
submission.zip
├── model_sub1.pt
└── model_sub2.pt
```

- `model_sub1.pt`: `forward()` accepts `(N, 1, 28, 28)` float32 and returns `(N, 10)` raw logits.
- `model_sub2.pt`: `forward()` accepts `(N, 3, 28, 28)` float32 and returns `(N, 10)` raw logits.

Do not apply softmax in `forward()`.

## Scoring

Both subtasks use macro-averaged accuracy across scanners with a parameter-count penalty:

- Subtask 1: maximum 30 points.
- Subtask 2: maximum 70 points.
- Parameter count is computed with `sum(p.numel() for p in model.parameters())`.

## Baseline Route

1. Normalize uint8 images to float32.
2. Train a small CNN or linear/MLP baseline.
3. Track accuracy per scanner, not only global accuracy.
4. Keep parameter count small enough to avoid losing score.
5. Export scripted models and build `submission.zip`.

## Completion Evidence

Save `submission.zip` containing `model_sub1.pt` and `model_sub2.pt`, plus parameter-count and per-scanner accuracy notes.
