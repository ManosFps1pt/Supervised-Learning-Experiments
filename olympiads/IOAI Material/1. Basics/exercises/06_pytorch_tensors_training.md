# Problem Set 06: PyTorch Tensors and Training

## Source

Use `../sources/L01.ipynb`, the PyTorch Tensor Manipulation Survival Kit, Dataset/DataLoader section, and section 5 on debugging models that do not learn.

## Concepts Covered

- Tensor shape inspection.
- NHWC versus NCHW image layout.
- `permute`, `contiguous`, `view`, and `reshape`.
- `TensorDataset` and `DataLoader`.
- Basic CPU/GPU-aware training loop structure.
- Correct logits/loss pairing.
- Overfitting one batch as a sanity check.
- `model.train()`, `model.eval()`, and `torch.no_grad()`.

## Problems

### 1. Image Layout Conversion

Start with image-like data in shape `(batch, height, width, channels)`. Convert it to the format expected by PyTorch convolution layers.

Self-check:

- The final shape should be `(batch, channels, height, width)`.
- Flattening should preserve the batch dimension.
- You should know when `.contiguous()` is needed.

Hints:

1. Write the meaning of each axis before moving it.
2. `permute` changes the view of dimensions.
3. Some later operations expect contiguous memory.

### 2. Dataset and DataLoader

Build a dataset from feature tensors and label tensors, then iterate through batches.

Self-check:

- Each batch should have matching feature and label batch sizes.
- The final batch may be smaller unless configured otherwise.
- Labels should have the dtype expected by the loss.

Hints:

1. Inspect one batch before training.
2. Classification labels for cross-entropy are class indices, not one-hot vectors.
3. Keep batch shape assumptions explicit.

### 3. Overfit One Batch

Train a small model on a single batch until it can nearly memorize that batch.

Self-check:

- Loss should drop clearly.
- Accuracy on that same batch should rise.
- If it fails, inspect loss choice, learning rate, labels, and model outputs.

Hints:

1. This is a debugging test, not a real validation score.
2. Use raw logits with cross-entropy.
3. Try a simple model before a complicated one.

### 4. Buggy Loop Diagnosis

Given a training loop that does not learn, identify likely bugs before changing the architecture.

Check:

- softmax before cross-entropy,
- too-large learning rate,
- labels on the wrong device,
- missing `zero_grad`,
- missing `backward`,
- missing `step`,
- evaluating without `model.eval()` or `torch.no_grad()`.

Self-check:

- For each suspected bug, explain the symptom it could cause.
- Fixes should be tested one at a time where possible.

Hints:

1. Start with the first five-minute checks from the lesson.
2. Print shapes and dtypes before assuming the model is bad.
3. A model that cannot overfit one batch is usually broken somewhere basic.

### Stretch

Write a "model does not learn" checklist that fits on one screen.
