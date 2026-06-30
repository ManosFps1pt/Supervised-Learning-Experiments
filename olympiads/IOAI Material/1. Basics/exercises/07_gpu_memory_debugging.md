# Problem Set 07: GPU, Speed, and Memory Debugging

## Source

Use `../sources/L01.ipynb`, section 6 on GPU speed and memory, plus the gradient accumulation and clipping extra section.

## Concepts Covered

- Device selection and tensor placement.
- Hidden GPU-to-CPU synchronization.
- Why repeated `.item()` calls can slow training.
- CUDA memory allocated versus reserved.
- Out-of-memory triage.
- Dataloader throughput.
- Gradient accumulation.
- Gradient clipping.

## Problems

### 1. Device Audit

Inspect a training setup and list every object that must live on the same device.

Self-check:

- Model parameters, input tensors, and target tensors should be accounted for.
- You should know which objects stay on CPU, such as raw pandas DataFrames.

Hints:

1. Device bugs often appear only when moving from CPU to GPU.
2. Check the batch inside the loop, not only the original dataset.
3. Keep device movement explicit.

### 2. Synchronization Trap

Explain why logging a scalar every training step can slow down GPU training. Then propose a safer logging rhythm.

Self-check:

- Your answer should mention GPU-to-CPU synchronization.
- Your plan should still allow you to monitor training.

Hints:

1. `.item()` moves a scalar to Python.
2. Python cannot read GPU results without waiting.
3. Logging every few batches is often enough.

### 3. OOM Triage

You hit an out-of-memory error during training. Rank the first actions you would try.

Options to consider:

- reduce batch size,
- reduce image size or sequence length,
- use gradient accumulation,
- delete unused tensors,
- use `no_grad` during evaluation,
- simplify the model,
- inspect whether validation keeps graphs alive.

Self-check:

- The plan should preserve correctness before chasing speed.
- The plan should distinguish training memory from evaluation memory.

Hints:

1. Batch size is usually the fastest lever.
2. Evaluation should not build gradients.
3. Keeping a list of loss tensors can accidentally keep computation graphs.

### 4. Gradient Accumulation Plan

Design a training plan for when the desired batch size does not fit in memory.

Self-check:

- The effective batch size should be clear.
- Optimizer steps should happen only after the chosen number of micro-batches.
- Loss scaling should be considered.

Hints:

1. Accumulation trades memory for time.
2. Clipping is useful when gradients can explode.
3. Keep the validation loop separate from accumulation logic.

### Stretch

Create a speed-debugging checklist for deciding whether the bottleneck is CPU data loading, GPU compute, logging, or memory pressure.
