# Error Pattern Review

Updated: 2026-07-11

Journal size: 36 entries total; 27 resolved, 8 open, 1 meta.

## Top Patterns

### 1. Contract errors are the main contest risk

Evidence: repeated `logic`, `shape`, `submission`, `path`, `metric`, and `sklearn` entries across Star Observatory, Trace Twins, Help BOBAI, Classical ML, NLP TF-IDF, and RL gridworld.

Underlying cause: the code often starts from a plausible idea before the exact contract is pinned down: train/test columns, file paths, output row count, feature shapes, label mapping, metric inputs, or submission formatting.

Contest reflex:

```text
Before modeling:
1. print train columns / test columns
2. print input shape / target shape
3. print metric direction and metric inputs
4. print expected output file columns and row count
5. assert saved artifact format after writing to disk
```

Next drill: for the next two tasks, write a validator cell before tuning. It must reload the saved output file and check row count, columns, answer format, and non-empty predictions.

### 2. Tensor and model API contracts are still fragile

Evidence: many PyTorch entries involve dtype mismatch, `[batch, features]` shape, target shape, loss contract, device mismatch, DataLoader-vs-batch confusion, logits-vs-label confusion, and argmax-before-loss.

Underlying cause: the first forward/loss call is sometimes attempted before proving the one-batch contract.

Contest reflex:

```text
Before training:
print(batch_x.shape, batch_x.dtype, batch_x.device)
print(model(batch_x).shape)
print(batch_y.shape, batch_y.dtype, batch_y.device)
print(loss_name)
```

Next drill: every PyTorch practice task must run a one-batch smoke test before the training loop. No DataLoader, optimizer, or epoch loop counts until this passes.

### 3. Feature availability must be checked from actual files, not statements

Evidence: Star Observatory metadata issue, TF-IDF refit issue, sklearn scaling of labels, and Help BOBAI subset evaluation all came from trusting an assumed feature/evaluation contract.

Underlying cause: the natural-language problem statement can be incomplete, mirrored fixtures can differ from full tasks, and intermediate labels/features can change meaning after preprocessing.

Contest reflex:

```text
Features used for train must be constructible for test.
Vectorizers/scalers fit on train only.
Evaluation must cover the full task unless explicitly scoring a subpart.
```

Next drill: add `assert list(X_train.columns) == list(X_test.columns)` for dataframe features, or print matching matrix dimensions for arrays.

### 4. Saved artifacts are not being treated as the source of truth early enough

Evidence: Star Observatory produced a good model but stale/bad `sample_submission.csv`; Hungary regressed to old-class-only labels before being repaired; several notebooks "ran" before satisfying the task contract.

Underlying cause: notebook state and visible cells can look successful while the file on disk is stale, malformed, or generated from an older variable.

Contest reflex:

```text
After saving:
reload the file from disk
print shape
print columns
print head/tail
group/check label or subtask counts
regex-check formatted fields when needed
```

Next drill: the zero-avoidance mock should regenerate and reload every required artifact: Hungary CSV, Star CSV, Trace Twins PKL, Panda MNIST ZIP, and Broken BERT CSV.

## Open Entries To Close

- Two-moons BCE accuracy: convert logits to labels before metric.
- Two-moons decision-boundary plot: reshape grid predictions to mesh shape.
- NLP mini-submission dataframe: build rows/columns directly, do not use `DataFrame.add`.
- RL gridworld goal comparison: use tuple equality or `np.array_equal`.
- MNIST eval loop: pass image batches to the model, not the DataLoader.
- Classical ML LogisticRegression: scale X only; keep y as labels.
- KMeans evaluation: do not use supervised classifier accuracy helper on raw cluster IDs.
- Help BOBAI 7-way wrapper: evaluate full old-vs-new routing contract.

## Current Priority Drill

Do a zero-avoidance mock:

1. Restart kernel or clean run.
2. Regenerate the output artifact.
3. Reload the saved artifact from disk.
4. Validate row count, shape, columns, file members, and answer formatting.
5. Record any failure as a journal entry immediately.

This is higher yield than adding more theory before CEOAI.
