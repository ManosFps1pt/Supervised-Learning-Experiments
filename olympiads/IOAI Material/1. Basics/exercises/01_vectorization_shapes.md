# Sprint 01: NumPy Shapes and Vectorization

## Source

Use `../sources/L01.ipynb`, sections 1, 1a, 1b, pairwise distance, and NumPy Shape Gym.

## Time Box

Target: **35-45 minutes**.

Do not explore every NumPy trick. Finish this when you can explain each output
shape without running the cell.

## One Dense Drill

Create a random matrix `X` with shape `(12, 5)`. Treat rows as samples and
columns as features.

Produce:

- a column-centered version,
- a row-normalized version,
- a version where all negative values are replaced by zero,
- the row indices where the first feature is positive.

Then create another matrix `M` with shape `(8, 4)` and compute the full `(8, 8)`
matrix of squared Euclidean distances between rows.

## Required Self-Checks

- `X_centered.mean(axis=0)` is close to zero.
- `np.linalg.norm(X_row_normalized, axis=1)` is close to one for nonzero rows.
- The ReLU-like matrix has no negative values.
- The first-feature result contains row indices, not full rows.
- The distance matrix has shape `(8, 8)`, diagonal close to zero, and is symmetric.

## Hints

1. Inspect shapes before and after each operation.
2. Use `axis=0` for column statistics and `axis=1` for row statistics.
3. Use `keepdims=True` when a denominator must broadcast across columns.
4. For distances, start from `||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i dot x_j`.
5. If a result is wrong, print only shapes first. Do not debug values before shapes.

## Stop Condition

Stop when the self-checks pass and you can say what each axis means. Skip
batched `einsum` for now unless a later neural-network task needs it.
