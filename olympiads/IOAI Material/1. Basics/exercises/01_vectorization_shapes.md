# Problem Set 01: Vectorization and Shapes

## Source

Use `../sources/L01.ipynb`, sections 1, 1a, 1b, the pairwise-distance exercise, loop-vs-vectorized timing, and NumPy Shape Gym.

## Concepts Covered

- Replacing Python loops with vectorized NumPy operations.
- Broadcasting with `keepdims`, `[:, None]`, and `[None, :]`.
- Boolean and fancy indexing.
- Matrix multiplication and `einsum`.
- Pairwise squared distances.
- Axis choice, reshape, concatenate, and `argmax`.
- Timing loop-based code versus vectorized code.

## Problems

### 1. Normalize Without Loops

Create a random matrix with shape `(12, 5)`. Produce:

- a column-centered version,
- a row-normalized version,
- a version where all negative values are replaced by zero,
- the row indices where the first feature is positive.

Self-check:

- The column means of the centered matrix should be close to zero.
- Every nonzero row of the normalized matrix should have norm close to one.
- The ReLU-like matrix should have no negative values.

Hints:

1. Inspect shapes before and after each operation.
2. Use `axis=0` for column statistics and `axis=1` for row statistics.
3. Use `keepdims=True` when a denominator must broadcast across columns.

### 2. Batched Matrix Products

Make two arrays with shapes `(7, 4, 6)` and `(7, 6, 3)`. Compute the batch-wise matrix product so the result has shape `(7, 4, 3)`.

Self-check:

- Compare the first batch result against ordinary matrix multiplication on batch `0`.
- Confirm that no explicit loop is used in the final version.

Hints:

1. The notebook uses `einsum` for this exact pattern.
2. Write down the meaning of each dimension before writing the expression.
3. If the output shape is wrong, your index labels probably preserve or sum over the wrong axis.

### 3. Pairwise Squared Distances

Given a matrix `M` of shape `(n, d)`, compute the full `(n, n)` matrix of squared Euclidean distances between rows.

Self-check:

- The diagonal should be close to zero.
- The matrix should be symmetric.
- All entries should be nonnegative, allowing tiny numerical error.

Hints:

1. Start from `||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i dot x_j`.
2. Compute all row squared norms as a vector of shape `(n,)`.
3. Use broadcasting to combine the norm vector with itself.

### 4. Shape Gym

Create an `(8, 4)` feature matrix and a weight vector with shape `(4,)`. Compute scores, binary predictions, class counts, and the index of the largest score.

Self-check:

- The score vector should have shape `(8,)`.
- The prediction vector should contain only `0` and `1`.
- The class counts should sum to `8`.

Hints:

1. Use matrix-vector multiplication for scores.
2. Convert booleans to integers only after the comparison is correct.
3. Think about whether `argmax` is over samples or over features.

### Stretch

Time a loop implementation and a vectorized implementation of one task above. Record the speed ratio and explain why vectorization matters in leaderboard-style experimentation.
