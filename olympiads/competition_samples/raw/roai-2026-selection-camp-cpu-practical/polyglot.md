# Polyglot

- Source: https://judge.nitro-ai.org/competitions/roai-2025/lot-2-2026/1/view
- Competition: ROAI Selection Camp - CPU Practical Round
- Local status: public statement mirrored; train data, sample output, and starter kit are listed on Nitro but not mirrored here.
- CEOAI tags: `4(b)`, `4(c)`, `2(b)`
- Priority: very high

## Task Type

NLP embedding-space alignment. Three different transformer encoders represent the same `N = 4000` documents in different L2-normalized embedding spaces.

The row order is shuffled in `M2` and `M3`. Some aligned anchors are given:

- `M2`: first `K1 = 400` rows are aligned with `M1`.
- `M3`: first `K2 = 20` rows are aligned with `M1`.

The goal is to recover the permutation mapping from each row of `M1` to the matching row in `M2` and `M3`.

## Output Contract

Submit one CSV with columns:

```text
subtaskID,datapointID,answer
```

- `subtaskID`: `1` for `M1 -> M2`, `2` for `M1 -> M3`.
- `datapointID`: row index in `M1`, from `0` to `3999`.
- `answer`: predicted row index in the target matrix.

## Scoring

Both subtasks use raw accuracy. Subtask 1 contributes 30 points, subtask 2 contributes 70 points. Accuracy below `0.20` receives only a small base score; accuracy at or above `0.95` receives full points for that subtask.

## Baseline Route

1. Load the matrices and inspect shapes.
2. Use anchor rows to learn or validate a similarity transform/alignment strategy.
3. Compute pairwise cosine similarity or distances after alignment.
4. Produce a one-to-one mapping, not just independent nearest-neighbor guesses if duplicates appear.
5. Save the submission CSV and validate row count: `8000` prediction rows.

## Completion Evidence

Save a submission-like CSV mapping rows for both subtasks, plus a short note with:

- matrix shapes;
- anchor counts used;
- validation accuracy on anchors or held-out anchors;
- whether assignment/permutation constraints were enforced.
