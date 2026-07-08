# Trace Twins

- Source: https://judge.nitro-ai.org/competitions/ceoai/ceoai-2026-practice-2/2/view
- Competition: EUROAI (CEOAI) 2026 - Practice Round 2
- Local status: public statement mirrored; train data, test data, starter kit, and pre-judging script are listed on Nitro but not mirrored here.
- CEOAI tags: `4(a)`, `4(b)`, `2(d)`
- Priority: very high

## Task Type

Sequence similarity / pair scoring for malware sandbox transcripts. The goal is to rank whether two transcript windows came from the same original program.

## Dataset Contract

Public statement lists `public_traces.zip`, containing:

```text
public_traces.csv
```

Columns:

- `program_id`: unique program ID;
- `category`: malware category such as Adware or Trojan;
- `tokens`: full transcript as a space-separated command sequence.

Each transcript is split into windows of exactly 200 commands. Positive pairs are windows from the same original transcript. Negative pairs are windows from different programs; some are same-category doppelgangers.

## Required Interface

Submit one `submission.pkl` containing a cloudpickle-serialized `Submission` object:

```python
class Submission:
    def score_A(self, windows, pairs):
        ...

    def score_B(self, windows, pairs):
        ...
```

- `windows`: list of windows, each a list of 200 commands.
- `pairs`: list of `(i, j)` index pairs into `windows`.
- Return one float score per pair, same order as `pairs`.
- Higher score means more likely same program.

Part A uses real command names. Part B independently scrambles command names in each window, so token identity cannot be trusted across windows.

The cloud will not retrain the solution. If using a trained model, it must already be stored inside `submission.pkl`. File size limit is 50 MB.

## Scoring

Both parts use ROC-AUC:

- Part A: maximum 50 points, full score at about AUC `0.84`.
- Part B: maximum 50 points, full score at about AUC `0.78`.

## Baseline Route

1. Generate windows and labeled pairs from public traces for validation.
2. For Part A, score token overlap, n-gram overlap, or TF-IDF cosine similarity.
3. For Part B, use token-position, repetition, frequency-rank, and run-length style features that survive command renaming.
4. Validate ROC-AUC separately for A and B.
5. Serialize the final `Submission` object with `cloudpickle`.

## Completion Evidence

Save `submission.pkl` with `Submission.score_A` and `Submission.score_B`, plus ROC-AUC validation notes for both parts.
