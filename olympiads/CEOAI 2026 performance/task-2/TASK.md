# Interleaved lines

Aiga was optimizing the throughput of her GPU cluster when a synchronization bug caused two parallel processes to write their outputs into the same memory buffer.

As a result, tokens from two different paragraphs of text (A and B) were randomly interleaved into a single sequence C. The order of tokens within each original paragraph was preserved, but the two paragraphs became mixed together.

For example, the two token sequences

```
A: <I> <like> <to> <go> <out>
B: <Ai> <ga> <often> <trains> <models>
```

could be interleaved as
```
C: <I> <like> <Ai> <to> <go> <ga> <often> <out> <trains> <models>
```
Your task is to recover which tokens came from which paragraph by predicting a binary mask for the mixed sequence (0 for tokens from paragraph A and 1 for tokens from paragraph B). Since the labeling of A and B is symmetric, both
```
0 0 1 0 0 1 1 0 1 1
```
and its inverse
```
1 1 0 1 1 0 0 1 0 0
```
are considered correct answers.

Observe that theoretically, there could be many possible answers to the same test example. Your solution will be evaluated against the correct mask that was produced by randomly interleaving the paragraphs A and B.

### Dataset Description

#### File 1: `train_data.csv`
This dataset contains 5 000 examples, and columns `id`, `c`, and `mask`. Each example represents two tokenized paragraphs that have been interleaved into a single sequence. All text is tokenized using the **Pythia-14M tokenizer**.

```csv
id,c,mask
"0","35529, 13, 359, ...","0, 0, 0, ..."
"1","2484, 261, 783, ...","0, 0, 1, ..."
...
```

The meanings of the variables/columns are the following:

* `id` - unique example identifier (0 - 4 999),
* `c` - the mixed token sequence, created by interleaving two paragraphs,
* `mask` - binary assignment mask for `c`; `0` marks tokens from paragraph A and `1` marks tokens from paragraph B.

#### File 2: `test_data.csv`
This dataset contains 500 examples (id: 5 000 - 5 499) for which correct masks are unknown - there are only columns `id` and `c`. Participants must predict the missing `mask`. The same tokenizer has been used, and the data is similar to `train_data.csv`.

#### File 3: `custom_archive.zip`
This file contains starter code `starter.ipynb`, weights of the `pythia-14m` language model, and a tokenizer in a format that can be loaded using Hugging Face `AutoTokenizer` and `AutoModelForCausalLM`. To unzip it, run:

```
unzip custom_archive.zip
```

### Output Format

The output file (`.csv`) must contain three columns: `subtaskID`(always equals 1), `datapointID`, and `answer`, with one row per test example.

```
subtaskID,datapointID,answer
"1","5000","0, 1, 1, 0, 0, ..., 0"
"1","5001","0, 0, 1, 1, 0, ..., 1"
...
"1","5499","0, 0, 1, 1, 0, ..., 1"
```

### Scoring

Submissions are evaluated by **token-level accuracy** on the hidden test set. The accuracy for a single example is calculated as the **maximum** of two possible matchings of predicted mask (`pred_mask`) and ground truth mask (`gold_mask`):

- `pred_mask` against `gold_mask`,
- `pred_mask` against the inverted mask (0 and 1 flipped).

In other words, label assignment is symmetric, that is, flipping all 0s and 1s does not affect the score.

The final score is the average accuracy over all test examples.

Points are awarded as follows:

| score | points |
| --- | --- |
| score < 0.60 | 5 points |
| score > 0.81 | 100 points |
| otherwise | 5 + round(95 × (accuracy - 0.60) / 0.21) |
