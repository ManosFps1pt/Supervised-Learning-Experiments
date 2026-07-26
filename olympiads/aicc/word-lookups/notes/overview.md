# Word Lookups Overview

## Source

- Kaggle competition: https://www.kaggle.com/competitions/aicc-round-9-word-lookups
- Baseline notebook: https://www.kaggle.com/code/kinggior/baseline-word-lookups-aicc-round-9
- Platform: Kaggle
- Contest: AICC Round 9

## Task Statement

Build a model for Mandarin Chinese word segmentation. The input is an unsegmented sequence of Chinese characters. The output is one BMES tag per character:

- `B`: beginning of a multi-character word
- `M`: middle of a multi-character word
- `E`: end of a multi-character word
- `S`: single-character word

The output sequence must have exactly the same length as the input character sequence.

Important restriction: no pretrained models, pretrained embeddings, external dictionaries, or manually labeled external data.

## Data

Kaggle lists:

- `train.csv`
- `test.csv`

Both have:

- `ID`, unique row identifier
- `chars`, a Chinese sentence represented as a stringified Python list of individual characters; parse with `ast.literal_eval`

Do not change the ID column.

## Evaluation

Metric: Boundary F1 score. The evaluator converts BMES tags into word-boundary positions. `E` and `S` create word boundaries; `B` and `M` indicate that the word continues. Higher is better.

## Submission Format

Submit a CSV with columns:

```csv
id,bio_tags
0,"['B','E','S']"
1,"['S','B','M','E']"
```

The `bio_tags` value must be a string representation of a Python list.

## Import Status

Baseline downloaded. Dataset downloaded and extracted under `data/`.
