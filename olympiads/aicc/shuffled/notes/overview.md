# Shuffled Overview

## Source

- Kaggle competition: https://www.kaggle.com/competitions/shuffled-aicc-round-9
- Baseline notebook: https://www.kaggle.com/code/antoningorokva/baseline-shuffled-aicc-round-9
- Platform: Kaggle
- Contest: AICC Round 9

## Task Statement

A CLIP ViT-B/16 model has had both positional embedding tables row-shuffled:

- 196 rows for image patches in a 14 by 14 grid
- 77 rows for text sequence positions

Six anchor rows reveal their true original positions. Predict the original position for every shuffled vision and text row.

## Data

Kaggle lists:

```text
clip/                  CLIP ViT-B/16 with shuffled positional tables
anchors.csv            6 known true positions
data/                  100 matched image-caption pairs for local checking
sample_submission.csv
```

Useful access pattern from the task page:

```python
from transformers import CLIPModel

model = CLIPModel.from_pretrained('/kaggle/input/competitions/shuffled-aicc-round-9/clip')
Vp = model.vision_model.embeddings.position_embedding.weight.data[1:]
Tp = model.text_model.embeddings.position_embedding.weight.data
```

`Vp` has shape `(196, 768)` and `Tp` has shape `(77, 512)`.

## Evaluation

Metric: exact-position accuracy over 267 scored rows. The 6 anchors are not scored. Kaggle page reports baseline solution `0.02` and reference solution `1.00`.

## Submission Format

Submit `submission.csv`:

```csv
row_id,position
vision_0,57
vision_1,3
text_0,44
```

Rows may be in any order.

## Import Status

Baseline downloaded. Dataset downloaded and extracted under `data/`.
