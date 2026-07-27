# Oriented Ship Overview

## Source

- Kaggle: https://www.kaggle.com/competitions/oriented-ship-aicc-round-7
- Baseline notebook: https://www.kaggle.com/code/nikolasgegenava/baseline-oriented-ships-aicc-round-7
- AICC solution status: coming soon

## Task Statement

Detect ships in aerial imagery with rotated bounding boxes `(cx, cy, w, h, theta)`.

Restrictions from the corpus/task listing: standard ImageNet backbones are allowed, but maritime data and oriented-detection pretraining are not.

## Data

- `Oriented/images/train/`
- `Oriented/images/val/`
- `Oriented/labels/`

## Evaluation

mAP@0.5 with rotated IoU.

## Submission Format

Submit each image's confidence-ranked rotated boxes in the format expected by the Kaggle baseline/task page.

## Import Status

Dataset downloaded and extracted. Baseline downloaded. Reference solution is not yet published by AICC.

