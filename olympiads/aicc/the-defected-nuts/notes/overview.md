# The Defected Nuts Overview

## Source

- Kaggle: https://www.kaggle.com/competitions/the-defected-nuts-aicc-round-1-2
- AICC editorial: https://aicc-official.org/solutions/round-1/the-defected-nuts
- Baseline notebook: https://www.kaggle.com/code/antoningorokva/baseline-the-defected-nuts-aicc-round-1

## Task Statement

Produce anomaly masks for hazelnut images. Training data contains clean images; test images may contain defects. The submitted artifact is a CSV of encoded pixel masks.

Restriction from corpus/task listing: only ImageNet ResNet18 is permitted as pretrained backbone.

## Data

- `data/train/`: clean training images
- `data/test/`: defect test images
- Kaggle-provided baseline notebook is also included in the competition data archive

## Evaluation

AUPRO over anomaly masks.

## Submission Format

Submit `submission.csv` with Base85-encoded masks in the format expected by the Kaggle baseline/task page.

## Import Status

Dataset downloaded and extracted. Baseline and reference artifacts downloaded.

