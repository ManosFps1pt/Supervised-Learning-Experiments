# Find Brain Tumors Overview

## Source

- Kaggle: https://www.kaggle.com/competitions/aicc-round-0-brain-tumor
- AICC editorial: https://aicc-official.org/solutions/round-0/find-brain-tumors
- Baseline notebook: https://www.kaggle.com/code/nikolatesla13/baseline-brain-tumor-aicc-round-0

## Task Statement

Classify brain CT images into 4 classes: no tumor plus 3 tumor types. The training set has roughly 2% labeled examples. Test images must receive class predictions.

Restrictions from the task page: no manual labeling, no pretrained models except ResNet18, notebook runtime at most 20 minutes, fits on a P100 GPU.

## Data

- `train.csv`: `image_id`, `label`
- `train/`: training images
- `test/`: test images

## Evaluation

Macro F1.

## Submission Format

```csv
ID,prediction
0664,label
1269,label
```

## Import Status

Dataset downloaded and extracted. Baseline and reference artifacts downloaded.

