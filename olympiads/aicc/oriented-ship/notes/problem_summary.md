# Oriented Ship Summary

## Source

- AICC/problem page: https://aicc-official.org/contests
- Kaggle competition: https://www.kaggle.com/competitions/oriented-ship-aicc-round-7
- Platform: Kaggle
- Contest: AICC Round 7
- Difficulty: hard

## Local Artifacts

- Original notebook: `source/baseline-oriented-ships-aicc-round-7.ipynb`
- Working notebook copy: `notebooks/oriented-ship_work.ipynb`
- Reference solution: `notes/reference_solution.md`
- Data directory: `data/`
- Dataset status: downloaded and extracted
- Submission script: `submission_script.bat`

## Task Shape

- Task type: oriented object detection
- Inputs: aerial images and rotated-box labels
- Outputs/submission format: confidence-ranked rotated boxes
- Metric: mAP@0.5 with rotated IoU

## IOAI Syllabus Coverage

- Primary coverage: Object Detection; Pre-trained Vision Encoders; Image Augmentation; Model Evaluation Metrics
- Secondary coverage: Tensor Manipulation; coordinate normalization
- Competition pattern: detection dataset contract, rotated geometry, mAP scoring, submission syntax

## Notebook Data-Flow Check

- Installs packages: no
- Downloads/prepares dataset: no
- Manual download needed: no
- Evidence: dataset exists under `data/Oriented/`; baseline copy exists under `notebooks/`

## Next Action

Use as a hard pattern-recognition example. Do not implement first unless the goal is object-detection format exposure.

