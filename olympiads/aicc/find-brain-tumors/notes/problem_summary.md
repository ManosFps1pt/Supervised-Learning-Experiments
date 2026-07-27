# Find Brain Tumors Summary

## Source

- AICC/problem page: https://aicc-official.org/contests
- Kaggle competition: https://www.kaggle.com/competitions/aicc-round-0-brain-tumor
- AICC editorial: https://aicc-official.org/solutions/round-0/find-brain-tumors
- Platform: Kaggle
- Contest: AICC Round 0
- Difficulty: medium

## Local Artifacts

- Original notebook: `source/baseline-brain-tumor-aicc-round-0.ipynb`
- Working notebook copy: `notebooks/find-brain-tumors_work.ipynb`
- Reference solution: `notes/reference_solution.md`
- Data directory: `data/`
- Dataset status: downloaded and extracted
- Submission script: `submission_script.bat`

## Task Shape

- Task type: CV image classification with scarce labels
- Inputs: CT image files plus `train.csv`
- Outputs/submission format: `ID,prediction`
- Metric: macro F1

## IOAI Syllabus Coverage

- Primary coverage: Image Classification; Pre-trained Vision Encoders; Image Augmentation; Model Finetuning
- Secondary coverage: PyTorch Basics; Data Processing; Model Evaluation Metrics
- Competition pattern: image loading, label joins, scarce-label baseline, macro-F1 discipline, submission CSV validation

## Notebook Data-Flow Check

- Installs packages: no
- Downloads/prepares dataset: no
- Manual download needed: no
- Evidence: dataset exists under `data/`; baseline copy exists under `notebooks/`

## Next Action

Use for baseline-recognition drill first; run only after writing your own four-line baseline idea.

