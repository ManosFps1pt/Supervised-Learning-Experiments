# The Defected Nuts Summary

## Source

- AICC/problem page: https://aicc-official.org/contests
- Kaggle competition: https://www.kaggle.com/competitions/the-defected-nuts-aicc-round-1-2
- AICC editorial: https://aicc-official.org/solutions/round-1/the-defected-nuts
- Platform: Kaggle
- Contest: AICC Round 1
- Difficulty: hard

## Local Artifacts

- Original notebook: `source/baseline-the-defected-nuts-aicc-round-1.ipynb`
- Working notebook copy: `notebooks/the-defected-nuts_work.ipynb`
- Reference solution: `notes/reference_solution.md`
- Data directory: `data/`
- Dataset status: downloaded and extracted
- Submission script: `submission_script.bat`

## Task Shape

- Task type: industrial anomaly segmentation
- Inputs: clean train images and defect test images
- Outputs/submission format: `submission.csv` with encoded anomaly masks
- Metric: AUPRO

## IOAI Syllabus Coverage

- Primary coverage: Image Segmentation; Pre-trained Vision Encoders; Image Augmentation; Model Evaluation Metrics
- Secondary coverage: Data Processing; Autoencoders as optional related method family
- Competition pattern: segmentation mask contract, encoded submission validation, anomaly framing, high-resolution image handling

## Notebook Data-Flow Check

- Installs packages: no
- Downloads/prepares dataset: no
- Manual download needed: no
- Evidence: dataset exists under `data/data/`; baseline copy exists under `notebooks/`

## Next Action

Use for baseline-recognition drill after easier classification/NLP tasks; focus on identifying the first legal artifact and submission format.

