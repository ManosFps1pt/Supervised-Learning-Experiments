# Essay Gap Summary

## Source

- AICC/problem page: https://aicc-official.org/contests
- Kaggle competition: https://www.kaggle.com/competitions/essay-gap-aicc-round-2
- AICC editorial: https://aicc-official.org/solutions/round-2/essay-gap
- Platform: Kaggle
- Contest: AICC Round 2
- Difficulty: easy

## Local Artifacts

- Original notebook: `source/baseline-essay-gap-aicc-round-2.ipynb`
- Working notebook copy: `notebooks/essay-gap_work.ipynb`
- Reference solution: `notes/reference_solution.md`
- Data directory: `data/`
- Dataset status: downloaded and extracted
- Submission script: `submission_script.bat`

## Task Shape

- Task type: NLP multiple-choice text coherence
- Inputs: `before`, `after`, four options
- Outputs/submission format: `sampleID,answer`
- Metric: macro F1

## IOAI Syllabus Coverage

- Primary coverage: NLP Text Classification; Pre-trained Text Encoders; Transformers; Model Evaluation Metrics
- Secondary coverage: Data Processing; tokenization
- Competition pattern: option expansion, class-label contract, macro-F1 validation, clean CSV output

## Notebook Data-Flow Check

- Installs packages: no
- Downloads/prepares dataset: no
- Manual download needed: no
- Evidence: baseline reads train/test CSV paths; dataset exists under `data/essay-gap/`

## Next Action

Best first drill target among the five because the dataset is small and the task format is clean.

