# Polarity Summary

## Source

- AICC/problem page: https://aicc-official.org/contests
- Kaggle competition: https://www.kaggle.com/competitions/polarity-aicc-round-7
- Official solution notebook: https://github.com/AI-Community-Contest/solutions/blob/main/round-7/polarity.ipynb
- Platform: Kaggle
- Contest: AICC Round 7
- Difficulty: medium

## Local Artifacts

- Original notebook: `source/baseline-polarity-aicc-round-7.ipynb`
- Working notebook copy: `notebooks/polarity_work.ipynb`
- Reference solution: `notes/reference_solution.md`
- Data directory: `data/`
- Dataset status: downloaded and extracted
- Submission script: `submission_script.bat`

## Task Shape

- Task type: NLP lexical binary classification
- Inputs: word pairs
- Outputs/submission format: `row_id,label`
- Metric: macro F1

## IOAI Syllabus Coverage

- Primary coverage: NLP Text Classification; Pre-trained Text Encoders; Transformers; Data Embeddings
- Secondary coverage: Model Evaluation Metrics; few-shot validation
- Competition pattern: restriction checking, tiny train set, binary macro-F1, pretrained-model input contract

## Notebook Data-Flow Check

- Installs packages: no
- Downloads/prepares dataset: no
- Manual download needed: no
- Evidence: baseline reads Kaggle train/test CSV paths; dataset exists under `data/`

## Next Action

Use for pretrained-text-encoder pattern recognition after Essay Gap.

