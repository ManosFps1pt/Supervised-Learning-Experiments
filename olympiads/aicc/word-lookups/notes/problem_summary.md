# Word Lookups Summary

## Source

- Kaggle competition: [Word Lookups](https://www.kaggle.com/competitions/aicc-round-9-word-lookups)
- Baseline notebook: [Baseline - Word Lookups | AICC Round 9](https://www.kaggle.com/code/kinggior/baseline-word-lookups-aicc-round-9)
- Platform: Kaggle
- Contest: AICC Round 9
- Difficulty: unknown

## Local Artifacts

- Original notebook: `source/baseline-word-lookups-aicc-round-9.ipynb`
- Working notebook copy: `notebooks/word-lookups_work.ipynb`
- Data directory: `data/`
- Dataset status: downloaded and extracted
- Submission script: `submission_script.bat`

## Task Shape

- Task type: NLP sequence tagging / word segmentation
- Inputs: `ID` plus `chars`, a stringified list of Mandarin characters
- Outputs: `id,bio_tags`
- Metric: Boundary F1
- Restrictions: no pretrained models, pretrained embeddings, external dictionaries, or manually labeled external data

## IOAI Syllabus Coverage

- Primary coverage: NLP Text Classification / sequence labeling; Language Modeling; Data Processing tokenization/vocabulary building; Model Evaluation Metrics
- Secondary coverage: Neural Networks embeddings if using a learned local character model; scikit-learn if using n-gram or rule-based baselines
- Competition pattern: submission-format validation, sequence length contract, no-pretrained-model constraint, metric-aware error analysis

## Notebook Data-Flow Check

- Installs packages: no
- Downloads/prepares dataset: no
- Manual download needed: no, already downloaded locally
- Evidence: baseline reads `/kaggle/input/competitions/aicc-round-9-word-lookups/train.csv` and `test.csv`

## Next Action

Accept Kaggle rules, download data, then first run the baseline notebook unchanged and validate that every `bio_tags` list length equals the corresponding `chars` length.
