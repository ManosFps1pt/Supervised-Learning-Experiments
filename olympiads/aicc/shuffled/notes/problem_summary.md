# Shuffled Summary

## Source

- Kaggle competition: [Shuffled](https://www.kaggle.com/competitions/shuffled-aicc-round-9)
- Baseline notebook: [Baseline - Shuffled | AICC Round 9](https://www.kaggle.com/code/antoningorokva/baseline-shuffled-aicc-round-9)
- Platform: Kaggle
- Contest: AICC Round 9
- Difficulty: unknown

## Local Artifacts

- Original notebook: `source/baseline-shuffled-aicc-round-9.ipynb`
- Working notebook copy: `notebooks/shuffled_work.ipynb`
- Data directory: `data/`
- Dataset status: downloaded and extracted
- Submission script: `submission_script.bat`

## Task Shape

- Task type: CV + NLP / CLIP positional embedding recovery
- Inputs: shuffled CLIP positional embeddings, anchors, optional matched image-caption pairs
- Outputs: `row_id,position`
- Metric: exact-position accuracy over 267 scored rows
- Baseline score reported by task page: `0.02`

## IOAI Syllabus Coverage

- Primary coverage: Computer Vision Pre-trained Vision Encoders; NLP / vision-text encoders; Data Embeddings; Tensor Manipulation
- Secondary coverage: Transformers, model-output interpretation, NumPy/PyTorch basics
- Competition pattern: pretrained-model inspection, tensor-shape validation, exact submission contract, use of anchor constraints

## Notebook Data-Flow Check

- Installs packages: no
- Downloads/prepares dataset: no
- Manual download needed: no, already downloaded locally
- Evidence: baseline reads `/kaggle/input/competitions/shuffled-aicc-round-9/clip`, `anchors.csv`, `data/pairs.csv`, and image files

## Next Action

Accept Kaggle rules, download data, then run the baseline unchanged and validate `submission.csv` covers all 273 `vision_*` and `text_*` rows exactly once.
