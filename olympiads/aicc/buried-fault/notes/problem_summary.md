# Buried Fault Summary

## Source

- Kaggle competition: [Buried Fault](https://www.kaggle.com/competitions/buried-fault-aicc-round-9)
- Baseline notebook: [Baseline - Buried Fault | AICC Round 9](https://www.kaggle.com/code/antoningorokva/baseline-buried-fault-aicc-round-9)
- Platform: Kaggle
- Contest: AICC Round 9
- Difficulty: unknown

## Local Artifacts

- Original notebook: `source/baseline-buried-fault-aicc-round-9.ipynb`
- Working notebook copy: `notebooks/buried-fault_work.ipynb`
- Data directory: `data/`
- Dataset status: downloaded and extracted
- Submission script: `submission_script.bat`

## Task Shape

- Task type: classical ML / time-series sensor fault classification and weakly supervised localization
- Inputs: sensor arrays shaped `(n, 6, 2048)` plus metadata CSVs
- Outputs: `recording_id,label,start,end`
- Metric: `0.5 * macro_F1 + 0.5 * mean_IoU`
- Baseline score reported by task page: `0.10`

## IOAI Syllabus Coverage

- Primary coverage: Supervised Learning; Model Evaluation Metrics; Feature Engineering; Data Processing; time-series / ragged missing-data handling
- Secondary coverage: NumPy/Pandas, scikit-learn, cross-validation, underfitting/overfitting
- Competition pattern: baseline-first modeling, NaN handling, shape validation, train/test machine-site shift, interval-output contract

## Notebook Data-Flow Check

- Installs packages: no
- Downloads/prepares dataset: no
- Manual download needed: no, already downloaded locally
- Evidence: baseline reads `/kaggle/input/competitions/buried-fault-aicc-round-9/train.npy`, `test.npy`, `train_meta.csv`, and `test_meta.csv`

## Next Action

Accept Kaggle rules, download data, then run the baseline unchanged and validate `submission.csv` has 1800 rows with legal integer labels and intervals.
