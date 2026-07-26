# Massive Problem Summary

## Source

- AICC contests page: [AICC contests](https://aicc-official.org/contests)
- Kaggle competition: [Massive Problem](https://www.kaggle.com/competitions/massive-problem-aicc-round-6)
- Platform: Kaggle
- Contest: AICC Round 6, April 2026
- Difficulty: Medium
- Author listed by AICC: Wang Jiayu
- Imported material: public Colab copy of the official Kaggle baseline notebook.

## Local Artifacts

- Original baseline notebook: `source/Massive_Problem_Baseline.ipynb`
- Working baseline notebook copy: `notebooks/Massive_Problem_Baseline_work.ipynb`
- Data directory: `data/`
- Dataset status: downloaded
- Download helper: `download_data.bat`
- Submission script: `submission_script.bat`

## Dataset Status

Dataset status: downloaded.

The Kaggle archive is stored at:

```text
data/massive-problem-aicc-round-6.zip
```

The extracted CSVs are stored at:

```text
data/massive-problem-aicc-round-X/task_data/
```

To re-download from this problem folder, run:

```powershell
kaggle competitions download -c massive-problem-aicc-round-6 -p olympiads\aicc\massive-problem\data
```

Or from this problem folder, run:

```bat
download_data.bat
```

The helper clears the dead proxy variables for that command before calling Kaggle.

Verified extracted files:

- `RNA_seq_patient_0.csv`: 51,548 rows
- `RNA_seq_patient_1.csv`: 7,291 rows
- `RNA_seq_patient_2.csv`: 16,716 rows
- `RNA_seq_patient_3.csv`: 102,135 rows
- `RNA_seq_patient_4.csv`: 59,796 rows
- `RNA_seq_patient_5.csv`: 1,012 rows
- `RNA_seq_patient_6.csv`: 16,043 rows
- `RNA_seq_patient_7.csv`: 3,306 rows
- `RNA_seq_patient_8.csv`: 3,543 rows
- `test.csv`: 7,401 rows

The baseline notebook itself also contains:

```python
! kaggle competitions download -c massive-problem-aicc-round-6
! 7z x massive-problem-aicc-round-6.zip
```

## Task Shape

- Task type: large tabular / biological gene-expression multiclass classification.
- Inputs: nine patient CSVs, each with `Gene1`-`Gene1434`, `batch`, and `label`; one held-out-patient `test.csv` with `Gene1`-`Gene1434` and `batch`.
- Outputs/submission format: `submission.csv` with columns `id` and `label`.
- Metric: macro-averaged F1 score.

## IOAI Syllabus Coverage

- Primary coverage: Foundational Skills & Classical Machine Learning.
- Secondary coverage: NumPy/Pandas data handling, scikit-learn, supervised learning, model evaluation, train/validation splitting, feature processing, PCA, XGBoost/model ensembles, and memory-aware data processing.
- Why this maps to the syllabus: the task requires practical use of pandas dtypes, high-dimensional feature matrices, multiclass classification, macro F1, dimensionality reduction, and validation design under patient/batch shift.

## Notebook Data-Flow Check

- Installs packages: no.
- Downloads/prepares dataset: yes, via Kaggle CLI in the notebook.
- Expects local data: yes, after extraction under `task_data/`.
- Manual download needed: yes until the Kaggle competition is joined.
- Evidence: importer inspection found `kaggle competitions download -c massive-problem-aicc-round-6`, `7z x massive-problem-aicc-round-6.zip`, and `read_csv` calls for `task_data/RNA_seq_patient_{p}.csv` plus `task_data/test.csv`.

## Baseline Notes

The baseline loads all nine training patient files with reduced integer dtypes, concatenates them, runs PCA to 100 components on gene features, appends `batch`, trains `XGBClassifier`, and writes `submission.csv`.

The baseline's random validation split reports about `0.5697` macro F1, but its Kaggle score is `0.2413`. Treat that as evidence that row-random validation is too optimistic for a held-out-patient test set. A better practice route is leave-one-patient-out validation using the `batch` column.

## Next Action

Join the Kaggle competition and download the dataset. Then open `notebooks/Massive_Problem_Baseline_work.ipynb`, change validation from a random row split to patient-aware validation, and only then try model improvements.
