# Deceptive Points Summary

## Source

- AICC/problem URL: https://aicc-official.org/solutions/round-0/deceptive-points
- AICC contests URL: https://aicc-official.org/contests
- Kaggle URL: https://www.kaggle.com/competitions/deceptive-points-aicc-round-0
- GitHub solution notebook: https://github.com/AI-Community-Contest/solutions/blob/main/round-0/deceptive-points.ipynb
- Platform: Kaggle
- Contest: AICC Round 0, October 2025
- Difficulty: Easy
- Author listed by AICC: Gior
- Imported material: official AICC solution notebook and Kaggle baseline notebook.

## Local Artifacts

- Original notebook: `source/deceptive-points.ipynb`
- Original Kaggle baseline notebook: `source/deceptive_points_baseline.ipynb`
- Working baseline notebook copy: `notebooks/deceptive_points_baseline_work.ipynb`
- Data directory: D:\projects\Supervised-Learning-Experiments\olympiads\aicc\deceptive-points\data
- Dataset status: downloaded

## Task Shape

- Task type: classical ML / tabular regression
- Inputs: `train.csv` with `feature1`, `feature2`, `feature3`, `feature4`, and `target`; `test.csv` with `ID` and the same four feature columns.
- Outputs/submission format: `submission.csv` with columns `ID` and `Target`.
- Metric: mean squared error, confirmed by the downloaded Kaggle baseline notebook. The solution notebook evaluates locally with cross-validated negative mean squared error.

## IOAI Syllabus Coverage

- Primary coverage: Foundational Skills & Classical Machine Learning.
- Secondary coverage: data science fundamentals, model evaluation, feature preprocessing, train/validation splitting, PCA, K-Means, robust regression, and leakage/overfitting checks.
- Why this maps to the syllabus: the task is a small tabular supervised-learning problem using pandas/NumPy, scikit-learn preprocessing, dimensionality reduction, clustering to identify useful/deceptive regions, and regression model validation.

## Notebook Data-Flow Check

- Installs packages: no
- Downloads/prepares dataset: no
- Manual download needed: no
- Evidence: importer inspection found `read_csv` calls for `train.csv` and `test.csv`, no Kaggle API calls, no direct downloads, and no archive extraction inside the solution notebook. Kaggle CLI authentication was configured, and `kaggle competitions download -c deceptive-points-aicc-round-0` downloaded `train.csv`, `test.csv`, and `deceptive_points_baseline.ipynb`.

## Next Action

Start from `notebooks/deceptive_points_baseline_work.ipynb` for practice. The AICC solution notebook is preserved in `source/deceptive-points.ipynb` for reference only.
