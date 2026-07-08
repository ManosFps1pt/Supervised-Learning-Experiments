# Kazakhstan TST Day 2 Player Clustering Attempt

## Status

Problematic / blocked. Do not keep spending contest-prep time on this unless the official CSVs become available.

Reason: the official Kaggle data cannot currently be obtained from the local environment or the configured Kaggle account.

What is present:

- Local baseline script: `solve_kazakhstan_day2.py`
- Source reference: `../../raw/kazakhstan-tst-day2/batyr-yerdenov-2-2.ipynb`
- Downloader: `../../raw/kazakhstan-tst-day2/download_kaggle_data.py`

What is missing:

- `train.csv`
- `sample_submission.csv`

## Data Checks Run

The repo and common user download/cache locations were searched for:

- `train.csv`
- `sample_submission.csv`
- `tst-day-2.zip`
- `tst-day-2-upsolving.zip`

No matching Kazakhstan Day 2 CSV files were found.

The only similar local archive found was `olympiads/train_data.zip`, but it contains PNG images and no CSV files, so it is not this task.

## Kaggle Download Result

Using the repository virtual environment:

```powershell
.\.venv\Scripts\python.exe olympiads\competition_samples\raw\kazakhstan-tst-day2\download_kaggle_data.py
```

Result:

```text
403 Client Error: Forbidden
```

The same happened for direct Kaggle CLI download with both `tst-day-2` and `tst-day-2-upsolving`.

Interpretation: Kaggle CLI is installed and credentials exist, but this account/session does not currently have download access to the competition data through the API. The likely fix is to open the Kaggle competition page in a browser, join/accept terms, then rerun the downloader.

Contest-prep decision: move to the next queued exercise instead of burning more time on dataset recovery.

## Baseline Route

Once the CSVs are present, run:

```powershell
.\.venv\Scripts\python.exe olympiads\competition_samples\task_attempts\kazakhstan_player_clustering\solve_kazakhstan_day2.py
```

The script looks for both CSV files in:

1. `olympiads/competition_samples/raw/kazakhstan-tst-day2/`
2. `olympiads/competition_samples/task_attempts/kazakhstan_player_clustering/`
3. repository root
4. `olympiads/`

Expected outputs:

- `submission.csv`
- `cluster_count_scores.csv`
- `run_summary.csv`
- `cluster_feature_means.csv`

Baseline method:

- Aggregate player attributes into broad skill features.
- Split goalkeepers from outfield players.
- Impute missing values.
- Standardize features.
- Select outfield cluster count by silhouette score.
- Fit Gaussian Mixture clusters.
- Assign goalkeepers to a separate cluster.
- Save a submission matching `sample_submission.csv`.
