from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


ATTEMPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ATTEMPT_DIR.parents[3]
RAW_DIR = REPO_ROOT / "olympiads" / "competition_samples" / "raw" / "kazakhstan-tst-day2"
OUTPUT_DIR = ATTEMPT_DIR

OUTFIELD_FEATURES = [
    "attacking_skill",
    "passing_ability",
    "dribble_mobility",
    "pace",
    "defense_skill",
    "physicality",
    "set_piece_specialist",
    "composure_score",
    "offensive_support",
    "attack_support",
    "defending_positioning",
]

GK_COLS = ["gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes"]


def find_input_files() -> tuple[Path, Path]:
    candidates = [
        RAW_DIR,
        ATTEMPT_DIR,
        REPO_ROOT,
        REPO_ROOT / "olympiads",
    ]
    for folder in candidates:
        train_path = folder / "train.csv"
        sample_path = folder / "sample_submission.csv"
        if train_path.exists() and sample_path.exists():
            return train_path, sample_path

    searched = "\n".join(f"  - {folder}" for folder in candidates)
    raise FileNotFoundError(
        "Missing Kazakhstan Day 2 Kaggle CSVs: train.csv and sample_submission.csv.\n"
        "Place both files in one of these folders, preferably the raw task folder:\n"
        f"{searched}"
    )


def add_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["attacking_skill"] = df[["finishing", "positioning", "shot_power", "volleys", "long_shots"]].mean(axis=1)
    df["passing_ability"] = df[["short_passing", "long_passing", "vision", "crossing"]].mean(axis=1)
    df["dribble_mobility"] = df[["dribbling", "agility", "balance", "ball_control"]].mean(axis=1)
    df["pace"] = df[["acceleration", "sprint_speed"]].mean(axis=1)
    df["defense_skill"] = df[["interceptions", "standing_tackle", "sliding_tackle", "defensive_awareness"]].mean(axis=1)
    df["physicality"] = df[["strength", "stamina", "jumping", "aggression"]].mean(axis=1)
    df["set_piece_specialist"] = df[["curve", "fk_accuracy", "penalties"]].mean(axis=1)
    df["goalkeeper_score"] = df[GK_COLS].mean(axis=1)
    df["composure_score"] = df[["composure", "reactions"]].mean(axis=1)
    df["offensive_support"] = df[["positioning", "vision", "short_passing"]].mean(axis=1)
    df["attack_support"] = df[["crossing", "curve", "long_passing"]].mean(axis=1)
    df["defending_positioning"] = df[["defensive_awareness", "interceptions", "reactions"]].mean(axis=1)
    df["is_gk"] = df[GK_COLS].gt(40).all(axis=1)
    return df


def preprocess(frame: pd.DataFrame) -> pd.DataFrame:
    values = SimpleImputer(strategy="mean").fit_transform(frame)
    return StandardScaler().fit_transform(values)


def choose_cluster_count(x_field, min_k: int = 5, max_k: int = 14) -> tuple[int, pd.DataFrame]:
    rows = []
    best_k = min_k
    best_score = -1.0

    max_allowed = min(max_k, len(x_field) - 1)
    for k in range(min_k, max_allowed + 1):
        labels = GaussianMixture(n_components=k, random_state=42).fit_predict(x_field)
        score = silhouette_score(x_field, labels)
        rows.append({"k": k, "silhouette": score})
        if score > best_score:
            best_k = k
            best_score = score

    return best_k, pd.DataFrame(rows)


def main() -> int:
    train_path, sample_path = find_input_files()
    df = pd.read_csv(train_path)
    sample = pd.read_csv(sample_path)

    df = add_meta_features(df)
    field_df = df.loc[~df["is_gk"]].copy()
    gk_df = df.loc[df["is_gk"]].copy()

    x_field = preprocess(field_df[OUTFIELD_FEATURES])
    best_k, scores = choose_cluster_count(x_field)
    field_labels = GaussianMixture(n_components=best_k, random_state=42).fit_predict(x_field)
    field_df["cluster"] = field_labels

    if len(gk_df) > 0:
        gk_df["cluster"] = best_k

    clustered = pd.concat([field_df[["id", "cluster"]], gk_df[["id", "cluster"]]], ignore_index=True)
    clustered["cluster"] = clustered["cluster"].astype(int)

    submission = sample.drop(columns=["cluster"], errors="ignore").merge(clustered, on="id", how="left")
    if submission["cluster"].isna().any():
        missing = int(submission["cluster"].isna().sum())
        raise ValueError(f"Submission has {missing} rows without a cluster label.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    scores.to_csv(OUTPUT_DIR / "cluster_count_scores.csv", index=False)

    summary = pd.DataFrame(
        {
            "item": [
                "train_path",
                "sample_path",
                "train_shape",
                "sample_shape",
                "outfield_players",
                "goalkeepers",
                "chosen_outfield_clusters",
                "total_clusters",
            ],
            "value": [
                str(train_path),
                str(sample_path),
                str(df.shape),
                str(sample.shape),
                str(len(field_df)),
                str(len(gk_df)),
                str(best_k),
                str(submission["cluster"].nunique()),
            ],
        }
    )
    summary.to_csv(OUTPUT_DIR / "run_summary.csv", index=False)

    sanity = (
        df.merge(clustered, on="id", how="left")
        .groupby("cluster")[OUTFIELD_FEATURES + ["goalkeeper_score"]]
        .mean()
        .round(2)
    )
    sanity.to_csv(OUTPUT_DIR / "cluster_feature_means.csv")

    print(f"Wrote {OUTPUT_DIR / 'submission.csv'}")
    print(f"Chosen outfield clusters: {best_k}")
    print(f"Total cluster labels: {submission['cluster'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
