import argparse
import hashlib
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DEFAULT_DATASET_PATH = Path("data/mlb_daily_training_2025.csv")
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "mlb_game_winner_model.joblib"
SIGNATURE_VERSION = 1
TRAINING_SIGNATURE_FILES = (
    Path("train.py"),
    Path("run_mlb_pipeline.py"),
    Path("mlb_daily.py"),
    Path("build_dataset.py"),
    Path("build_daily_features.py"),
    Path("requirements.txt"),
)
CLASSIFIER_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 6,
    "max_iter": 300,
    "min_samples_leaf": 20,
    "random_state": 42,
}
METADATA_COLUMNS = {
    "official_date",
    "game_datetime",
    "game_pk",
    "home_team_id",
    "away_team_id",
    "home_team_name",
    "away_team_name",
}
EXCLUDED_FEATURE_PREFIXES = (
    "home_last_season_",
    "away_last_season_",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLB game winner prediction model.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to a CSV file containing game-level MLB features and a home_win target column.",
    )
    parser.add_argument(
        "--fit-all",
        action="store_true",
        help="Fit on the full dataset without a holdout split. Use this for final production predictions.",
    )
    return parser.parse_args()


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(**CLASSIFIER_PARAMS),
            ),
        ]
    )


def load_dataset(dataset_path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(dataset_path)
    if "home_win" not in df.columns:
        raise ValueError("Dataset must include a 'home_win' column.")
    feature_columns = [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS
        and column != "home_win"
        and not column.startswith(EXCLUDED_FEATURE_PREFIXES)
    ]
    return df, feature_columns


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_training_signature(feature_columns: list[str]) -> dict[str, object]:
    return {
        "version": SIGNATURE_VERSION,
        "pipeline": {
            "imputer": {"strategy": "median"},
            "model": {
                "class": "sklearn.ensemble.HistGradientBoostingClassifier",
                "params": CLASSIFIER_PARAMS,
            },
        },
        "feature_columns": feature_columns,
        "metadata_columns": sorted(METADATA_COLUMNS),
        "excluded_feature_prefixes": list(EXCLUDED_FEATURE_PREFIXES),
        "source_hashes": {str(path): file_sha256(path) for path in TRAINING_SIGNATURE_FILES},
    }


def training_signature_matches(saved: dict[str, object], feature_columns: list[str]) -> bool:
    return saved.get("training_signature") == build_training_signature(feature_columns)


def latest_official_date(df: pd.DataFrame) -> str | None:
    if "official_date" not in df.columns or df.empty:
        return None
    dates = df["official_date"].dropna().astype(str)
    if dates.empty:
        return None
    return str(dates.max())


def load_saved_model() -> dict[str, object] | None:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def save_model(
    pipeline: Pipeline,
    feature_columns: list[str],
    dataset_path: Path,
    *,
    training_signature: dict[str, object] | None = None,
    latest_training_date: str | None = None,
    update_type: str = "full",
) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": pipeline,
            "feature_names": feature_columns,
            "dataset_path": str(dataset_path),
            "training_signature": training_signature or build_training_signature(feature_columns),
            "latest_training_date": latest_training_date,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "update_type": update_type,
        },
        MODEL_PATH,
    )
    print(f"\nSaved model to: {MODEL_PATH}")


def main() -> None:
    args = parse_args()

    df, feature_columns = load_dataset(args.dataset)
    X = df[feature_columns]
    y = df["home_win"]

    pipeline = build_pipeline()

    if args.fit_all:
        pipeline.fit(X, y)
        print(f"Fitted model on all {len(df)} rows.")
        save_model(
            pipeline,
            feature_columns,
            args.dataset,
            latest_training_date=latest_official_date(df),
        )
        return

    if "official_date" in df.columns:
        ordered = df.sort_values(["official_date", "game_pk"]).reset_index(drop=True)
        split_index = int(len(ordered) * 0.8)
        train_df = ordered.iloc[:split_index]
        test_df = ordered.iloc[split_index:]
        X_train = train_df[feature_columns]
        y_train = train_df["home_win"]
        X_test = test_df[feature_columns]
        y_test = test_df["home_win"]
        print(
            "Using time-based split: "
            f"{train_df['official_date'].iloc[0]} to {train_df['official_date'].iloc[-1]} for training, "
            f"{test_df['official_date'].iloc[0]} to {test_df['official_date'].iloc[-1]} for testing."
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"ROC AUC:  {roc_auc:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["away_win", "home_win"]))

    save_model(
        pipeline,
        feature_columns,
        args.dataset,
        latest_training_date=latest_official_date(train_df if "official_date" in df.columns else df),
    )


if __name__ == "__main__":
    main()
