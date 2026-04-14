import argparse
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
METADATA_COLUMNS = {
    "official_date",
    "game_datetime",
    "game_pk",
    "home_team_id",
    "away_team_id",
    "home_team_name",
    "away_team_name",
}


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
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=6,
                    max_iter=300,
                    min_samples_leaf=20,
                    random_state=42,
                ),
            ),
        ]
    )


def load_dataset(dataset_path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(dataset_path)
    if "home_win" not in df.columns:
        raise ValueError("Dataset must include a 'home_win' column.")
    feature_columns = [column for column in df.columns if column not in METADATA_COLUMNS and column != "home_win"]
    return df, feature_columns


def save_model(pipeline: Pipeline, feature_columns: list[str], dataset_path: Path) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": pipeline,
            "feature_names": feature_columns,
            "dataset_path": str(dataset_path),
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
        save_model(pipeline, feature_columns, args.dataset)
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

    save_model(pipeline, feature_columns, args.dataset)


if __name__ == "__main__":
    main()
