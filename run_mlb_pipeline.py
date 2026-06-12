import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from build_daily_features import slugify
from mlb_daily import DATA_DIR, build_daily_prediction_rows, build_training_rows, write_rows_to_csv
from predict import DEFAULT_REPORT_PATH, build_accuracy_lines, build_text_report, print_console_summary, predict_dataframe
from train import (
    build_pipeline,
    build_training_signature,
    latest_official_date,
    load_dataset,
    load_saved_model,
    save_model,
    training_signature_matches,
)


INCREMENTAL_LOOKBACK_DAYS = 3
INCREMENTAL_EXTRA_ITERATIONS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-command MLB workflow: rebuild training data, retrain on all available historical games, "
            "build daily features, and write predictions to mlbResults.txt."
        )
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Prediction date in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to the prediction report text file.",
    )
    return parser.parse_args()


def select_incremental_rows(
    df: pd.DataFrame,
    target_date: str,
    last_training_date: str | None,
) -> pd.DataFrame:
    if "official_date" not in df.columns:
        return df.iloc[0:0]

    target_day = date.fromisoformat(target_date)
    lookback_start = (target_day - timedelta(days=INCREMENTAL_LOOKBACK_DAYS)).isoformat()
    official_dates = df["official_date"].astype(str)
    mask = (official_dates >= lookback_start) & (official_dates < target_date)
    if last_training_date:
        mask &= official_dates > last_training_date
    return df.loc[mask].sort_values(["official_date", "game_pk"]).reset_index(drop=True)


def warm_start_recent_update(
    pipeline: Pipeline,
    recent_df: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    model = pipeline.named_steps["model"]
    imputer = pipeline.named_steps["imputer"]
    current_max_iter = int(model.get_params()["max_iter"])
    model.set_params(
        warm_start=True,
        max_iter=current_max_iter + INCREMENTAL_EXTRA_ITERATIONS,
    )
    model.fit(imputer.transform(recent_df[feature_columns]), recent_df["home_win"])


def load_or_update_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    training_dataset_path: Path,
    target_date: str,
) -> Pipeline:
    saved = load_saved_model()
    current_signature = build_training_signature(feature_columns)
    latest_date = latest_official_date(df)

    if saved is None:
        pipeline = build_pipeline()
        pipeline.fit(df[feature_columns], df["home_win"])
        save_model(
            pipeline,
            feature_columns,
            training_dataset_path,
            training_signature=current_signature,
            latest_training_date=latest_date,
        )
        print(f"Model fit complete using {len(df)} historical games through {target_date}.")
        return pipeline

    saved_feature_names = saved.get("feature_names")
    if saved_feature_names != feature_columns or not training_signature_matches(saved, feature_columns):
        print("Training setup changed; running a full model retrain.")
        pipeline = build_pipeline()
        pipeline.fit(df[feature_columns], df["home_win"])
        save_model(
            pipeline,
            feature_columns,
            training_dataset_path,
            training_signature=current_signature,
            latest_training_date=latest_date,
        )
        print(f"Model fit complete using {len(df)} historical games through {target_date}.")
        return pipeline

    pipeline = saved["model"]
    recent_df = select_incremental_rows(
        df,
        target_date,
        saved.get("latest_training_date"),
    )
    if recent_df.empty:
        print("Training setup unchanged; reusing saved model because there are no new last-3-day rows.")
        return pipeline
    if recent_df["home_win"].nunique() < 2:
        print("Training setup unchanged; reusing saved model because recent rows contain only one class.")
        return pipeline

    warm_start_recent_update(pipeline, recent_df, feature_columns)
    save_model(
        pipeline,
        feature_columns,
        training_dataset_path,
        training_signature=current_signature,
        latest_training_date=latest_official_date(recent_df) or saved.get("latest_training_date"),
        update_type="incremental",
    )
    print(
        "Model warm-start update complete using "
        f"{len(recent_df)} new games from the last {INCREMENTAL_LOOKBACK_DAYS} days."
    )
    return pipeline


def main() -> None:
    args = parse_args()
    target_date = args.date
    target_year = date.fromisoformat(target_date).year
    previous_year = target_year - 1

    print("MLB Prediction Pipeline")
    print("-----------------------")
    print(f"Target date: {target_date}")
    print("")

    training_rows = []
    training_rows.extend(build_training_rows(previous_year))
    training_rows.extend(build_training_rows(target_year, before_date=target_date, use_cache=False))

    training_dataset_path = DATA_DIR / f"mlb_training_through_{target_date}.csv"
    write_rows_to_csv(training_rows, training_dataset_path)
    print(f"Training dataset: {training_dataset_path}")
    print(f"Training rows: {len(training_rows)}")

    df, feature_columns = load_dataset(training_dataset_path)
    pipeline = load_or_update_model(df, feature_columns, training_dataset_path, target_date)

    prediction_rows = build_daily_prediction_rows(target_date, season=target_year)
    prediction_csv_path = DATA_DIR / f"daily_features_{target_date}.csv"
    prediction_json_dir = DATA_DIR / f"prediction_json_{target_date}"
    write_rows_to_csv(prediction_rows, prediction_csv_path)
    prediction_json_dir.mkdir(parents=True, exist_ok=True)
    for row in prediction_rows:
        filename = (
            f"{row['official_date']}_{slugify(str(row['away_team_name']))}_at_"
            f"{slugify(str(row['home_team_name']))}.json"
        )
        with (prediction_json_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(row, handle, indent=2)
    print(f"Daily feature file: {prediction_csv_path}")
    print(f"Scheduled games: {len(prediction_rows)}")

    prediction_df = pd.DataFrame(prediction_rows)
    predictions = predict_dataframe(prediction_df, feature_columns, pipeline)
    accuracy_lines = build_accuracy_lines(target_date, feature_columns, pipeline)
    args.output_report.write_text(build_text_report(predictions, accuracy_lines) + "\n", encoding="utf-8")

    print("")
    print_console_summary(predictions, args.output_report, accuracy_lines)


if __name__ == "__main__":
    main()
