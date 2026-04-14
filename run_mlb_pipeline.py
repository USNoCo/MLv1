import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from build_daily_features import slugify
from mlb_daily import DATA_DIR, build_daily_prediction_rows, build_training_rows, write_rows_to_csv
from predict import DEFAULT_REPORT_PATH, predict_dataframe
from train import build_pipeline, load_dataset, save_model


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


def main() -> None:
    args = parse_args()
    target_date = args.date
    target_year = date.fromisoformat(target_date).year
    previous_year = target_year - 1

    training_rows = []
    training_rows.extend(build_training_rows(previous_year))
    training_rows.extend(build_training_rows(target_year, before_date=target_date))

    training_dataset_path = DATA_DIR / f"mlb_training_through_{target_date}.csv"
    write_rows_to_csv(training_rows, training_dataset_path)
    print(f"Saved combined training data to: {training_dataset_path}")

    df, feature_columns = load_dataset(training_dataset_path)
    pipeline = build_pipeline()
    pipeline.fit(df[feature_columns], df["home_win"])
    save_model(pipeline, feature_columns, training_dataset_path)
    print(f"Fitted production model on {len(df)} historical rows through {target_date}.")

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
    print(f"Saved scheduled-game features to: {prediction_csv_path}")

    prediction_df = pd.DataFrame(prediction_rows)
    predictions = predict_dataframe(prediction_df, feature_columns, pipeline)

    lines: list[str] = []
    for _, row in predictions.iterrows():
        lines.append(f"Date: {row.get('official_date', '')}")
        lines.append(f"Game ID: {row.get('game_pk', '')}")
        lines.append(f"Matchup: {row.get('away_team_name', 'Away Team')} at {row.get('home_team_name', 'Home Team')}")
        lines.append(f"Predicted winner: {row['predicted_winner']}")
        lines.append(f"Confidence: {row['confidence_label']}")
        lines.append(f"Probability label: {row['probability_label']}")
        lines.append("")
    args.output_report.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    display_columns = [
        "official_date",
        "game_pk",
        "away_team_name",
        "home_team_name",
        "predicted_winner",
        "confidence_label",
        "probability_label",
    ]
    print(predictions[display_columns].to_string(index=False))
    print(f"\nSaved report to: {args.output_report}")


if __name__ == "__main__":
    main()
