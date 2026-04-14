import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd


MODEL_PATH = Path("artifacts/mlb_game_winner_model.joblib")
METADATA_COLUMNS = ["official_date", "game_datetime", "game_pk", "away_team_name", "home_team_name"]
DEFAULT_REPORT_PATH = Path("mlbResults.txt")
LOCAL_TIMEZONE = ZoneInfo("America/New_York")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the probability that the home team wins an MLB game.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--features-json",
        type=Path,
        help="Path to a JSON file with the same feature names used during training.",
    )
    group.add_argument(
        "--features-csv",
        type=Path,
        help="Path to a CSV file containing one or more feature rows.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to the text report file. Defaults to 'mlb results.txt'.",
    )
    return parser.parse_args()


def predict_dataframe(df: pd.DataFrame, feature_names: list[str], model: object) -> pd.DataFrame:
    features = df.reindex(columns=feature_names)
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    confidence_scores = [max(probability, 1.0 - probability) for probability in probabilities]

    output = df[[column for column in METADATA_COLUMNS if column in df.columns]].copy()
    output["predicted_winner"] = [
        df.iloc[index]["home_team_name"] if value == 1 else df.iloc[index]["away_team_name"]
        for index, value in enumerate(predictions)
    ]
    output["home_win_probability"] = probabilities
    output["predicted_winner_probability"] = [
        probability if prediction == 1 else 1.0 - probability
        for probability, prediction in zip(probabilities, predictions)
    ]
    output["confidence_score"] = confidence_scores
    output["confidence_label"] = [confidence_label(score) for score in confidence_scores]
    output["probability_label"] = [probability_label(score) for score in confidence_scores]
    output = output.sort_values(
        by=["confidence_score", "official_date", "game_pk"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return output


def confidence_label(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.65:
        return "Medium"
    return "Low"


def probability_label(score: float) -> str:
    if score >= 0.75:
        return "Very Likely"
    if score >= 0.65:
        return "Likely"
    if score >= 0.55:
        return "Lean"
    return "Toss-Up"


def build_matchup_label(row: pd.Series) -> str:
    return f"{row.get('away_team_name', 'Away Team')} at {row.get('home_team_name', 'Home Team')}"


def format_start_line(row: pd.Series) -> str:
    game_datetime = row.get("game_datetime")
    if isinstance(game_datetime, str) and game_datetime:
        dt = datetime.fromisoformat(game_datetime.replace("Z", "+00:00")).astimezone(LOCAL_TIMEZONE)
        return dt.strftime("%I:%M %p ET")
    return ""


def build_report_lines(predictions: pd.DataFrame) -> list[str]:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "MLB Daily Prediction Report",
        f"Generated: {generated_at}",
        f"Games: {len(predictions)}",
        "",
    ]

    for index, (_, row) in enumerate(predictions.iterrows(), start=1):
        lines.append(f"{index}. {build_matchup_label(row)}")
        lines.append(f"   Start Time: {format_start_line(row)}")
        lines.append(f"   Selection: {row['predicted_winner']}")
        lines.append(f"   Confidence Tier: {row['confidence_label']}")
        lines.append(f"   Confidence Outlook: {row['probability_label']}")
        lines.append("")

    return lines


def build_text_report(predictions: pd.DataFrame) -> str:
    return "\n".join(build_report_lines(predictions)).strip()


def print_console_summary(predictions: pd.DataFrame, report_path: Path) -> None:
    print("MLB Daily Predictions")
    print("---------------------")

    if len(predictions) == 1:
        row = predictions.iloc[0]
        print(f"Matchup: {build_matchup_label(row)}")
        print(f"Selection: {row['predicted_winner']}")
        print(f"Confidence Tier: {row['confidence_label']}")
        print(f"Confidence Outlook: {row['probability_label']}")
        print(f"Report: {report_path}")
        return

    display_columns = [
        "official_date",
        "game_datetime",
        "game_pk",
        "away_team_name",
        "home_team_name",
        "predicted_winner",
        "confidence_label",
        "probability_label",
    ]
    renamed = predictions[display_columns].rename(
        columns={
            "official_date": "date",
            "game_datetime": "start_time_utc",
            "game_pk": "game_id",
            "away_team_name": "away_team",
            "home_team_name": "home_team",
            "predicted_winner": "selection",
            "confidence_label": "confidence_tier",
            "probability_label": "confidence_outlook",
        }
    )
    if "start_time_utc" in renamed.columns:
        renamed["start_time_utc"] = renamed["start_time_utc"].fillna("").astype(str)
    print(renamed.to_string(index=False))
    print(f"\nReport: {report_path}")


def main() -> None:
    args = parse_args()
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_names = saved["feature_names"]

    if args.features_json:
        with args.features_json.open("r", encoding="utf-8") as handle:
            feature_values = json.load(handle)
        df = pd.DataFrame([feature_values])
    else:
        df = pd.read_csv(args.features_csv)

    predictions = predict_dataframe(df, feature_names, model)
    report_text = build_text_report(predictions)
    args.output_report.write_text(report_text + "\n", encoding="utf-8")
    print_console_summary(predictions, args.output_report)


if __name__ == "__main__":
    main()
