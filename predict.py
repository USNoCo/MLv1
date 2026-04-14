import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("artifacts/mlb_game_winner_model.joblib")
METADATA_COLUMNS = ["official_date", "game_pk", "away_team_name", "home_team_name"]
DEFAULT_REPORT_PATH = Path("mlbResults.txt")


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


def build_text_report(predictions: pd.DataFrame) -> str:
    lines: list[str] = []

    if len(predictions) == 1:
        row = predictions.iloc[0]
        if "official_date" in row:
            lines.append(f"Date: {row['official_date']}")
        if "game_pk" in row:
            lines.append(f"Game ID: {row['game_pk']}")
        if "away_team_name" in row and "home_team_name" in row:
            lines.append(f"Matchup: {row['away_team_name']} at {row['home_team_name']}")
        lines.append(f"Predicted winner: {row['predicted_winner']}")
        lines.append(f"Confidence: {row['confidence_label']}")
        lines.append(f"Probability label: {row['probability_label']}")
        return "\n".join(lines)

    for _, row in predictions.iterrows():
        lines.append(f"Date: {row.get('official_date', '')}")
        lines.append(f"Game ID: {row.get('game_pk', '')}")
        lines.append(f"Matchup: {row.get('away_team_name', 'Away Team')} at {row.get('home_team_name', 'Home Team')}")
        lines.append(f"Predicted winner: {row['predicted_winner']}")
        lines.append(f"Confidence: {row['confidence_label']}")
        lines.append(f"Probability label: {row['probability_label']}")
        lines.append("")

    return "\n".join(lines).strip()


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

    if len(predictions) == 1:
        row = predictions.iloc[0]
        if "away_team_name" in row and "home_team_name" in row:
            print(f"Matchup: {row['away_team_name']} at {row['home_team_name']}")
        print(f"Predicted winner: {row['predicted_winner']}")
        print(f"Confidence: {row['confidence_label']}")
        print(f"Probability label: {row['probability_label']}")
        print(f"Saved report to: {args.output_report}")
        return

    display_columns = [
        column
        for column in [
            "official_date",
            "game_pk",
            "away_team_name",
            "home_team_name",
            "predicted_winner",
            "confidence_label",
            "probability_label",
        ]
        if column in predictions.columns
    ]
    print(predictions[display_columns].to_string(index=False))
    print(f"\nSaved report to: {args.output_report}")


if __name__ == "__main__":
    main()
