import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

from mlb_daily import DATA_DIR, extract_game_outcome, fetch_live_feed, safe_int


MODEL_PATH = Path("artifacts/mlb_game_winner_model.joblib")
METADATA_COLUMNS = ["official_date", "game_datetime", "game_pk", "away_team_name", "home_team_name"]
DEFAULT_REPORT_PATH = Path("mlbResults.txt")
LOCAL_TIMEZONE = ZoneInfo("America/New_York")
ACCURACY_WINDOWS = (
    ("Yesterday", 1),
    ("Past 7 Days", 7),
    ("Past Month", 30),
)


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


def final_home_win_for_game(game_pk: int) -> int | None:
    feed = fetch_live_feed(game_pk, use_cache=False)
    status = feed.get("gameData", {}).get("status", {}) or {}
    if status.get("codedGameState") != "F":
        return None
    home_runs, away_runs = extract_game_outcome(feed)
    return int(home_runs > away_runs)


def load_historical_prediction_rows(before_date: date) -> pd.DataFrame:
    frames = []
    for path in sorted(DATA_DIR.glob("daily_features_*.csv")):
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if "official_date" not in df.columns or df.empty:
            continue
        df["official_date"] = df["official_date"].astype(str)
        df = df[df["official_date"] < before_date.isoformat()]
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["official_date", "game_pk"])


def build_accuracy_dataframe(
    target_date: str,
    feature_names: list[str],
    model: object,
) -> pd.DataFrame:
    target_day = date.fromisoformat(target_date)
    historical_df = load_historical_prediction_rows(target_day)
    if historical_df.empty:
        return pd.DataFrame()

    predictions = predict_dataframe(historical_df, feature_names, model)
    actual_home_wins: dict[int, int] = {}
    rows = []
    for _, row in predictions.iterrows():
        game_pk = safe_int(row.get("game_pk"))
        if not game_pk:
            continue
        if game_pk not in actual_home_wins:
            actual_home_win = final_home_win_for_game(game_pk)
            if actual_home_win is None:
                continue
            actual_home_wins[game_pk] = actual_home_win
        actual_winner = row["home_team_name"] if actual_home_wins[game_pk] == 1 else row["away_team_name"]
        rows.append(
            {
                "official_date": str(row["official_date"]),
                "game_pk": game_pk,
                "correct": int(row["predicted_winner"] == actual_winner),
            }
        )
    return pd.DataFrame(rows)


def format_accuracy_record(label: str, accuracy_df: pd.DataFrame, mask: pd.Series) -> str:
    window_df = accuracy_df.loc[mask]
    if window_df.empty:
        return f"{label}: N/A (0 games)"
    correct = int(window_df["correct"].sum())
    total = len(window_df)
    return f"{label}: {correct}/{total} ({correct / total:.1%})"


def build_accuracy_lines(
    target_date: str,
    feature_names: list[str],
    model: object,
) -> list[str]:
    accuracy_df = build_accuracy_dataframe(target_date, feature_names, model)
    lines = ["Model Accuracy"]
    if accuracy_df.empty:
        lines.append("Yesterday: N/A (0 games)")
        lines.append("Past 7 Days: N/A (0 games)")
        lines.append("Past Month: N/A (0 games)")
        lines.append("This Season: N/A (0 games)")
        lines.append("All Time: N/A (0 games)")
        return lines

    target_day = date.fromisoformat(target_date)
    official_dates = pd.to_datetime(accuracy_df["official_date"]).dt.date
    for label, days in ACCURACY_WINDOWS:
        start_day = target_day - timedelta(days=days)
        if days == 1:
            mask = official_dates == start_day
        else:
            mask = (official_dates >= start_day) & (official_dates < target_day)
        lines.append(format_accuracy_record(label, accuracy_df, mask))

    season_start = date(target_day.year, 1, 1)
    season_mask = (official_dates >= season_start) & (official_dates < target_day)
    all_time_mask = official_dates < target_day
    lines.append(format_accuracy_record("This Season", accuracy_df, season_mask))
    lines.append(format_accuracy_record("All Time", accuracy_df, all_time_mask))
    return lines


def build_report_lines(
    predictions: pd.DataFrame,
    accuracy_lines: list[str] | None = None,
) -> list[str]:
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

    if accuracy_lines:
        lines.extend(accuracy_lines)
        lines.append("")

    return lines


def build_text_report(predictions: pd.DataFrame, accuracy_lines: list[str] | None = None) -> str:
    return "\n".join(build_report_lines(predictions, accuracy_lines)).strip()


def print_accuracy_summary(accuracy_lines: list[str] | None) -> None:
    if not accuracy_lines:
        return
    print("")
    for line in accuracy_lines:
        print(line)


def print_console_summary(
    predictions: pd.DataFrame,
    report_path: Path,
    accuracy_lines: list[str] | None = None,
) -> None:
    print("MLB Daily Predictions")
    print("---------------------")

    if len(predictions) == 1:
        row = predictions.iloc[0]
        print(f"Matchup: {build_matchup_label(row)}")
        print(f"Selection: {row['predicted_winner']}")
        print(f"Confidence Tier: {row['confidence_label']}")
        print(f"Confidence Outlook: {row['probability_label']}")
        print(f"Report: {report_path}")
        print_accuracy_summary(accuracy_lines)
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
    print_accuracy_summary(accuracy_lines)
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
    target_date = str(df["official_date"].max()) if "official_date" in df.columns else date.today().isoformat()
    accuracy_lines = build_accuracy_lines(target_date, feature_names, model)
    report_text = build_text_report(predictions, accuracy_lines)
    args.output_report.write_text(report_text + "\n", encoding="utf-8")
    print_console_summary(predictions, args.output_report, accuracy_lines)


if __name__ == "__main__":
    main()
