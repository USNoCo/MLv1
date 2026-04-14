import argparse
import json
from pathlib import Path

from mlb_daily import DATA_DIR, build_daily_prediction_rows, write_rows_to_csv


def slugify(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build daily MLB prediction features for scheduled games on a given date."
    )
    parser.add_argument("--date", required=True, help="Target date in YYYY-MM-DD format.")
    parser.add_argument("--season", type=int, help="Season override. Defaults to the year from --date.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Output CSV path. Defaults to data/daily_features_<date>.csv",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        help="Optional directory where one JSON feature file per game will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_daily_prediction_rows(args.date, season=args.season)
    output_csv = args.output_csv or DATA_DIR / f"daily_features_{args.date}.csv"
    write_rows_to_csv(rows, output_csv)
    print(f"Saved {len(rows)} scheduled-game rows to: {output_csv}")

    if args.json_dir:
        args.json_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            filename = (
                f"{row['official_date']}_{slugify(str(row['away_team_name']))}_at_"
                f"{slugify(str(row['home_team_name']))}.json"
            )
            with (args.json_dir / filename).open("w", encoding="utf-8") as handle:
                json.dump(row, handle, indent=2)
        print(f"Saved per-game JSON files to: {args.json_dir}")


if __name__ == "__main__":
    main()
