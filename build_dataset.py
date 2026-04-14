import argparse
from pathlib import Path

from mlb_daily import DATA_DIR, build_training_rows, write_rows_to_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an MLB training dataset with rolling team form, venue splits, bullpen workload, "
            "and probable starter history."
        )
    )
    parser.add_argument("--target-season", type=int, required=True, help="Season to build, for example 2025.")
    parser.add_argument(
        "--before-date",
        help="Optional cutoff date in YYYY-MM-DD. Completed games on or after this date are excluded.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path. Defaults to data/mlb_daily_training_<season>.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = f"_through_{args.before_date}" if args.before_date else ""
    output_path = args.output or DATA_DIR / f"mlb_daily_training_{args.target_season}{suffix}.csv"

    print(f"Building rolling MLB training dataset for {args.target_season}...")
    rows = build_training_rows(args.target_season, before_date=args.before_date)
    write_rows_to_csv(rows, output_path)
    print(f"Saved {len(rows)} rows to: {output_path}")


if __name__ == "__main__":
    main()
