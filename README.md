# MLB Daily Prediction Model

This project now builds a stronger MLB game-winner model that updates from daily stats.

The feature set uses only information available before first pitch:

- rolling team form from prior completed games
- home and away split performance
- recent bullpen workload
- probable starter history from prior starts
- chronological train/test evaluation

## Files

- `build_dataset.py` builds a season training set with rolling daily features
- `build_daily_features.py` builds prediction features for scheduled games on a date
- `mlb_daily.py` contains the shared feature-engineering pipeline
- `train.py` trains the model on the generated dataset
- `predict.py` scores either a JSON file or a CSV of daily features

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## One command

```powershell
python run_mlb_pipeline.py --date 2026-04-14
```

That one command will:

- rebuild historical training data through the target date
- retrain the model on all available past games
- generate same-day features for scheduled games
- write the final predictions to `mlbResults.txt`

## Build a daily training dataset

```powershell
python build_dataset.py --target-season 2025
```

That creates `data/mlb_daily_training_2025.csv`.

## Train the model

```powershell
python train.py --dataset data/mlb_daily_training_2025.csv
```

If the dataset includes `official_date`, training uses the first 80% of games for training and the last 20% for testing.

## Build features for a specific date

```powershell
python build_daily_features.py --date 2025-09-01 --json-dir data/prediction_json
```

That creates:

- `data/daily_features_2025-09-01.csv`
- one JSON file per scheduled game in `data/prediction_json`

## Score scheduled games

```powershell
python predict.py --features-csv data/daily_features_2025-09-01.csv
```

## Important note

This is a much better baseline, but it is still not a finished betting-grade system.

The next biggest upgrades would be:

- injured-list and roster availability
- park factors and weather
- handedness splits against the scheduled starter
- confirmed lineup features
- multi-season training instead of one season
