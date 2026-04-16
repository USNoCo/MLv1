# MLB Daily Prediction Model

A polished end-to-end MLB prediction workflow that rebuilds historical data, retrains a model on pre-game information, and produces same-day matchup selections in a clean text report.

The project is built around a practical forecasting goal: generate daily MLB predictions using only information that would have been available before first pitch.

## Overview

This project:

- builds rolling team and pitcher features from MLB API data
- trains a supervised classification model to predict game winners
- evaluates the model with chronological splits
- generates same-day features only for games that have not started yet
- writes sorted predictions to `mlbResults.txt`

At its core, the system avoids future leakage by creating each training row from information available before the game being predicted.

## Project Structure

- `run_mlb_pipeline.py`: one-command workflow for rebuilding data, retraining, and generating the final report
- `mlb_daily.py`: shared data collection, caching, rolling feature engineering, and dataset generation logic
- `build_dataset.py`: builds a season-level training dataset from rolling daily features
- `build_daily_features.py`: builds same-day prediction features for scheduled games
- `train.py`: trains and saves the model
- `predict.py`: scores feature files and writes the final text report

## Modeling Approach

The model is a `HistGradientBoostingClassifier` from `scikit-learn`.

Training target:

- `home_win = 1` when the home team wins
- `home_win = 0` when the away team wins

Evaluation approach:

- chronological train/test splitting when date information is present
- final production runs can fit on all historical rows available through the chosen prediction date

## Feature Set

The model uses pre-game information only.

### Team performance factors

- season win percentage
- season average runs scored
- season average runs allowed
- season average run differential
- season average home runs
- season average walks
- season average strikeouts
- season average OPS
- season average stolen bases
- season pitching ERA
- season pitching WHIP
- season pitching K/9
- season pitching BB/9
- season average errors
- home or away split win percentage
- home or away split average run differential
- last 3 game win percentage
- last 5 game win percentage
- last 10 game win percentage
- last 3 game scoring average
- last 3 game runs allowed average
- last 5 game scoring average
- last 5 game runs allowed average
- last 10 game scoring average
- last 10 game runs allowed average
- last 5 game OPS average
- last 5 game home run average
- last 5 game walk average
- last 5 game strikeout average
- last 5 game pitching ERA
- last 5 game pitching WHIP
- bullpen outs average over the last 3 games
- bullpen pitches average over the last 3 games
- bullpen ERA over the last 3 games
- days since last game

### Probable pitcher factors

- prior starts
- ERA from prior starts
- WHIP from prior starts
- K/9 from prior starts
- BB/9 from prior starts
- innings per start
- pitches per start
- ERA over the last 3 starts
- WHIP over the last 3 starts
- K/9 over the last 3 starts
- days since last start

### Lineup availability factors

- confirmed MLB batting order when available
- projected batting order from recent starters when no confirmed lineup has posted
- current injured-list filtering for projected daily lineups
- projected or confirmed lineup OPS, OBP, SLG, plate appearances, and starts

## Quick Start

### Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### One-command run

```powershell
python run_mlb_pipeline.py --date YYYY-MM-DD
```

That command will:

- rebuild historical training data through the chosen date
- retrain the production model on all available past games
- generate feature rows for not-yet-started games on that date
- write a ranked prediction report to `mlbResults.txt`

## Step-by-Step Workflow

### Build a training dataset

```powershell
python build_dataset.py --target-season 2025
```

Output:

- `data/mlb_daily_training_2025.csv`

### Train the model

```powershell
python train.py --dataset data/mlb_daily_training_2025.csv
```

### Build same-day features

```powershell
python build_daily_features.py --date 2025-09-01 --json-dir data/prediction_json
```

Outputs:

- `data/daily_features_2025-09-01.csv`
- per-game JSON files in `data/prediction_json`

### Generate predictions

```powershell
python predict.py --features-csv data/daily_features_2025-09-01.csv
```

Report output:

- `mlbResults.txt`

## Output Format

The final report is ranked by strongest confidence first and includes:

- matchup
- start time in Eastern Time
- selection
- confidence tier
- confidence outlook
- generated timestamp
- total number of games in the report

Example report entry:

```text
1. Chicago Cubs at Philadelphia Phillies
   Start Time: 06:40 PM ET
   Selection: Philadelphia Phillies
   Confidence Tier: High
   Confidence Outlook: Very Likely
```

## Current Limitations

This is a strong project baseline, but it is not yet a full production-grade betting system.

Not included yet:

- weather
- betting market odds
- umpire assignments
- handedness matchup splits
- Statcast contact-quality features
- park factor adjustments
- multi-season training beyond the current workflow

## Next Improvements

The highest-impact next upgrades would be:

- weather and park-factor integration
- handedness-based batting and pitching splits
- multi-season backfilling for a larger training set
- richer pitcher and bullpen quality metrics
