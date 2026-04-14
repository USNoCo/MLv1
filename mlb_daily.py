import csv
import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://statsapi.mlb.com/api/v1"
LIVE_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game"
SPORT_ID = 1
REGULAR_SEASON_GAME_TYPE = "R"
DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"


def safe_float(value: Any) -> float | None:
    if value in (None, "", "--", "-.--", ".---"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def era(earned_runs: float, outs_recorded: float) -> float | None:
    if outs_recorded == 0:
        return None
    return earned_runs * 27.0 / outs_recorded


def per_nine(value: float, outs_recorded: float) -> float | None:
    if outs_recorded == 0:
        return None
    return value * 27.0 / outs_recorded


def days_between(earlier: str | None, later: str) -> int | None:
    if not earlier:
        return None
    return (date.fromisoformat(later) - date.fromisoformat(earlier)).days


def cache_path(prefix: str, key: str) -> Path:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    path = CACHE_DIR / prefix
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{digest}.json"


def fetch_json(url: str, params: dict[str, Any], cache_group: str, cache_key: str) -> dict[str, Any]:
    path = cache_path(cache_group, cache_key)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def fetch_schedule_for_season(season: int) -> list[dict[str, Any]]:
    data = fetch_json(
        f"{BASE_URL}/schedule",
        {
            "sportId": SPORT_ID,
            "season": season,
            "gameTypes": REGULAR_SEASON_GAME_TYPE,
            "hydrate": "probablePitcher,team,linescore",
        },
        "schedule",
        f"season-{season}",
    )
    games: list[dict[str, Any]] = []
    for date_block in data.get("dates", []):
        games.extend(date_block.get("games", []))
    return sorted(games, key=lambda game: (game.get("officialDate", ""), game.get("gamePk", 0)))


def fetch_schedule_for_date(target_date: str) -> list[dict[str, Any]]:
    data = fetch_json(
        f"{BASE_URL}/schedule",
        {
            "sportId": SPORT_ID,
            "date": target_date,
            "gameTypes": REGULAR_SEASON_GAME_TYPE,
            "hydrate": "probablePitcher,team,linescore",
        },
        "schedule",
        f"date-{target_date}",
    )
    games: list[dict[str, Any]] = []
    for date_block in data.get("dates", []):
        games.extend(date_block.get("games", []))
    return sorted(games, key=lambda game: (game.get("officialDate", ""), game.get("gamePk", 0)))


def fetch_live_feed(game_pk: int) -> dict[str, Any]:
    return fetch_json(
        f"{LIVE_FEED_URL}/{game_pk}/feed/live",
        {},
        "game_feeds",
        f"game-{game_pk}",
    )


def fetch_pitcher_game_logs(player_id: int, season: int) -> list[dict[str, Any]]:
    data = fetch_json(
        f"{BASE_URL}/people/{player_id}/stats",
        {
            "stats": "gameLog",
            "group": "pitching",
            "season": season,
            "gameType": REGULAR_SEASON_GAME_TYPE,
        },
        "pitcher_logs",
        f"pitcher-{player_id}-season-{season}",
    )
    stats = data.get("stats", [])
    splits = stats[0].get("splits", []) if stats else []
    return sorted(splits, key=lambda item: item.get("date", ""))


@dataclass
class TeamGameRecord:
    date: str
    is_home: bool
    win: int
    runs_scored: int
    runs_allowed: int
    batting_ops: float | None
    batting_home_runs: int
    batting_walks: int
    batting_strikeouts: int
    stolen_bases: int
    errors: int
    pitching_hits_allowed: int
    pitching_walks_allowed: int
    pitching_strikeouts: int
    pitching_earned_runs: int
    pitching_outs: int
    bullpen_outs: int
    bullpen_pitches: int
    bullpen_earned_runs: int


@dataclass
class TeamState:
    games: int = 0
    wins: int = 0
    runs_scored: int = 0
    runs_allowed: int = 0
    batting_home_runs: int = 0
    batting_walks: int = 0
    batting_strikeouts: int = 0
    batting_ops_total: float = 0.0
    batting_ops_games: int = 0
    stolen_bases: int = 0
    pitching_hits_allowed: int = 0
    pitching_walks_allowed: int = 0
    pitching_strikeouts: int = 0
    pitching_earned_runs: int = 0
    pitching_outs: int = 0
    errors: int = 0
    home_games: int = 0
    home_wins: int = 0
    home_runs_scored: int = 0
    home_runs_allowed: int = 0
    away_games: int = 0
    away_wins: int = 0
    away_runs_scored: int = 0
    away_runs_allowed: int = 0
    last_game_date: str | None = None
    recent_games: deque[TeamGameRecord] = field(default_factory=lambda: deque(maxlen=10))
    recent_bullpen: deque[TeamGameRecord] = field(default_factory=lambda: deque(maxlen=3))


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def team_snapshot(state: TeamState, as_home_team: bool, game_date: str) -> dict[str, float | None]:
    last3_games = list(state.recent_games)[-3:]
    last5_games = list(state.recent_games)[-5:]
    last10_games = list(state.recent_games)[-10:]
    bullpen_games = list(state.recent_bullpen)[-3:]

    split_games = state.home_games if as_home_team else state.away_games
    split_wins = state.home_wins if as_home_team else state.away_wins
    split_runs_scored = state.home_runs_scored if as_home_team else state.away_runs_scored
    split_runs_allowed = state.home_runs_allowed if as_home_team else state.away_runs_allowed

    return {
        "games_played": state.games,
        "season_win_pct": ratio(state.wins, state.games),
        "season_runs_scored_avg": ratio(state.runs_scored, state.games),
        "season_runs_allowed_avg": ratio(state.runs_allowed, state.games),
        "season_run_diff_avg": ratio(state.runs_scored - state.runs_allowed, state.games),
        "season_batting_home_runs_avg": ratio(state.batting_home_runs, state.games),
        "season_batting_walks_avg": ratio(state.batting_walks, state.games),
        "season_batting_strikeouts_avg": ratio(state.batting_strikeouts, state.games),
        "season_batting_ops_avg": ratio(state.batting_ops_total, state.batting_ops_games),
        "season_stolen_bases_avg": ratio(state.stolen_bases, state.games),
        "season_pitching_era": era(state.pitching_earned_runs, state.pitching_outs),
        "season_pitching_whip": ratio(state.pitching_hits_allowed + state.pitching_walks_allowed, state.pitching_outs / 3.0),
        "season_pitching_k9": per_nine(state.pitching_strikeouts, state.pitching_outs),
        "season_pitching_bb9": per_nine(state.pitching_walks_allowed, state.pitching_outs),
        "season_errors_avg": ratio(state.errors, state.games),
        "venue_split_win_pct": ratio(split_wins, split_games),
        "venue_split_run_diff_avg": ratio(split_runs_scored - split_runs_allowed, split_games),
        "last3_win_pct": average([game.win for game in last3_games]),
        "last5_win_pct": average([game.win for game in last5_games]),
        "last10_win_pct": average([game.win for game in last10_games]),
        "last3_runs_scored_avg": average([game.runs_scored for game in last3_games]),
        "last3_runs_allowed_avg": average([game.runs_allowed for game in last3_games]),
        "last5_runs_scored_avg": average([game.runs_scored for game in last5_games]),
        "last5_runs_allowed_avg": average([game.runs_allowed for game in last5_games]),
        "last10_runs_scored_avg": average([game.runs_scored for game in last10_games]),
        "last10_runs_allowed_avg": average([game.runs_allowed for game in last10_games]),
        "last5_batting_ops_avg": average([game.batting_ops for game in last5_games if game.batting_ops is not None]),
        "last5_home_runs_avg": average([float(game.batting_home_runs) for game in last5_games]),
        "last5_walks_avg": average([float(game.batting_walks) for game in last5_games]),
        "last5_strikeouts_avg": average([float(game.batting_strikeouts) for game in last5_games]),
        "last5_pitching_era": era(
            sum(game.pitching_earned_runs for game in last5_games),
            sum(game.pitching_outs for game in last5_games),
        ),
        "last5_pitching_whip": ratio(
            sum(game.pitching_hits_allowed + game.pitching_walks_allowed for game in last5_games),
            sum(game.pitching_outs for game in last5_games) / 3.0,
        ),
        "bullpen_last3_outs_avg": average([float(game.bullpen_outs) for game in bullpen_games]),
        "bullpen_last3_pitches_avg": average([float(game.bullpen_pitches) for game in bullpen_games]),
        "bullpen_last3_era": era(
            sum(game.bullpen_earned_runs for game in bullpen_games),
            sum(game.bullpen_outs for game in bullpen_games),
        ),
        "days_since_last_game": days_between(state.last_game_date, game_date),
    }


def update_team_state(state: TeamState, record: TeamGameRecord) -> None:
    state.games += 1
    state.wins += record.win
    state.runs_scored += record.runs_scored
    state.runs_allowed += record.runs_allowed
    state.batting_home_runs += record.batting_home_runs
    state.batting_walks += record.batting_walks
    state.batting_strikeouts += record.batting_strikeouts
    if record.batting_ops is not None:
        state.batting_ops_total += record.batting_ops
        state.batting_ops_games += 1
    state.stolen_bases += record.stolen_bases
    state.pitching_hits_allowed += record.pitching_hits_allowed
    state.pitching_walks_allowed += record.pitching_walks_allowed
    state.pitching_strikeouts += record.pitching_strikeouts
    state.pitching_earned_runs += record.pitching_earned_runs
    state.pitching_outs += record.pitching_outs
    state.errors += record.errors

    if record.is_home:
        state.home_games += 1
        state.home_wins += record.win
        state.home_runs_scored += record.runs_scored
        state.home_runs_allowed += record.runs_allowed
    else:
        state.away_games += 1
        state.away_wins += record.win
        state.away_runs_scored += record.runs_scored
        state.away_runs_allowed += record.runs_allowed

    state.last_game_date = record.date
    state.recent_games.append(record)
    state.recent_bullpen.append(record)


def extract_team_record(feed: dict[str, Any], side: str) -> TeamGameRecord:
    team_box = feed.get("liveData", {}).get("boxscore", {}).get("teams", {}).get(side, {})
    batting = team_box.get("teamStats", {}).get("batting", {})
    pitching = team_box.get("teamStats", {}).get("pitching", {})
    fielding = team_box.get("teamStats", {}).get("fielding", {})
    players = team_box.get("players", {})

    starter_outs = 0
    starter_pitches = 0
    starter_earned_runs = 0

    for player in players.values():
        pitching_stats = player.get("stats", {}).get("pitching", {})
        if safe_int(pitching_stats.get("gamesStarted")) == 1:
            starter_outs += safe_int(pitching_stats.get("outs"))
            starter_pitches += safe_int(pitching_stats.get("numberOfPitches"))
            starter_earned_runs += safe_int(pitching_stats.get("earnedRuns"))

    total_outs = safe_int(pitching.get("outs"))
    total_pitches = safe_int(pitching.get("numberOfPitches") or pitching.get("pitchesThrown"))
    total_earned_runs = safe_int(pitching.get("earnedRuns"))

    return TeamGameRecord(
        date=feed.get("gameData", {}).get("datetime", {}).get("officialDate", ""),
        is_home=(side == "home"),
        win=0,
        runs_scored=safe_int(batting.get("runs")),
        runs_allowed=safe_int(pitching.get("runs")),
        batting_ops=safe_float(batting.get("ops")),
        batting_home_runs=safe_int(batting.get("homeRuns")),
        batting_walks=safe_int(batting.get("baseOnBalls")),
        batting_strikeouts=safe_int(batting.get("strikeOuts")),
        stolen_bases=safe_int(batting.get("stolenBases")),
        errors=safe_int(fielding.get("errors")),
        pitching_hits_allowed=safe_int(pitching.get("hits")),
        pitching_walks_allowed=safe_int(pitching.get("baseOnBalls")),
        pitching_strikeouts=safe_int(pitching.get("strikeOuts")),
        pitching_earned_runs=total_earned_runs,
        pitching_outs=total_outs,
        bullpen_outs=max(0, total_outs - starter_outs),
        bullpen_pitches=max(0, total_pitches - starter_pitches),
        bullpen_earned_runs=max(0, total_earned_runs - starter_earned_runs),
    )


def extract_game_outcome(feed: dict[str, Any]) -> tuple[int, int]:
    linescore = feed.get("liveData", {}).get("linescore", {}).get("teams", {})
    home_runs = safe_int(linescore.get("home", {}).get("runs"))
    away_runs = safe_int(linescore.get("away", {}).get("runs"))
    return home_runs, away_runs


def build_pitcher_snapshot(player_id: int | None, season: int, before_date: str) -> dict[str, float | None]:
    if not player_id:
        return {
            "starts": None,
            "era": None,
            "whip": None,
            "k9": None,
            "bb9": None,
            "innings_per_start": None,
            "pitches_per_start": None,
            "last3_era": None,
            "last3_whip": None,
            "last3_k9": None,
            "days_since_last_start": None,
        }

    starts = []
    for split in fetch_pitcher_game_logs(player_id, season):
        if split.get("date", "") >= before_date:
            break
        stat = split.get("stat", {})
        if safe_int(stat.get("gamesStarted")) < 1:
            continue
        starts.append(
            {
                "date": split.get("date", ""),
                "outs": safe_int(stat.get("outs")),
                "earned_runs": safe_int(stat.get("earnedRuns")),
                "hits": safe_int(stat.get("hits")),
                "walks": safe_int(stat.get("baseOnBalls")),
                "strikeouts": safe_int(stat.get("strikeOuts")),
                "pitches": safe_int(stat.get("numberOfPitches")),
            }
        )

    if not starts:
        return {
            "starts": 0.0,
            "era": None,
            "whip": None,
            "k9": None,
            "bb9": None,
            "innings_per_start": None,
            "pitches_per_start": None,
            "last3_era": None,
            "last3_whip": None,
            "last3_k9": None,
            "days_since_last_start": None,
        }

    last3 = starts[-3:]
    total_outs = sum(start["outs"] for start in starts)
    total_er = sum(start["earned_runs"] for start in starts)
    total_hits = sum(start["hits"] for start in starts)
    total_walks = sum(start["walks"] for start in starts)
    total_strikeouts = sum(start["strikeouts"] for start in starts)
    total_pitches = sum(start["pitches"] for start in starts)

    return {
        "starts": float(len(starts)),
        "era": era(total_er, total_outs),
        "whip": ratio(total_hits + total_walks, total_outs / 3.0),
        "k9": per_nine(total_strikeouts, total_outs),
        "bb9": per_nine(total_walks, total_outs),
        "innings_per_start": ratio(total_outs / 3.0, len(starts)),
        "pitches_per_start": ratio(total_pitches, len(starts)),
        "last3_era": era(sum(start["earned_runs"] for start in last3), sum(start["outs"] for start in last3)),
        "last3_whip": ratio(
            sum(start["hits"] + start["walks"] for start in last3),
            sum(start["outs"] for start in last3) / 3.0,
        ),
        "last3_k9": per_nine(sum(start["strikeouts"] for start in last3), sum(start["outs"] for start in last3)),
        "days_since_last_start": days_between(starts[-1]["date"], before_date),
    }


def make_feature_row(
    game: dict[str, Any],
    team_states: dict[int, TeamState],
    season: int,
) -> dict[str, Any]:
    game_date = game.get("officialDate", "")
    home_team = game.get("teams", {}).get("home", {}).get("team", {})
    away_team = game.get("teams", {}).get("away", {}).get("team", {})
    home_team_id = safe_int(home_team.get("id"))
    away_team_id = safe_int(away_team.get("id"))

    row: dict[str, Any] = {
        "official_date": game_date,
        "game_datetime": game.get("gameDate"),
        "game_pk": safe_int(game.get("gamePk")),
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_team_name": home_team.get("name"),
        "away_team_name": away_team.get("name"),
    }

    for prefix, team_id, is_home in (("home", home_team_id, True), ("away", away_team_id, False)):
        snapshot = team_snapshot(team_states.setdefault(team_id, TeamState()), is_home, game_date)
        for feature_name, feature_value in snapshot.items():
            row[f"{prefix}_{feature_name}"] = feature_value

    home_pitcher_id = game.get("teams", {}).get("home", {}).get("probablePitcher", {}).get("id")
    away_pitcher_id = game.get("teams", {}).get("away", {}).get("probablePitcher", {}).get("id")
    for prefix, pitcher_id in (("home_pitcher", home_pitcher_id), ("away_pitcher", away_pitcher_id)):
        snapshot = build_pitcher_snapshot(safe_int(pitcher_id) or None, season, game_date)
        for feature_name, feature_value in snapshot.items():
            row[f"{prefix}_{feature_name}"] = feature_value

    return row


def process_completed_games_before(
    season: int,
    before_date: str | None = None,
) -> dict[int, TeamState]:
    states: dict[int, TeamState] = {}
    for game in fetch_schedule_for_season(season):
        if game.get("status", {}).get("codedGameState") != "F":
            continue
        game_date = game.get("officialDate", "")
        if before_date is not None and game_date >= before_date:
            continue

        feed = fetch_live_feed(safe_int(game.get("gamePk")))
        home_runs, away_runs = extract_game_outcome(feed)
        home_record = extract_team_record(feed, "home")
        away_record = extract_team_record(feed, "away")
        home_record.win = int(home_runs > away_runs)
        away_record.win = int(away_runs > home_runs)

        home_team_id = safe_int(game.get("teams", {}).get("home", {}).get("team", {}).get("id"))
        away_team_id = safe_int(game.get("teams", {}).get("away", {}).get("team", {}).get("id"))
        update_team_state(states.setdefault(home_team_id, TeamState()), home_record)
        update_team_state(states.setdefault(away_team_id, TeamState()), away_record)
    return states


def build_training_rows(season: int, before_date: str | None = None) -> list[dict[str, Any]]:
    states: dict[int, TeamState] = {}
    rows: list[dict[str, Any]] = []

    for game in fetch_schedule_for_season(season):
        if game.get("status", {}).get("codedGameState") != "F":
            continue
        if before_date is not None and game.get("officialDate", "") >= before_date:
            continue

        row = make_feature_row(game, states, season)
        feed = fetch_live_feed(safe_int(game.get("gamePk")))
        home_runs, away_runs = extract_game_outcome(feed)
        row["home_win"] = int(home_runs > away_runs)
        rows.append(row)

        home_record = extract_team_record(feed, "home")
        away_record = extract_team_record(feed, "away")
        home_record.win = int(home_runs > away_runs)
        away_record.win = int(away_runs > home_runs)

        home_team_id = safe_int(game.get("teams", {}).get("home", {}).get("team", {}).get("id"))
        away_team_id = safe_int(game.get("teams", {}).get("away", {}).get("team", {}).get("id"))
        update_team_state(states.setdefault(home_team_id, TeamState()), home_record)
        update_team_state(states.setdefault(away_team_id, TeamState()), away_record)

    return rows


def build_daily_prediction_rows(target_date: str, season: int | None = None) -> list[dict[str, Any]]:
    resolved_season = season or date.fromisoformat(target_date).year
    states = process_completed_games_before(resolved_season, before_date=target_date)

    rows: list[dict[str, Any]] = []
    for game in fetch_schedule_for_date(target_date):
        if game.get("status", {}).get("codedGameState") == "F":
            continue
        rows.append(make_feature_row(game, states, resolved_season))
    return rows


def write_rows_to_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No rows were generated.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
