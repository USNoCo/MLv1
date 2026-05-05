import csv
import hashlib
import json
from collections import Counter, deque
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
BVP_HISTORY_SEASONS = 3


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


def fetch_json(
    url: str,
    params: dict[str, Any],
    cache_group: str,
    cache_key: str,
    use_cache: bool = True,
) -> dict[str, Any]:
    path = cache_path(cache_group, cache_key)
    if use_cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if use_cache:
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


def fetch_schedule_for_date(target_date: str, use_cache: bool = True) -> list[dict[str, Any]]:
    data = fetch_json(
        f"{BASE_URL}/schedule",
        {
            "sportId": SPORT_ID,
            "date": target_date,
            "gameTypes": REGULAR_SEASON_GAME_TYPE,
            "hydrate": "probablePitcher,team,lineups,linescore",
        },
        "schedule",
        f"date-{target_date}",
        use_cache=use_cache,
    )
    games: list[dict[str, Any]] = []
    for date_block in data.get("dates", []):
        games.extend(date_block.get("games", []))
    return sorted(games, key=lambda game: (game.get("officialDate", ""), game.get("gamePk", 0)))


def game_has_not_started(game: dict[str, Any]) -> bool:
    status = game.get("status", {}) or {}
    abstract_state = status.get("abstractGameState")
    abstract_code = status.get("abstractGameCode")
    detailed_state = str(status.get("detailedState", "")).lower()

    if abstract_state != "Preview" and abstract_code != "P":
        return False

    unavailable_states = ("postponed", "cancelled", "canceled", "suspended", "final", "completed")
    return not any(state in detailed_state for state in unavailable_states)


def fetch_live_feed(game_pk: int, use_cache: bool = True) -> dict[str, Any]:
    return fetch_json(
        f"{LIVE_FEED_URL}/{game_pk}/feed/live",
        {},
        "game_feeds",
        f"game-{game_pk}",
        use_cache=use_cache,
    )


def fetch_team_roster(team_id: int, season: int, use_cache: bool = True) -> list[dict[str, Any]]:
    data = fetch_json(
        f"{BASE_URL}/teams/{team_id}/roster",
        {"rosterType": "40Man", "season": season},
        "team_rosters",
        f"team-{team_id}-season-{season}-40man",
        use_cache=use_cache,
    )
    return data.get("roster", [])


def fetch_injured_player_ids(team_id: int, season: int, use_cache: bool = True) -> set[int]:
    injured_ids: set[int] = set()
    for roster_entry in fetch_team_roster(team_id, season, use_cache=use_cache):
        status_description = (roster_entry.get("status", {}) or {}).get("description", "")
        if "Injured" not in status_description:
            continue
        player_id = safe_int(roster_entry.get("person", {}).get("id"))
        if player_id:
            injured_ids.add(player_id)
    return injured_ids


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
    recent_batting_orders: deque[list[int]] = field(default_factory=lambda: deque(maxlen=10))
    seen_batter_ids: set[int] = field(default_factory=set)


@dataclass
class PlayerBattingState:
    games: int = 0
    starts: int = 0
    plate_appearances: int = 0
    at_bats: int = 0
    hits: int = 0
    doubles: int = 0
    triples: int = 0
    home_runs: int = 0
    walks: int = 0
    hit_by_pitch: int = 0
    sac_flies: int = 0
    total_bases: int = 0

    @property
    def obp(self) -> float | None:
        denominator = self.at_bats + self.walks + self.hit_by_pitch + self.sac_flies
        return ratio(self.hits + self.walks + self.hit_by_pitch, denominator)

    @property
    def slg(self) -> float | None:
        return ratio(self.total_bases, self.at_bats)

    @property
    def ops(self) -> float | None:
        obp = self.obp
        slg = self.slg
        if obp is None or slg is None:
            return None
        return obp + slg


@dataclass
class BatterPitcherMatchupState:
    games: set[int] = field(default_factory=set)
    plate_appearances: int = 0
    at_bats: int = 0
    hits: int = 0
    doubles: int = 0
    triples: int = 0
    home_runs: int = 0
    walks: int = 0
    hit_by_pitch: int = 0
    sac_flies: int = 0
    strikeouts: int = 0
    total_bases: int = 0

    @property
    def obp(self) -> float | None:
        denominator = self.at_bats + self.walks + self.hit_by_pitch + self.sac_flies
        return ratio(self.hits + self.walks + self.hit_by_pitch, denominator)

    @property
    def slg(self) -> float | None:
        return ratio(self.total_bases, self.at_bats)

    @property
    def ops(self) -> float | None:
        obp = self.obp
        slg = self.slg
        if obp is None or slg is None:
            return None
        return obp + slg


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


def last_season_team_snapshot(state: TeamState | None, as_home_team: bool) -> dict[str, float | None]:
    if state is None:
        return {
            "games_played": 0.0,
            "win_pct": None,
            "runs_scored_avg": None,
            "runs_allowed_avg": None,
            "run_diff_avg": None,
            "batting_home_runs_avg": None,
            "batting_walks_avg": None,
            "batting_strikeouts_avg": None,
            "batting_ops_avg": None,
            "stolen_bases_avg": None,
            "pitching_era": None,
            "pitching_whip": None,
            "pitching_k9": None,
            "pitching_bb9": None,
            "errors_avg": None,
            "venue_split_win_pct": None,
            "venue_split_run_diff_avg": None,
        }

    split_games = state.home_games if as_home_team else state.away_games
    split_wins = state.home_wins if as_home_team else state.away_wins
    split_runs_scored = state.home_runs_scored if as_home_team else state.away_runs_scored
    split_runs_allowed = state.home_runs_allowed if as_home_team else state.away_runs_allowed

    return {
        "games_played": float(state.games),
        "win_pct": ratio(state.wins, state.games),
        "runs_scored_avg": ratio(state.runs_scored, state.games),
        "runs_allowed_avg": ratio(state.runs_allowed, state.games),
        "run_diff_avg": ratio(state.runs_scored - state.runs_allowed, state.games),
        "batting_home_runs_avg": ratio(state.batting_home_runs, state.games),
        "batting_walks_avg": ratio(state.batting_walks, state.games),
        "batting_strikeouts_avg": ratio(state.batting_strikeouts, state.games),
        "batting_ops_avg": ratio(state.batting_ops_total, state.batting_ops_games),
        "stolen_bases_avg": ratio(state.stolen_bases, state.games),
        "pitching_era": era(state.pitching_earned_runs, state.pitching_outs),
        "pitching_whip": ratio(state.pitching_hits_allowed + state.pitching_walks_allowed, state.pitching_outs / 3.0),
        "pitching_k9": per_nine(state.pitching_strikeouts, state.pitching_outs),
        "pitching_bb9": per_nine(state.pitching_walks_allowed, state.pitching_outs),
        "errors_avg": ratio(state.errors, state.games),
        "venue_split_win_pct": ratio(split_wins, split_games),
        "venue_split_run_diff_avg": ratio(split_runs_scored - split_runs_allowed, split_games),
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


def extract_batting_order(feed: dict[str, Any], side: str) -> list[int]:
    raw_order = feed.get("liveData", {}).get("boxscore", {}).get("teams", {}).get(side, {}).get("battingOrder", [])
    return [safe_int(player_id) for player_id in raw_order if safe_int(player_id)]


def record_team_batters(state: TeamState, feed: dict[str, Any], side: str) -> None:
    order = extract_batting_order(feed, side)
    if order:
        state.recent_batting_orders.append(order)
        state.seen_batter_ids.update(order)

    players = feed.get("liveData", {}).get("boxscore", {}).get("teams", {}).get(side, {}).get("players", {})
    for player_key, player in players.items():
        batting = player.get("stats", {}).get("batting", {})
        if not batting:
            continue
        player_id = safe_int(player.get("person", {}).get("id")) or safe_int(str(player_key).replace("ID", ""))
        if player_id:
            state.seen_batter_ids.add(player_id)


def update_player_batting_states(
    player_states: dict[int, PlayerBattingState],
    feed: dict[str, Any],
    side: str,
) -> None:
    batting_order = set(extract_batting_order(feed, side))
    players = feed.get("liveData", {}).get("boxscore", {}).get("teams", {}).get(side, {}).get("players", {})

    for player_key, player in players.items():
        batting = player.get("stats", {}).get("batting", {})
        if not batting:
            continue

        player_id = safe_int(player.get("person", {}).get("id")) or safe_int(str(player_key).replace("ID", ""))
        if not player_id:
            continue

        plate_appearances = safe_int(batting.get("plateAppearances"))
        at_bats = safe_int(batting.get("atBats"))
        walks = safe_int(batting.get("baseOnBalls"))
        hit_by_pitch = safe_int(batting.get("hitByPitch"))
        sac_flies = safe_int(batting.get("sacFlies"))
        if plate_appearances == 0:
            plate_appearances = at_bats + walks + hit_by_pitch + sac_flies
        if plate_appearances == 0 and at_bats == 0:
            continue

        hits = safe_int(batting.get("hits"))
        doubles = safe_int(batting.get("doubles"))
        triples = safe_int(batting.get("triples"))
        home_runs = safe_int(batting.get("homeRuns"))
        total_bases = safe_int(batting.get("totalBases"))
        if total_bases == 0 and hits:
            singles = max(0, hits - doubles - triples - home_runs)
            total_bases = singles + (2 * doubles) + (3 * triples) + (4 * home_runs)

        state = player_states.setdefault(player_id, PlayerBattingState())
        state.games += 1
        state.starts += int(player_id in batting_order)
        state.plate_appearances += plate_appearances
        state.at_bats += at_bats
        state.hits += hits
        state.doubles += doubles
        state.triples += triples
        state.home_runs += home_runs
        state.walks += walks
        state.hit_by_pitch += hit_by_pitch
        state.sac_flies += sac_flies
        state.total_bases += total_bases


def update_lineup_history(
    team_state: TeamState,
    player_states: dict[int, PlayerBattingState],
    feed: dict[str, Any],
    side: str,
) -> None:
    record_team_batters(team_state, feed, side)
    update_player_batting_states(player_states, feed, side)


def project_lineup(team_state: TeamState, unavailable_player_ids: set[int]) -> tuple[list[int], int]:
    start_counts: Counter[int] = Counter()
    recency: dict[int, int] = {}

    for index, order in enumerate(team_state.recent_batting_orders):
        for player_id in order:
            start_counts[player_id] += 1
            recency[player_id] = index

    ranked_players = sorted(
        start_counts,
        key=lambda player_id: (start_counts[player_id], recency.get(player_id, -1)),
        reverse=True,
    )
    unavailable_regulars = sum(1 for player_id in ranked_players[:9] if player_id in unavailable_player_ids)
    projected = [player_id for player_id in ranked_players if player_id not in unavailable_player_ids][:9]

    if len(projected) < 9:
        fallback_players = [
            player_id
            for player_id in team_state.seen_batter_ids
            if player_id not in unavailable_player_ids and player_id not in projected
        ]
        projected.extend(fallback_players[: 9 - len(projected)])

    return projected, unavailable_regulars


def lineup_snapshot(
    team_state: TeamState,
    player_states: dict[int, PlayerBattingState],
    lineup_feed: dict[str, Any] | None,
    side: str,
    unavailable_player_ids: set[int],
) -> dict[str, float | None]:
    lineup, confirmed, unavailable_regulars = resolve_lineup(team_state, lineup_feed, side, unavailable_player_ids)

    hitter_states = [
        player_states[player_id]
        for player_id in lineup
        if player_id in player_states and player_states[player_id].plate_appearances > 0
    ]
    ops_values = [state.ops for state in hitter_states if state.ops is not None]
    obp_values = [state.obp for state in hitter_states if state.obp is not None]
    slg_values = [state.slg for state in hitter_states if state.slg is not None]

    return {
        "lineup_confirmed": confirmed,
        "lineup_batters": float(len(lineup)),
        "lineup_unavailable_regulars": float(unavailable_regulars),
        "lineup_avg_ops": average(ops_values),
        "lineup_avg_obp": average(obp_values),
        "lineup_avg_slg": average(slg_values),
        "lineup_avg_plate_appearances": average([float(state.plate_appearances) for state in hitter_states]),
        "lineup_avg_starts": average([float(state.starts) for state in hitter_states]),
    }


def resolve_lineup(
    team_state: TeamState,
    lineup_feed: dict[str, Any] | None,
    side: str,
    unavailable_player_ids: set[int],
) -> tuple[list[int], float, int]:
    confirmed_order = extract_batting_order(lineup_feed, side) if lineup_feed else []
    if confirmed_order:
        return confirmed_order, 1.0, 0

    lineup, unavailable_regulars = project_lineup(team_state, unavailable_player_ids)
    return lineup, 0.0, unavailable_regulars


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


def empty_pitcher_snapshot() -> dict[str, float | None]:
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


def build_pitcher_snapshot(player_id: int | None, season: int, before_date: str | None = None) -> dict[str, float | None]:
    if not player_id:
        return empty_pitcher_snapshot()

    starts = []
    for split in fetch_pitcher_game_logs(player_id, season):
        if before_date is not None and split.get("date", "") >= before_date:
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
        "days_since_last_start": days_between(starts[-1]["date"], before_date) if before_date is not None else None,
    }


def empty_bvp_snapshot() -> dict[str, float | None]:
    return {
        "known_batters": 0.0,
        "games": 0.0,
        "plate_appearances": 0.0,
        "at_bats": 0.0,
        "avg": None,
        "obp": None,
        "slg": None,
        "ops": None,
        "home_runs": 0.0,
        "strikeout_rate": None,
        "walk_rate": None,
    }


def bvp_snapshot_from_states(
    lineup: list[int],
    pitcher_id: int | None,
    matchup_states: dict[tuple[int, int], BatterPitcherMatchupState] | None,
) -> dict[str, float | None]:
    if not lineup or not pitcher_id or matchup_states is None:
        return empty_bvp_snapshot()

    states = [matchup_states[(batter_id, pitcher_id)] for batter_id in lineup if (batter_id, pitcher_id) in matchup_states]
    return aggregate_bvp_states(states)


def aggregate_bvp_states(states: list[BatterPitcherMatchupState]) -> dict[str, float | None]:
    if not states:
        return empty_bvp_snapshot()

    games: set[int] = set()
    for state in states:
        games.update(state.games)

    plate_appearances = sum(state.plate_appearances for state in states)
    at_bats = sum(state.at_bats for state in states)
    hits = sum(state.hits for state in states)
    walks = sum(state.walks for state in states)
    hit_by_pitch = sum(state.hit_by_pitch for state in states)
    sac_flies = sum(state.sac_flies for state in states)
    total_bases = sum(state.total_bases for state in states)
    strikeouts = sum(state.strikeouts for state in states)

    obp_denominator = at_bats + walks + hit_by_pitch + sac_flies
    obp = ratio(hits + walks + hit_by_pitch, obp_denominator)
    slg = ratio(total_bases, at_bats)
    return {
        "known_batters": float(len(states)),
        "games": float(len(games)),
        "plate_appearances": float(plate_appearances),
        "at_bats": float(at_bats),
        "avg": ratio(hits, at_bats),
        "obp": obp,
        "slg": slg,
        "ops": None if obp is None or slg is None else obp + slg,
        "home_runs": float(sum(state.home_runs for state in states)),
        "strikeout_rate": ratio(strikeouts, plate_appearances),
        "walk_rate": ratio(walks, plate_appearances),
    }


def update_batter_pitcher_matchups(
    matchup_states: dict[tuple[int, int], BatterPitcherMatchupState],
    feed: dict[str, Any],
) -> None:
    game_pk = safe_int(feed.get("gamePk")) or safe_int(feed.get("gameData", {}).get("game", {}).get("pk"))
    for play in feed.get("liveData", {}).get("plays", {}).get("allPlays", []):
        about = play.get("about", {})
        if about and about.get("isComplete") is False:
            continue
        matchup = play.get("matchup", {})
        batter_id = safe_int(matchup.get("batter", {}).get("id"))
        pitcher_id = safe_int(matchup.get("pitcher", {}).get("id"))
        if not batter_id or not pitcher_id:
            continue
        event_type = play.get("result", {}).get("eventType", "")
        apply_plate_appearance(
            matchup_states.setdefault((batter_id, pitcher_id), BatterPitcherMatchupState()),
            event_type,
            game_pk,
        )


def apply_plate_appearance(state: BatterPitcherMatchupState, event_type: str, game_pk: int) -> None:
    if not event_type:
        return

    non_pa_events = {"pickoff_1b", "pickoff_2b", "pickoff_3b", "caught_stealing_2b", "caught_stealing_3b"}
    if event_type in non_pa_events:
        return

    state.games.add(game_pk)
    state.plate_appearances += 1

    hit_bases = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
    if event_type in hit_bases:
        state.at_bats += 1
        state.hits += 1
        state.total_bases += hit_bases[event_type]
        state.doubles += int(event_type == "double")
        state.triples += int(event_type == "triple")
        state.home_runs += int(event_type == "home_run")
        return

    if event_type in {"walk", "intent_walk", "intentional_walk"}:
        state.walks += 1
        return
    if event_type == "hit_by_pitch":
        state.hit_by_pitch += 1
        return
    if event_type == "sac_fly":
        state.sac_flies += 1
        return
    if event_type == "catcher_interf":
        return

    if event_type in {"strikeout", "strikeout_double_play"}:
        state.strikeouts += 1
    if event_type != "sac_bunt":
        state.at_bats += 1


def make_feature_row(
    game: dict[str, Any],
    team_states: dict[int, TeamState],
    player_states: dict[int, PlayerBattingState],
    season: int,
    batter_pitcher_states: dict[tuple[int, int], BatterPitcherMatchupState] | None = None,
    last_season_team_states: dict[int, TeamState] | None = None,
    lineup_feed: dict[str, Any] | None = None,
    use_current_injuries: bool = False,
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

    home_pitcher_id = safe_int(game.get("teams", {}).get("home", {}).get("probablePitcher", {}).get("id")) or None
    away_pitcher_id = safe_int(game.get("teams", {}).get("away", {}).get("probablePitcher", {}).get("id")) or None

    for prefix, team_id, is_home in (("home", home_team_id, True), ("away", away_team_id, False)):
        team_state = team_states.setdefault(team_id, TeamState())
        snapshot = team_snapshot(team_state, is_home, game_date)
        for feature_name, feature_value in snapshot.items():
            row[f"{prefix}_{feature_name}"] = feature_value

        last_season_state = (last_season_team_states or {}).get(team_id)
        last_season_snapshot = last_season_team_snapshot(last_season_state, is_home)
        for feature_name, feature_value in last_season_snapshot.items():
            row[f"{prefix}_last_season_{feature_name}"] = feature_value

        side = "home" if is_home else "away"
        unavailable_player_ids = (
            fetch_injured_player_ids(team_id, season, use_cache=False) if use_current_injuries and team_id else set()
        )
        lineup_features = lineup_snapshot(team_state, player_states, lineup_feed, side, unavailable_player_ids)
        for feature_name, feature_value in lineup_features.items():
            row[f"{prefix}_{feature_name}"] = feature_value

        lineup, _, _ = resolve_lineup(team_state, lineup_feed, side, unavailable_player_ids)
        opposing_pitcher_id = away_pitcher_id if is_home else home_pitcher_id
        bvp_features = bvp_snapshot_from_states(lineup, opposing_pitcher_id, batter_pitcher_states)
        opposing_prefix = "away_pitcher" if is_home else "home_pitcher"
        for feature_name, feature_value in bvp_features.items():
            row[f"{prefix}_bvp_vs_{opposing_prefix}_{feature_name}"] = feature_value

    for prefix, pitcher_id in (("home_pitcher", home_pitcher_id), ("away_pitcher", away_pitcher_id)):
        snapshot = build_pitcher_snapshot(pitcher_id, season, game_date)
        for feature_name, feature_value in snapshot.items():
            row[f"{prefix}_{feature_name}"] = feature_value

        last_season_snapshot = build_pitcher_snapshot(pitcher_id, season - 1)
        for feature_name, feature_value in last_season_snapshot.items():
            if feature_name == "days_since_last_start":
                continue
            row[f"{prefix}_last_season_{feature_name}"] = feature_value

    return row


def process_completed_games_before(
    season: int,
    before_date: str | None = None,
) -> dict[int, TeamState]:
    states, _ = process_completed_games_and_players_before(season, before_date=before_date)
    return states


def process_completed_games_and_players_before(
    season: int,
    before_date: str | None = None,
    batter_pitcher_states: dict[tuple[int, int], BatterPitcherMatchupState] | None = None,
) -> tuple[dict[int, TeamState], dict[int, PlayerBattingState]]:
    states: dict[int, TeamState] = {}
    player_states: dict[int, PlayerBattingState] = {}
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
        home_state = states.setdefault(home_team_id, TeamState())
        away_state = states.setdefault(away_team_id, TeamState())
        update_team_state(home_state, home_record)
        update_team_state(away_state, away_record)
        update_lineup_history(home_state, player_states, feed, "home")
        update_lineup_history(away_state, player_states, feed, "away")
        if batter_pitcher_states is not None:
            update_batter_pitcher_matchups(batter_pitcher_states, feed)
    return states, player_states


def process_bvp_history_before(
    season: int,
    before_date: str | None = None,
    lookback_seasons: int = BVP_HISTORY_SEASONS,
) -> dict[tuple[int, int], BatterPitcherMatchupState]:
    matchup_states: dict[tuple[int, int], BatterPitcherMatchupState] = {}
    start_season = max(1876, season - lookback_seasons)
    for history_season in range(start_season, season + 1):
        cutoff = before_date if history_season == season else None
        for game in fetch_schedule_for_season(history_season):
            if game.get("status", {}).get("codedGameState") != "F":
                continue
            game_date = game.get("officialDate", "")
            if cutoff is not None and game_date >= cutoff:
                continue
            feed = fetch_live_feed(safe_int(game.get("gamePk")))
            update_batter_pitcher_matchups(matchup_states, feed)
    return matchup_states


def build_training_rows(season: int, before_date: str | None = None) -> list[dict[str, Any]]:
    states: dict[int, TeamState] = {}
    player_states: dict[int, PlayerBattingState] = {}
    batter_pitcher_states = process_bvp_history_before(season, before_date=f"{season}-01-01")
    last_season_team_states = process_completed_games_before(season - 1)
    rows: list[dict[str, Any]] = []

    for game in fetch_schedule_for_season(season):
        if game.get("status", {}).get("codedGameState") != "F":
            continue
        if before_date is not None and game.get("officialDate", "") >= before_date:
            continue

        feed = fetch_live_feed(safe_int(game.get("gamePk")))
        row = make_feature_row(
            game,
            states,
            player_states,
            season,
            batter_pitcher_states=batter_pitcher_states,
            last_season_team_states=last_season_team_states,
            lineup_feed=feed,
        )
        home_runs, away_runs = extract_game_outcome(feed)
        row["home_win"] = int(home_runs > away_runs)
        rows.append(row)

        home_record = extract_team_record(feed, "home")
        away_record = extract_team_record(feed, "away")
        home_record.win = int(home_runs > away_runs)
        away_record.win = int(away_runs > home_runs)

        home_team_id = safe_int(game.get("teams", {}).get("home", {}).get("team", {}).get("id"))
        away_team_id = safe_int(game.get("teams", {}).get("away", {}).get("team", {}).get("id"))
        home_state = states.setdefault(home_team_id, TeamState())
        away_state = states.setdefault(away_team_id, TeamState())
        update_team_state(home_state, home_record)
        update_team_state(away_state, away_record)
        update_lineup_history(home_state, player_states, feed, "home")
        update_lineup_history(away_state, player_states, feed, "away")
        update_batter_pitcher_matchups(batter_pitcher_states, feed)

    return rows


def build_daily_prediction_rows(target_date: str, season: int | None = None) -> list[dict[str, Any]]:
    resolved_season = season or date.fromisoformat(target_date).year
    batter_pitcher_states = process_bvp_history_before(resolved_season, before_date=f"{resolved_season}-01-01")
    states, player_states = process_completed_games_and_players_before(
        resolved_season,
        before_date=target_date,
        batter_pitcher_states=batter_pitcher_states,
    )
    last_season_team_states = process_completed_games_before(resolved_season - 1)

    rows: list[dict[str, Any]] = []
    for game in fetch_schedule_for_date(target_date, use_cache=False):
        if not game_has_not_started(game):
            continue
        lineup_feed = fetch_live_feed(safe_int(game.get("gamePk")), use_cache=False)
        rows.append(
            make_feature_row(
                game,
                states,
                player_states,
                resolved_season,
                batter_pitcher_states=batter_pitcher_states,
                last_season_team_states=last_season_team_states,
                lineup_feed=lineup_feed,
                use_current_injuries=True,
            )
        )
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
