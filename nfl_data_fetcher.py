"""Utility script for fetching NFL head-to-head data and updating the training set.

This script relies on the public ESPN schedule endpoint which does not require an
API key.  It prompts the user to choose the two teams that should be compared,
fetches completed match results for the requested seasons and appends them to
``data.txt`` in the ``score_team1,score_team2`` format expected by ``footballai.py``.

Once the data has been updated the script can optionally execute ``footballai.py``
to retrain the model immediately.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import json
from urllib import parse as _urlparse
from urllib import request as _urlrequest
from urllib.error import HTTPError, URLError


DATA_FILE = "data.txt"
FOOTBALL_AI_SCRIPT = "footballai.py"

SCHEDULE_CACHE_ENVVAR = "NFL_DATA_FETCHER_CACHE_DIR"

ESPN_SCHEDULE_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/schedule"
)
HTTP_HEADERS = {"User-Agent": "FootballAI-DataFetcher/1.0 (+https://github.com/)"}
DEFAULT_SEASON_SPAN = 5  # number of most-recent seasons fetched when no input is provided
DEFAULT_SEASON_TYPES = (2, 3)  # 2: regular season, 3: playoffs


@dataclass(frozen=True)
class Team:
    """Small helper structure describing an NFL franchise."""

    display_name: str
    abbreviation: str
    espn_id: str


TEAMS: Dict[int, Team] = {
    1: Team("Arizona Cardinals", "ARI", "22"),
    2: Team("Atlanta Falcons", "ATL", "1"),
    3: Team("Baltimore Ravens", "BAL", "33"),
    4: Team("Buffalo Bills", "BUF", "2"),
    5: Team("Carolina Panthers", "CAR", "29"),
    6: Team("Chicago Bears", "CHI", "3"),
    7: Team("Cincinnati Bengals", "CIN", "4"),
    8: Team("Cleveland Browns", "CLE", "5"),
    9: Team("Dallas Cowboys", "DAL", "6"),
    10: Team("Denver Broncos", "DEN", "7"),
    11: Team("Detroit Lions", "DET", "8"),
    12: Team("Green Bay Packers", "GB", "9"),
    13: Team("Houston Texans", "HOU", "34"),
    14: Team("Indianapolis Colts", "IND", "11"),
    15: Team("Jacksonville Jaguars", "JAX", "30"),
    16: Team("Kansas City Chiefs", "KC", "12"),
    17: Team("Las Vegas Raiders", "LV", "13"),
    18: Team("Los Angeles Chargers", "LAC", "24"),
    19: Team("Los Angeles Rams", "LAR", "14"),
    20: Team("Miami Dolphins", "MIA", "15"),
    21: Team("Minnesota Vikings", "MIN", "16"),
    22: Team("New England Patriots", "NE", "17"),
    23: Team("New Orleans Saints", "NO", "18"),
    24: Team("New York Giants", "NYG", "19"),
    25: Team("New York Jets", "NYJ", "20"),
    26: Team("Philadelphia Eagles", "PHI", "21"),
    27: Team("Pittsburgh Steelers", "PIT", "23"),
    28: Team("San Francisco 49ers", "SF", "25"),
    29: Team("Seattle Seahawks", "SEA", "26"),
    30: Team("Tampa Bay Buccaneers", "TB", "27"),
    31: Team("Tennessee Titans", "TEN", "10"),
    32: Team("Washington Commanders", "WAS", "28"),
}


class DataFetchError(RuntimeError):
    """Raised when the ESPN endpoint cannot be contacted or parsed."""


def _schedule_cache_path(
    cache_dir: Path, team_id: str, season: int | None, season_type: int | None
) -> Path:
    season_label = str(season) if season is not None else "current"
    season_type_label = str(season_type) if season_type is not None else "default"
    filename = f"{team_id}_{season_label}_{season_type_label}.json"
    return cache_dir / filename


def _request_schedule(
    team_id: str,
    season: int | None,
    season_type: int | None,
    cache_dir: Path | None = None,
) -> dict:
    """Retrieve the raw schedule payload for ``team_id``.

    Parameters
    ----------
    team_id:
        The ESPN team identifier (string form of an integer).
    season:
        The four-digit season year. ``None`` fetches the default (current) season.
    season_type:
        ``2`` for the regular season, ``3`` for playoffs. ``None`` lets ESPN decide.
    """

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = _schedule_cache_path(cache_dir, team_id, season, season_type)
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as cached:
                return json.load(cached)

    params = {}
    if season is not None:
        params["season"] = str(season)
    if season_type is not None:
        params["seasontype"] = str(season_type)

    query = _urlparse.urlencode(params)
    url = ESPN_SCHEDULE_URL.format(team_id=team_id)
    if query:
        url = f"{url}?{query}"

    request = _urlrequest.Request(url, headers=HTTP_HEADERS)
    try:
        with _urlrequest.urlopen(request, timeout=30) as response:
            payload = response.read()
    except (HTTPError, URLError) as exc:  # pragma: no cover - network failure path
        raise DataFetchError(
            f"Failed to contact ESPN schedule endpoint for team {team_id}: {exc}"
        ) from exc

    try:
        data = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - unexpected payload
        raise DataFetchError("Received invalid JSON from ESPN endpoint") from exc

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as cached:
            json.dump(data, cached, indent=2)

    return data


def _coerce_score(value: Any) -> int | None:
    """Best-effort conversion of ESPN score payloads to integers.

    The public schedule endpoint sometimes returns a simple string/number when a
    game has completed, but other times the ``score`` field is a nested
    dictionary containing the numeric value.  Future games may also emit an
    empty string, ``None`` or other placeholder structures.  This helper walks
    the common shapes and returns ``None`` whenever a reliable integer cannot be
    produced.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        # Guard against booleans (which are ints) sneaking through from unrelated
        # payload fields – treat them as missing values instead of 0/1 scores.
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None

    if isinstance(value, dict):
        for key in ("value", "displayValue", "score", "total"):
            if key in value:
                coerced = _coerce_score(value[key])
                if coerced is not None:
                    return coerced
        return None

    if isinstance(value, (list, tuple)):
        for item in value:
            coerced = _coerce_score(item)
            if coerced is not None:
                return coerced
        return None

    return None


def fetch_head_to_head(
    team_one: Team,
    team_two: Team,
    seasons: Sequence[int],
    season_types: Sequence[int] = DEFAULT_SEASON_TYPES,
    cache_dir: Path | None = None,
) -> List[Tuple[str, int, int, str, str]]:
    """Fetch completed matches between ``team_one`` and ``team_two``.

    Returns a list of tuples containing ``(event_id, score_one, score_two, event_name, date)``.
    """

    seen_events = set()
    results: List[Tuple[str, int, int, str, str]] = []

    for season in seasons:
        for season_type in season_types:
            payload = _request_schedule(
                team_one.espn_id, season, season_type, cache_dir=cache_dir
            )
            events = payload.get("events", [])
            for event in events:
                event_id = event.get("id")
                competitions = event.get("competitions", [])
                if not competitions:
                    continue
                competition = competitions[0]
                status = competition.get("status", {}).get("type", {})
                if not status.get("completed"):
                    continue
                competitors = competition.get("competitors", [])
                team_lookup = {
                    comp.get("team", {}).get("id"): comp for comp in competitors
                }
                team_one_entry = team_lookup.get(team_one.espn_id)
                team_two_entry = team_lookup.get(team_two.espn_id)
                if not team_one_entry or not team_two_entry:
                    # Opponent does not match or incomplete data
                    continue

                score_one = _coerce_score(team_one_entry.get("score"))
                score_two = _coerce_score(team_two_entry.get("score"))
                if score_one is None or score_two is None:
                    continue

                if event_id in seen_events:
                    continue
                seen_events.add(event_id)

                event_name = event.get("name", "")
                event_date = event.get("date", "")
                results.append((event_id, score_one, score_two, event_name, event_date))

    results.sort(key=lambda item: item[4])
    return results


def _ensure_data_file_exists(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")


def append_results_to_dataset(path: str, results: Iterable[Tuple[str, int, int, str, str]]) -> int:
    """Append ``results`` to ``path`` in ``score_one,score_two`` format.

    Returns the number of new rows that have been written.
    """

    _ensure_data_file_exists(path)

    # Preserve the file's trailing newline state so appends remain tidy.
    with open(path, "rb") as existing_file:
        existing_file.seek(0, os.SEEK_END)
        needs_newline = False
        if existing_file.tell() > 0:
            existing_file.seek(-1, os.SEEK_END)
            needs_newline = existing_file.read(1) != b"\n"

    rows_written = 0
    with open(path, "a", encoding="utf-8") as handle:
        if needs_newline:
            handle.write("\n")
        for index, (_, score_one, score_two, _, _) in enumerate(results):
            if index > 0:
                handle.write("\n")
            handle.write(f"{score_one},{score_two}")
            rows_written += 1
        if rows_written > 0:
            handle.write("\n")
    return rows_written


def _prompt_for_team(prompt: str) -> Team:
    while True:
        selection = input(prompt).strip()
        if not selection.isdigit():
            print("Please enter a numeric team identifier.")
            continue
        team_id = int(selection)
        team = TEAMS.get(team_id)
        if not team:
            print("Unknown team identifier. Please pick a number from the list above.")
            continue
        return team


def _default_seasons() -> List[int]:
    current_year = _dt.datetime.now().year
    return list(range(current_year - DEFAULT_SEASON_SPAN + 1, current_year + 1))


def parse_season_input(raw_input: str | None) -> List[int]:
    if not raw_input:
        return _default_seasons()

    raw_input = raw_input.strip()
    if not raw_input:
        return _default_seasons()

    if "-" in raw_input:
        try:
            start, end = [int(part) for part in raw_input.split("-", 1)]
        except ValueError:
            raise ValueError("Season range must look like 2018-2023")
        if start > end:
            start, end = end, start
        return list(range(start, end + 1))

    seasons = []
    for part in raw_input.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            seasons.append(int(part))
        except ValueError as exc:
            raise ValueError(
                "Season list must contain integers separated by commas"
            ) from exc
    if not seasons:
        raise ValueError("No valid season values were provided")
    return seasons


def display_team_table() -> None:
    print("Available NFL Teams:")
    print("-------------------")
    for identifier, team in TEAMS.items():
        print(f"{identifier:>2}: {team.display_name} ({team.abbreviation})")
    print()


def run_football_ai_script() -> None:
    print("\nLaunching footballai.py for training/prediction...\n")
    try:
        subprocess.run([sys.executable, FOOTBALL_AI_SCRIPT], check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - subprocess failure
        print(f"footballai.py exited with a non-zero status code: {exc.returncode}")
    except FileNotFoundError:  # pragma: no cover - missing interpreter/script
        print("Unable to locate footballai.py. Please ensure the script exists next to this file.")


def _parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch historical NFL scores for two teams and append the results to data.txt. "
            "If no arguments are supplied the script runs in interactive mode."
        )
    )
    parser.add_argument("--team1", type=int, help="Numeric identifier for Team 1")
    parser.add_argument("--team2", type=int, help="Numeric identifier for Team 2")
    parser.add_argument(
        "--seasons",
        type=str,
        help=(
            "Season selection. Accepts comma-separated years (e.g. 2020,2021) or "
            "a range (e.g. 2018-2023). Defaults to the last five seasons."
        ),
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Do not execute footballai.py after updating the dataset.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help=(
            "Optional directory containing cached ESPN schedule responses. "
            "If provided (or set via the NFL_DATA_FETCHER_CACHE_DIR environment "
            "variable), the script will prefer cached data and update the cache "
            "whenever a network call succeeds."
        ),
    )
    return parser.parse_args(argv)


def _team_from_argument(value: int | None, argument_name: str) -> Team | None:
    if value is None:
        return None
    team = TEAMS.get(value)
    if team is None:
        valid_choices = ", ".join(str(key) for key in sorted(TEAMS))
        raise SystemExit(
            f"Unknown value '{value}' for {argument_name}. Choose one of: {valid_choices}"
        )
    return team


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_arguments(argv or sys.argv[1:])

    team_one = _team_from_argument(args.team1, "--team1")
    team_two = _team_from_argument(args.team2, "--team2")

    if not team_one or not team_two:
        display_team_table()
        if not team_one:
            team_one = _prompt_for_team("Select Team 1 by number: ")
        if not team_two:
            team_two = _prompt_for_team("Select Team 2 by number: ")

    cache_dir_value = args.cache_dir or os.environ.get(SCHEDULE_CACHE_ENVVAR)
    cache_dir = Path(cache_dir_value).expanduser() if cache_dir_value else None

    try:
        seasons = parse_season_input(args.seasons)
    except ValueError as exc:
        raise SystemExit(str(exc))

    print(
        f"\nFetching head-to-head results for {team_one.display_name} vs. {team_two.display_name}"
    )
    print(f"Seasons: {', '.join(str(season) for season in seasons)}")

    try:
        results = fetch_head_to_head(
            team_one, team_two, seasons, cache_dir=cache_dir
        )
    except DataFetchError as exc:
        raise SystemExit(str(exc))

    if not results:
        print("No completed games were found for the selected teams and seasons.")
        return

    print(f"Found {len(results)} completed games. Appending to {DATA_FILE}...")
    written = append_results_to_dataset(DATA_FILE, results)
    print(f"Added {written} new rows to {DATA_FILE}.")

    print("\nSummary of appended games:")
    for _, score_one, score_two, event_name, event_date in results:
        print(f"  {event_date[:10]} - {event_name}: {score_one}-{score_two}")

    if not args.skip_train:
        run_training = input("\nRun footballai.py now? [Y/n]: ").strip().lower()
        if run_training in {"", "y", "yes"}:
            run_football_ai_script()
        else:
            print("Skipping model training.")


if __name__ == "__main__":
    main()
