
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
white_shorts.etl.update_history_from_api
Fetches FINAL daily player results from SportsData.io and merges into a history CSV.
This is intended to *replace* maintaining a rolling YTD file manually.

Usage examples:
  python -m white_shorts.etl.update_history_from_api \

    --history_csv data/NHL_HISTORY_UNION.csv \

    --since_days 2 \

    --key $SPORTSDATA_API_KEY

  python -m white_shorts.etl.update_history_from_api \

    --history_csv data/NHL_HISTORY_UNION.csv \

    --start 2025-04-01 --end 2025-05-01 \

    --key $SPORTSDATA_API_KEY
"""
from __future__ import annotations
import argparse, os, sys, time, json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd
import numpy as np

try:
    import requests
except Exception:
    requests = None

DATE_FMT = "%Y-%m-%d"

def dprint(*a):
    print("[update_history]", *a, flush=True)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history_csv", required=True, help="Existing union history CSV to update in-place.")
    win = ap.add_mutually_exclusive_group(required=True)
    win.add_argument("--since_days", type=int, help="Fetch results for N days back from today (inclusive).")
    win.add_argument("--start", help="Start date YYYY-MM-DD")
    ap.add_argument("--end", help="End date YYYY-MM-DD (inclusive). If omitted with --start, uses today.")
    ap.add_argument("--key", default=None, help="SportsData.io API key (or env SPORTSDATA_API_KEY)")
    ap.add_argument("--rate_delay", type=float, default=1.1, help="seconds between API calls")
    ap.add_argument("--write", default=None, help="Optional alt output path (defaults to --history_csv)")
    ap.add_argument("--backup", action="store_true", help="Write a .bak copy before overwriting history")
    ap.add_argument("--dry_run", action="store_true", help="Do not write; just print counts")
    return ap.parse_args()

def ensure_key(args):
    key = args.key or os.environ.get("SPORTSDATA_API_KEY")
    if not key:
        raise SystemExit("No API key. Pass --key or set env SPORTSDATA_API_KEY")
    return key

def date_iter(start: datetime, end: datetime):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def to_bool_home(home_or_away: str) -> int:
    s = str(home_or_away or "").strip().upper()
    if s == "HOME": return 1
    if s == "AWAY": return 0
    return 0

def normalize_result_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Columns we expect from SportsData.io PlayerGameStatsByDate
    # Robust mapping to our schema
    cols = {
        "Name": "name",
        "Team": "team",
        "Opponent": "opponent",
        "HomeOrAway": "home_or_away_str",
        "Day": "day",
        "DateTime": "datetime",
        "Minutes": "minutes",
        # scoring
        "Points": "points",
        "Goals": "goals",
        "Assists": "assists",
        # game status
        "IsGameOver": "is_over",
        "Updated": "updated_ts",
        "PlayerID": "player_id",
        "GameID": "game_id",
        "GoaltendingGoalsAgainst": "goal_tending_goals_against",
        "PowerPlayGoals": "power_play_goals",
        "PowerPlayAssists": "power_play_assists",
        "ShotsOnGoal": "shots_on_goal"
        
    }
    keep = [c for c in cols if c in df.columns]
    out = df[keep].rename(columns=cols).copy()

    # Choose date: prefer 'day', fallback to 'datetime'
    if "day" in out.columns and out["day"].notna().any():
        out["date"] = pd.to_datetime(out["day"], errors="coerce").dt.date.astype(str)
    else:
        out["date"] = pd.to_datetime(out.get("datetime"), errors="coerce").dt.date.astype(str)

    # Home/Away bool -> home_or_away (1=HOME,0=AWAY)
    out["home_or_away"] = out["home_or_away_str"].apply(to_bool_home).astype(int)

    # fill zeros
    out["points"] = pd.to_numeric(out["points"], errors="coerce").fillna(0).astype(float)
    out["assists"] = pd.to_numeric(out["assists"], errors="coerce").fillna(0).astype(float)
    out["goals"] = pd.to_numeric(out["goals"], errors="coerce").fillna(0).astype(float)
    out["goal_tending_goals_against"] = pd.to_numeric(out["goal_tending_goals_against"], errors="coerce").fillna(0).astype(float)
    out["power_play_goals"] = pd.to_numeric(out["power_play_goals"], errors="coerce").fillna(0).astype(float)
    out["power_play_assists"] = pd.to_numeric(out["power_play_assists"], errors="coerce").fillna(0).astype(float)
    out["shots_on_goal"] = pd.to_numeric(out["shots_on_goal"], errors="coerce").fillna(0).astype(float)

   
    #player_id,name,date,minutes,points,goals,assists,home_or_away,gap_in_days
    # Select final columns in our schema
    final = out[["game_id","team","opponent","player_id","name","date","minutes","points","goals","assists","home_or_away","shots_on_goal","power_play_assists","power_play_goals","goal_tending_goals_against"]].copy()
    final = final.dropna(subset=["name","team","date"])
    return final

def fetch_results_by_date(date_mon: str, key: str, tries: int = 3, timeout: int = 45) -> pd.DataFrame:
    if requests is None:
        raise SystemExit("The 'requests' package is unavailable; install it locally to call the API.")

    base = "https://api.sportsdata.io/api/nhl/fantasy/json/PlayerGameStatsByDate"
    url = f"{base}/{date_mon}?key={key}"

    last_err = None
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                # Gracefully coerce to empty DF when payload is unexpected
                return pd.DataFrame([])
            return pd.DataFrame(data)
        except requests.RequestException as e:
            last_err = e
            # backoff: 2s, 4s, 8s ...
            time.sleep(2 ** attempt)
    # If we exhausted retries, re-raise so caller can decide
    raise last_err


def to_mon_date(d: datetime) -> str:
    return d.strftime("%Y-%b-%d")  # e.g., 2025-May-07

def main():
    args = parse_args()
    key = ensure_key(args)

     
    # Build date range
    today = datetime.utcnow().date()
    if args.since_days is not None:
        start = today - timedelta(days=int(args.since_days))
        end = today
    else:
        try:
            start = datetime.strptime(args.start, DATE_FMT).date()
        except ValueError:
            raise SystemExit("Bad --start (YYYY-MM-DD)")
        end = datetime.strptime(args.end, DATE_FMT).date() if args.end else today

    # Load current history
    hist_path = Path(args.history_csv)
    if not hist_path.exists():
        raise SystemExit(f"history_csv not found: {hist_path}")
    hist = pd.read_csv(hist_path)

    # Accumulate new rows
    all_new = []
    for d in date_iter(start, end):
        mon = to_mon_date(d)
        dprint(f"fetch {mon}")
        try:
            df_raw = fetch_results_by_date(mon, key, tries=3, timeout=45)
        except Exception as e:
            dprint(f"WARNING: fetch failed for {mon}: {type(e).__name__}: {e}")
            # Skip this day and continue; do not fail the whole run
            time.sleep(args.rate_delay)
            continue

        if df_raw.empty:
            time.sleep(args.rate_delay)
            continue

        # Filter: only completed games; defensively keep all if flag missing
        if "IsGameOver" in df_raw.columns:
            df_raw = df_raw[df_raw["IsGameOver"] == True].copy()

        new_rows = normalize_result_rows(df_raw)
        if not new_rows.empty:
            all_new.append(new_rows)
        time.sleep(args.rate_delay)

    if not all_new:
        dprint("No rows fetched.")
        sys.exit(0)

    add_df = pd.concat(all_new, ignore_index=True)
    dprint(f"Fetched rows: {len(add_df)}")

    # Standardize types
    add_df["home_or_away"] = add_df["home_or_away"].astype(int)
    add_df["points"] = add_df["points"].astype(float)
    add_df["goals"] = add_df["goals"].astype(float)
    add_df["assists"] = add_df["assists"].astype(float)
    add_df["goal_tending_goals_against"] = add_df["goal_tending_goals_against"].astype(float)
    add_df["power_play_goals"] = add_df["power_play_goals"].astype(float)
    add_df["power_play_assists"] = add_df["power_play_assists"].astype(float)
    add_df["shots_on_goal"] = add_df["shots_on_goal"].astype(float)

        
    # Merge & dedupe: prefer the *latest* row for a (name,team,date,opponent)
    # NO MORE MERGE - col issues
    # merged = pd.concat([hist, add_df], ignore_index=True, sort=False)
    # key_cols = ["name","team","date","opponent"]
    #if not set(key_cols).issubset(merged.columns):
    #    missing = [c for c in key_cols if c not in merged.columns]
    #    raise SystemExit(f"History is missing columns: {missing}")
    #merged = merged.drop_duplicates(subset=key_cols, keep="last")

    #dprint(f"History size: {len(hist)} -> {len(merged)} (+{len(merged)-len(hist)})")

    if args.dry_run:
        dprint("Dry run, not writing.")
        return

    out_path = Path(args.write) if args.write else hist_path
    if args.backup and out_path.exists():
        bak = out_path.with_suffix(out_path.suffix + ".bak")
        out_path.replace(bak)
        dprint(f"Backup written: {bak}")
        out_path = Path(args.write) if args.write else hist_path  # refresh

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ###*******************************Save result to db
    #********TODO******************************************
    ###*********Save result to db**********************
    add_df.to_csv(out_path, index=False)
    
    
    
    dprint(f"Wrote updated history: {out_path}")

if __name__ == "__main__":
    main()
