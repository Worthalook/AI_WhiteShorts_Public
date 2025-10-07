
from __future__ import annotations
from collections import Counter, defaultdict
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

def normalize_cols(columns):
    lower = {c.lower(): c for c in columns}
    m = {}
    for k in ["name","player","player_name","playername"]:
        if k in lower: m["name"] = lower[k]; break
    for k in ["points","pts","score","scored"]:
        if k in lower: m["points"] = lower[k]; break
    for k in ["homeoraway","home_away","home","is_home","homeaway"]:
        if k in lower: m["home_or_away"] = lower[k]; break
    for k in ["team","team_name","club"]:
        if k in lower: m["team"] = lower[k]; break
    for k in ["opponent","opp","opp_team","versus","vs"]:
        if k in lower: m["opponent"] = lower[k]; break
    for k in ["date","game_date","match_date","timestamp","time","datetime","day"]:
        if k in lower: m["date"] = lower[k]; break
    for k in ["season","season_id","year","yr"]:
        if k in lower: m["season"] = lower[k]; break
    return m

def to_home_flag(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (bool, np.bool_)): return int(bool(x))
    s = str(x).strip().lower()
    if s in ["home","h","1","true","t","yes","y"]: return 1
    if s in ["away","a","0","false","f","no","n"]: return 0
    try: return int(float(s) != 0.0)
    except: return np.nan

def try_parse_date(x):
    try: return pd.to_datetime(x)
    except: return pd.NaT

def add_lags(g: pd.DataFrame, lag_k=3) -> pd.DataFrame:
    g = g.copy()
    for k in range(1, lag_k+1):
        g[f"lag_{k}"] = g["points"].shift(k)
    return g

def add_days_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        df["days_rest_capped"] = 1.0
        df["rest_anomaly"] = 0
        return df
    df["parsed_date"] = df["date"].apply(try_parse_date)
    df = df.sort_values(["name","parsed_date"])
    dr = df.groupby("name")["parsed_date"].diff().dt.days
    player_median = dr.groupby(df["name"]).transform("median")
    global_median = dr.median()
    dr = dr.fillna(player_median).fillna(global_median).fillna(1.0)
    df["days_rest_capped"] = dr.clip(upper=7).astype(float)
    df["rest_anomaly"] = (dr > 7).astype(int)
    return df

def build_team_prev_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    if not {"team","date"}.issubset(df.columns):
        df["team_prev_mean_pts"] = np.nan
        df["team_prev_std_pts"] = np.nan
        return df
    df = df.copy()
    df["parsed_date"] = df["date"].apply(try_parse_date)
    df = df.sort_values(["team","parsed_date"])
    team_dates = df.drop_duplicates(["team","parsed_date"]).sort_values(["team","parsed_date"])
    team_dates["team_prev_date"] = team_dates.groupby("team")["parsed_date"].shift(1)
    prev_date_map = team_dates.set_index(["team","parsed_date"])["team_prev_date"]
    cur_mean = df.groupby(["team","parsed_date"])["points"].mean()
    cur_std  = df.groupby(["team","parsed_date"])["points"].std()
    df["team_prev_date"] = list(zip(df.get("team", np.nan), df.get("parsed_date", np.nan)))
    df["team_prev_date"] = df["team_prev_date"].map(prev_date_map)
    def lookup_prev_mean(row):
        return cur_mean.get((row.get("team", np.nan), row.get("team_prev_date", np.nan)), np.nan)
    def lookup_prev_std(row):
        return cur_std.get((row.get("team", np.nan), row.get("team_prev_date", np.nan)), np.nan)
    df["team_prev_mean_pts"] = df.apply(lookup_prev_mean, axis=1)
    df["team_prev_std_pts"]  = df.apply(lookup_prev_std, axis=1)
    return df

def build_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["player_vs_opp_avg","team_vs_opp_last_mean","opp_days_since_last_meeting"]:
        df[c] = np.nan
    if not {"opponent","date"}.issubset(df.columns):
        return df
    df = df.copy()
    df["parsed_date"] = df["date"].apply(try_parse_date)
    df = df.sort_values(["name","parsed_date"])
    df["player_vs_opp_avg"] = (
        df.groupby(["name","opponent"])["points"]
          .apply(lambda s: s.shift(1).expanding().mean())
          .reset_index(level=[0,1], drop=True)
    )
    if "team" in df.columns:
        meet = df.drop_duplicates(["team","opponent","parsed_date"]).sort_values(["team","opponent","parsed_date"])
        meet["last_meet_date"] = meet.groupby(["team","opponent"])["parsed_date"].shift(1)
        last_meet_map = meet.set_index(["team","opponent","parsed_date"])["last_meet_date"]
        df["triple"] = list(zip(df.get("team", np.nan), df.get("opponent", np.nan), df.get("parsed_date", np.nan)))
        df["last_meet_date"] = df["triple"].map(last_meet_map)
        team_on_day = df.groupby(["team","parsed_date"])["points"].mean()
        def lookup_team_last_mean(row):
            return team_on_day.get((row.get("team", np.nan), row.get("last_meet_date", np.nan)), np.nan)
        df["team_vs_opp_last_mean"] = df.apply(lookup_team_last_mean, axis=1)
        df["opp_days_since_last_meeting"] = (df["parsed_date"] - df["last_meet_date"]).dt.days
        df = df.drop(columns=["triple"])
    return df

def build_top_teammate_features(df: pd.DataFrame, top_k:int=3):
    feat_cols = []
    for i in range(1, top_k+1):
        feat_cols += [f"tm{i}_prev_points", f"tm{i}_days_since_played"]
    for c in feat_cols:
        df[c] = np.nan
    if not {"team","date","name"}.issubset(df.columns):
        return df, []
    df = df.copy()
    df["parsed_date"] = df["date"].apply(try_parse_date)
    df = df.sort_values(["team","parsed_date"])
    co_counts = defaultdict(Counter)
    for (team, dt), g in df.groupby(["team","parsed_date"]):
        names = g["name"].dropna().unique().tolist()
        for a in names:
            for b in names:
                if a != b: co_counts[a][b] += 1
    top_teammates = {p:[b for b,_ in co_counts[p].most_common(top_k)] for p in co_counts}
    last_played_date, last_points = {}, {}
    for idx in df.index:
        row = df.loc[idx]; p=row["name"]; d=row.get("parsed_date", pd.NaT)
        for i, tm in enumerate(top_teammates.get(p, [])[:top_k], start=1):
            lp = last_points.get(tm, np.nan); ld = last_played_date.get(tm, np.nan)
            days_since = (d - ld).days if (isinstance(ld, pd.Timestamp) and isinstance(d, pd.Timestamp)) else np.nan
            df.at[idx, f"tm{i}_prev_points"] = lp
            df.at[idx, f"tm{i}_days_since_played"] = days_since
        last_points[p] = row["points"]; last_played_date[p] = d
    return df, feat_cols
