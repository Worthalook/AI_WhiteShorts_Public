#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# train_flexi_forecast.py
import argparse, pandas as pd
import yaml
from trainer import GenericTrainer
from typing import List, Tuple
import numpy as np
from collections import Counter, defaultdict

p = argparse.ArgumentParser()
p.add_argument("--task", choices=["points","goals","assists"], required=True)
p.add_argument("--cfg", default="config.yaml")
p.add_argument("--csv_ytd", required=True)
p.add_argument("--csv_last", required=False)
p.add_argument("--out_dir", required=True)
p.add_argument("--split", required=True)
p.add_argument("--epochs", required=True)
p.add_argument("--last_weight", required=True)
p.add_argument("--season_col", default=None)
args = p.parse_args()


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

def add_lags(g: pd.DataFrame, lag_k=3) -> pd.DataFrame:
    g = g.copy()
    for k in range(1, lag_k+1):
        g[f"lag_{k}"] = g["points"].shift(k)
    return g

def try_parse_date(x):
    try: return pd.to_datetime(x)
    except: return pd.NaT
    
def to_home_flag(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (bool, np.bool_)): return int(bool(x))
    s = str(x).strip().lower()
    if s in ["home","h","1","true","t","yes","y"]: return 1
    if s in ["away","a","0","false","f","no","n"]: return 0
    try: return int(float(s) != 0.0)
    except: return np.nan

def infer_season(df: pd.DataFrame, season_col: str | None):
    if season_col and season_col in df.columns:
        return df[season_col].astype(str)
    if "date" in df.columns:
        return pd.to_datetime(df["date"], errors="coerce").dt.year.astype("Int64").astype(str)
    return pd.Series(["unknown"]*len(df))

def prepare_data(csv_path: str, lag_k=3, season_col: str | None = None) -> Tuple[pd.DataFrame, list]:
    df = pd.read_csv(csv_path)
     #-------------------------------------------
    # Always-on deterministic fallback for sorting
    df["_order"] = np.arange(len(df), dtype=np.int64)

    # Prefer a parsed date when available
    if "date" in df.columns:
        df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce")

    # Initial canonical sort (only keep keys that exist)
    primary_team_key = "team" if "team" in df.columns else "name"
    sort_keys = [c for c in (primary_team_key, "parsed_date", "points", "_order") if c in df.columns]
    if sort_keys:
        df = df.sort_values(sort_keys, na_position="last")

    # Your existing cleaning
    must_have = [c for c in ["name", "points", "home_or_away"] if c in df.columns]
    if must_have:
        df = df.dropna(subset=must_have)
    #---------------------------------------------
    cm = df.columns.tolist()# normalize_cols(df.columns.tolist())
    for req in ["name","points","home_or_away"]:
        if req not in cm:
            raise SystemExit(f"CSV missing required column '{req}'. Found: {list(df.columns)}")
   
    df["home_or_away"] = df["home_or_away"].apply(to_home_flag).astype(float)
    
    if "date" in df.columns:
        df["parsed_date"] = df["date"].apply(try_parse_date)
        df = df.sort_values(["team" if "team" in df.columns else "name","parsed_date","points"], na_position="last")
    else:
        df["_row"] = np.arange(len(df))
        df = df.sort_values(["name","_row"])
    df = df.dropna(subset=["name","points","home_or_away"])

    df["season_label"] = infer_season(df, season_col)

    df = df.groupby("name", group_keys=False).apply(lambda g: add_lags(g, lag_k))
    df = add_days_rest_features(df)
    df = build_team_prev_aggregates(df)
    df = build_opponent_features(df)
    df, tm_feats = build_top_teammate_features(df, top_k=3)

    base_feats = [f"lag_{k}" for k in range(1, lag_k+1)] + ["home_or_away","days_rest_capped","rest_anomaly"]
    team_feats = ["team_prev_mean_pts", "team_prev_std_pts"] if "team_prev_mean_pts" in df.columns else []
    opp_feats  = [c for c in ["player_vs_opp_avg","team_vs_opp_last_mean","opp_days_since_last_meeting"] if c in df.columns]
    feat_cols  = base_feats + team_feats + opp_feats + tm_feats

    global_pts_mean = float(df["points"].mean())
    median_rest_cap = float(df["days_rest_capped"].median()) if "days_rest_capped" in df else 1.0
    fill_defaults = {c: global_pts_mean for c in feat_cols if "prev_points" in c or c.endswith("_mean_pts") or c=="player_vs_opp_avg" or c=="team_vs_opp_last_mean"}
    fill_defaults.update({c: median_rest_cap for c in feat_cols if c.endswith("_days_since_played") or c.endswith("_std_pts") or c=="opp_days_since_last_meeting"})
    for c in feat_cols:
        if c not in df.columns: df[c] = np.nan
    df[feat_cols] = df[feat_cols].fillna(fill_defaults).fillna(0.0)

    df = df.dropna(subset=[f"lag_{k}" for k in range(1, lag_k+1)])

    return df, feat_cols

def load_weighted_union( csv_ytd: str | None, csv_last: str | None, last_weight: float, season_col: str | None):

    frames = []
    feat_cols_ref = None
    if csv_ytd:
        d1, feat_cols_ref = prepare_data(csv_ytd, lag_k=3, season_col=season_col)
        d1["row_weight"] = 1.0
        frames.append(d1)
    if csv_last:
        d2, feat_cols2 = prepare_data(csv_last, lag_k=3, season_col=season_col)
        d2["row_weight"] = last_weight
        frames.append(d2)
        if feat_cols_ref is None:
            feat_cols_ref = feat_cols2
    common_cols = set.intersection(*[set(f.columns) for f in frames]) if len(frames) > 1 else set(frames[0].columns)
    frames = [f[list(common_cols)] for f in frames]
    df = pd.concat(frames, ignore_index=True).sort_values(["name","parsed_date" if "parsed_date" in common_cols else "points"])
    feat_cols = [c for c in feat_cols_ref if c in df.columns]
    return df, feat_cols



cfg = yaml.safe_load(open(args.cfg))
#df = pd.read_csv(args.train_csv)
#join all last year with rolling ytd
df, feat_cols = load_weighted_union(args.csv_ytd, args.csv_last, args.last_weight, args.season_col)

trainer = GenericTrainer(args.task, cfg)
trainer.fit(df)


