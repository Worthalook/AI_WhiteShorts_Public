
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
from .featurize import *

def prepare_history_union(csv: str | None, csv_ytd: str | None, csv_last: str | None, lag_k=3, season_col: Optional[str]=None):
    if csv and (csv_ytd or csv_last):
        raise SystemExit("Use either --csv OR (--csv_ytd and/or --csv_last), not both.")
    if csv:
        raw = pd.read_csv(csv)
    else:
        frames = []
        if csv_last: frames.append(pd.read_csv(csv_last))
        if csv_ytd:  frames.append(pd.read_csv(csv_ytd))
        if not frames: raise SystemExit("Provide --csv or at least one of --csv_ytd / --csv_last.")
        raw = pd.concat(frames, ignore_index=True)
        if set(["name","team","date","opponent"]).issubset(raw.columns):
            raw = raw.drop_duplicates(subset=["name","team","date","opponent"], keep="last")

    cm = normalize_cols(raw.columns.tolist())
    for req in ["name","points","home_or_away"]:
        if req not in cm:
            raise SystemExit(f"CSV missing required column '{req}'. Found: {list(raw.columns)}")
    df = raw.rename(columns={
        cm["name"]:"name",
        cm["points"]:"points",
        cm["home_or_away"]:"home_or_away",
        **({cm["team"]:"team"} if "team" in cm else {}),
        **({cm["opponent"]:"opponent"} if "opponent" in cm else {}),
        **({cm["date"]:"date"} if "date" in cm else {}),
        **({cm.get('season','season'):'season'} if 'season' in cm else {}),
    })
    df["home_or_away"] = df["home_or_away"].apply(to_home_flag).astype(float)
    if "date" in df.columns:
        df["parsed_date"] = df["date"].apply(try_parse_date)
        df = df.sort_values(["team" if "team" in df.columns else "name","parsed_date","points"], na_position="last")
    else:
        df["_row"] = np.arange(len(df))
        df = df.sort_values(["name","_row"])
    df = df.dropna(subset=["name","points","home_or_away"])

    df["season_label"] = infer_season(df, season_col=None)

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
    fill_defaults = {c: global_pts_mean for c in feat_cols if "prev_points" in c or c.endswith("_mean_pts") or c=="player_vs_opp_avg" or c=="team_vs_opp_last_meeting"}
    fill_defaults.update({c: median_rest_cap for c in feat_cols if c.endswith("_days_since_played") or c.endswith("_std_pts") or c=="opp_days_since_last_meeting"})
    for c in feat_cols:
        if c not in df.columns: df[c] = np.nan
    df[feat_cols] = df[feat_cols].fillna(fill_defaults).fillna(0.0)
    df = df.dropna(subset=[f"lag_{k}" for k in range(1, lag_k+1)])
    return df, feat_cols
