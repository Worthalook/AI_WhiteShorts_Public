#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multivariate time-series forecaster for per-player points
Features:
- Lags of points (lag_1..lag_K)
- Home/Away flag
- days_rest (capped at 7) and rest_anomaly (>7 days)

Inference:
- Optional --next_date YYYY-MM-DD to compute projected days_rest from last played date.
- If omitted, uses player's median days_rest (capped), with anomaly flag accordingly.
"""
import argparse, json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

def to_home_flag(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (bool, np.bool_)): return int(bool(x))
    s = str(x).strip().lower()
    if s in ["home", "h", "1", "true", "t", "yes", "y"]: return 1
    if s in ["away", "a", "0", "false", "f", "no", "n"]: return 0
    try:
        v = float(s); return int(v != 0.0)
    except: return np.nan

def normalize_cols(columns):
    lower = {c.lower(): c for c in columns}
    mapping = {}
    for key in ["name", "player", "player_name", "playername"]:
        if key in lower: mapping["name"] = lower[key]; break
    for key in ["points", "pts", "score", "scored"]:
        if key in lower: mapping["points"] = lower[key]; break
    for key in ["homeoraway", "home_away", "home", "is_home", "homeaway"]:
        if key in lower: mapping["home_or_away"] = lower[key]; break
    for key in ["date", "game_date", "match_date", "timestamp", "time"]:
        if key in lower: mapping["date"] = lower[key]; break
    return mapping

def try_parse_date(x):
    try: return pd.to_datetime(x)
    except: return pd.NaT

def add_lags(dfi, lag_k=3):
    dfi = dfi.copy()
    for k in range(1, lag_k+1):
        dfi[f"lag_{k}"] = dfi["points"].shift(k)
    return dfi

def add_days_rest_features(df):
    df = df.copy()
    if "date" not in df.columns:
        df["days_rest"] = 1.0
        df["days_rest_capped"] = 1.0
        df["rest_anomaly"] = 0
        return df
    df["parsed_date"] = df["date"].apply(try_parse_date)
    df = df.sort_values(["name", "parsed_date"])
    df["days_rest"] = df.groupby("name")["parsed_date"].diff().dt.days
    med_by_player = df.groupby("name")["days_rest"].transform(lambda s: s.median())
    global_med = df["days_rest"].median()
    df["days_rest"] = df["days_rest"].fillna(med_by_player).fillna(global_med).fillna(1.0)
    df["days_rest_capped"] = df["days_rest"].clip(upper=7)
    df["rest_anomaly"] = (df["days_rest"] > 7).astype(int)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to training CSV")
    ap.add_argument("--name", required=True, help="Player name to forecast")
    ap.add_argument("--home_or_away", required=True, help="Home/Away flag (Home/Away/H/A/True/False/1/0)")
    ap.add_argument("--next_date", default=None, help="Projected next game date (YYYY-MM-DD)")
    ap.add_argument("--lag_k", type=int, default=3, help="Number of lags for points")
    ap.add_argument("--save_model", default=None, help="Optional path to save trained model *.pkl")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    colmap = normalize_cols(df.columns.tolist())
    for req in ["name", "points", "home_or_away"]:
        if req not in colmap:
            raise SystemExit(f"CSV missing required column for '{req}'. Found: {list(df.columns)}")
    df = df.rename(columns={colmap["name"]:"name", colmap["points"]:"points", colmap["home_or_away"]:"home_or_away", **({colmap["date"]:"date"} if "date" in colmap else {})})
    df["home_or_away"] = df["home_or_away"].apply(to_home_flag).astype(float)

    if "date" in df.columns:
        df["parsed_date"] = df["date"].apply(try_parse_date)
        df = df.sort_values(["name", "parsed_date", "points"], na_position="last")
    else:
        df["_row"] = np.arange(len(df))
        df = df.sort_values(["name", "_row"])

    df = df.dropna(subset=["name", "points", "home_or_away"])

    lag_k = args.lag_k
    df = df.groupby("name", group_keys=False).apply(lambda g: add_lags(g, lag_k))
    df = add_days_rest_features(df)

    feature_cols = [f"lag_{k}" for k in range(1, lag_k+1)] + ["home_or_away", "days_rest_capped", "rest_anomaly"]
    df_model = df.dropna(subset=feature_cols + ["points"]).copy()
    if df_model.empty:
        raise SystemExit("Not enough rows after feature engineering.")

    X = df_model[feature_cols].values
    y = df_model["points"].values
    model = RandomForestRegressor(n_estimators=800, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X, y)

    if args.save_model:
        joblib.dump(model, args.save_model)

    g = df[df["name"] == args.name].copy()
    if g.empty:
        raise SystemExit(f"No history for player '{args.name}'.")
    if "parsed_date" in g.columns:
        g = g.sort_values(["parsed_date", "points"], na_position="last")
    else:
        g = g.sort_values("_row" if "_row" in g.columns else g.index)

    recent_points = g["points"].tolist()
    lags = [recent_points[-k] if len(recent_points) - k >= 0 else np.nan for k in range(1, lag_k+1)]
    if any(pd.isna(lags)):
        fill = float(np.nanmean(recent_points)) if len(recent_points) else float(np.nanmean(df["points"]))
        lags = [fill if pd.isna(v) else v for v in lags]

    # next-game rest
    if args.next_date is not None and ("parsed_date" in g.columns) and (not g["parsed_date"].isna().all()):
        last_date = g["parsed_date"].dropna().iloc[-1]
        try:
            nd = pd.to_datetime(args.next_date)
            dr = (nd - last_date).days
        except Exception:
            dr = float(g["days_rest"].median()) if "days_rest" in g.columns else 1.0
    else:
        dr = float(g["days_rest"].median()) if "days_rest" in g.columns else 1.0

    dr_capped = float(np.clip(dr, None, 7))
    rest_anom = int(dr > 7)

    hoa = to_home_flag(args.home_or_away)
    if pd.isna(hoa):
        raise SystemExit("Could not interpret Home/Away. Use Home/Away/H/A/True/False/1/0.")

    feats = np.array(lags + [hoa, dr_capped, rest_anom], dtype=float).reshape(1, -1)
    pred = model.predict(feats)[0]
    print(f"Forecasted points for {args.name} (HomeOrAway={int(hoa)}, days_rest={dr}, capped={dr_capped}, anomaly={rest_anom}): {pred:.3f}")

if __name__ == "__main__":
    main()
