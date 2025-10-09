#!/usr/bin/env python
import sqlite3, pandas as pd, numpy as np, datetime as dt, json

DB_PATH = "whiteshorts.db"
CONFIG_PATH = "config.json"

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def brier_score(y_true, p):
    return np.mean((p - y_true)**2)

def log_loss(y_true, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y_true*np.log(p) + (1-y_true)*np.log(1-p))

cfg = load_config()
with sqlite3.connect(DB_PATH) as conn:
    preds = pd.read_sql_query("SELECT * FROM predictions", conn, parse_dates=["created_at"])
    outs = pd.read_sql_query("SELECT * FROM outcomes", conn)

if preds.empty or outs.empty:
    print("Insufficient data: need predictions and outcomes.")
else:
    df = preds.merge(outs[["event_id","result"]], on="event_id", how="inner").copy()
    df["y"] = df["result"].map({"H":1.0, "A":0.0, "D":0.5}).fillna(0.0)
    df["date"] = pd.to_datetime(df["event_date"], errors="coerce").fillna(pd.to_datetime(df["created_at"]).dt.date).astype("datetime64[ns]")
    windows = cfg["metrics"]["windows"]
    bins = cfg["metrics"]["calibration_bins"]
    today = pd.Timestamp.utcnow().normalize()
    records, cal_rows = [], []
    for w in windows:
        start = today - pd.Timedelta(days=w)
        sub = df[df["date"] >= start].copy()
        n = len(sub)
        if n == 0:
            continue
        acc = np.mean((sub["home_win_prob"] >= 0.5).astype(float) == (sub["y"] >= 0.5).astype(float))
        brier = brier_score(sub["y"].values, sub["home_win_prob"].values)
        ll = log_loss(sub["y"].values, sub["home_win_prob"].values)
        records.append({
            "date": today.strftime("%Y-%m-%d"),
            "window_days": int(w),
            "n": int(n),
            "accuracy": float(acc),
            "brier": float(brier),
            "logloss": float(ll),
            "created_at": pd.Timestamp.utcnow().isoformat()
        })
        # calibration
        edges = np.linspace(0,1,bins+1)
        sub["bin"] = pd.cut(sub["home_win_prob"], edges, include_lowest=True)
        for interval, grp in sub.groupby("bin"):
            if grp.empty: 
                continue
            low, high = float(interval.left), float(interval.right)
            cal_rows.append({
                "date": today.strftime("%Y-%m-%d"),
                "window_days": int(w),
                "bin_lower": low,
                "bin_upper": high,
                "n": int(len(grp)),
                "avg_pred": float(grp["home_win_prob"].mean()),
                "avg_outcome": float(grp["y"].mean())
            })
    with sqlite3.connect(DB_PATH) as conn:
        if records:
            pd.DataFrame(records).to_sql("metrics_daily", conn, if_exists="append", index=False)
        if cal_rows:
            pd.DataFrame(cal_rows).to_sql("calibration_bins", conn, if_exists="append", index=False)
    print("Computed metrics and updated tables.")
