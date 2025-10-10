#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_predict_by_player.py — v0.3.3
... (see header in previous attempt) ...
"""
import argparse, sys, json, os
from datetime import datetime
from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

try:
    import requests
except Exception:
    requests = None

# ---------- Featurization helpers (same as v0.3) ----------
def normalize_cols(columns):
    lower = {c.lower(): c for c in columns}
    m = {}
    for k in ["name","player","player_name","playername"]:
        if k in lower: m["name"] = lower[k]; break
    for k in ["points","pts","score","scored"]:
        if k in lower: m["points"] = lower[k]; break
    for k in ["home_or_away", "homeoraway","home_away","home","is_home","homeaway"]:
        if k in lower: m["home_or_away"] = lower[k]; break
    for k in ["team","team_name","club"]:
        if k in lower: m["team"] = lower[k]; break
    for k in ["opponent","opp","opp_team","versus","vs"]:
        if k in lower: m["opponent"] = lower[k]; break
    for k in ["date","game_date","match_date","timestamp","time","datetime","day"]:
        if k in lower: m["date"] = lower[k]; break
    return m

def to_home_flag(x):
    import pandas as pd, numpy as np
    if pd.isna(x): return np.nan
    if isinstance(x, (bool, np.bool_)): return int(bool(x))
    s = str(x).strip().lower()
    if s in ["home","h","1","true","t","yes","y"]: return 1
    if s in ["away","a","0","false","f","no","n"]: return 0
    try: return int(float(s) != 0.0)
    except: return np.nan

def try_parse_date(x):
    import pandas as pd
    try: return pd.to_datetime(x)
    except: return pd.NaT

def add_lags(g: pd.DataFrame, lag_k=3) -> pd.DataFrame:
    g = g.copy()
    for k in range(1, lag_k+1):
        g[f"lag_{k}"] = g["points"].shift(k)
    return g

def add_days_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
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

from collections import Counter, defaultdict
def build_team_prev_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd, numpy as np
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
    import pandas as pd
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
    import pandas as pd, numpy as np
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

def _feature_engineer_from_raw(raw_df: pd.DataFrame, lag_k=3):
    df = raw_df.copy()
    cm = normalize_cols(df.columns.tolist())
    for req in ["name","points","home_or_away"]:
        if req not in cm:
            raise SystemExit(f"CSV missing required column '{req}'. Found: {list(df.columns)}")
    df = df.rename(columns={
        cm["name"]:"name",
        cm["points"]:"points",
        cm["home_or_away"]:"home_or_away",
        **({cm["team"]:"team"} if "team" in cm else {}),
        **({cm["opponent"]:"opponent"} if "opponent" in cm else {}),
        **({cm["date"]:"date"} if "date" in cm else {})
    })
    df["home_or_away"] = df["home_or_away"].apply(to_home_flag).astype(float)
    if "date" in df.columns:
        df["parsed_date"] = df["date"].apply(try_parse_date)
        df = df.sort_values(["team" if "team" in df.columns else "name","parsed_date","points"], na_position="last")
    else:
        df["_row"] = np.arange(len(df))
        df = df.sort_values(["name","_row"])
    df = df.dropna(subset=["name","points","home_or_away"])

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

def prepare_history_union(csv: str | None, csv_ytd: str | None, csv_last: str | None, lag_k=3):
   # raw = pd.read_csv(csv_ytd)
  #  return _feature_engineer_from_raw(raw, lag_k=lag_k)
  #  if csv and (csv_ytd or csv_last):
  #      raise SystemExit("Use either --csv OR (--csv_ytd and/or --csv_last), not both.")
  #  if csv:
  #      raw = pd.read_csv(csv)
  #      return _feature_engineer_from_raw(raw, lag_k=lag_k)
  #  if not csv_ytd and not csv_last:
  #      raise SystemExit("Provide --csv or at least one of --csv_ytd / --csv_last.")
    frames = []
    if csv_last:
        frames.append(pd.read_csv(csv_last))
    if csv_ytd:
        frames.append(pd.read_csv(csv_ytd))
    raw_union = pd.concat(frames, ignore_index=True)
    if set(["name","team","date","opponent"]).issubset(raw_union.columns):
        raw_union = raw_union.drop_duplicates(subset=["name","team","date","opponent"], keep="last")
    return _feature_engineer_from_raw(raw_union, lag_k=lag_k)

class DSSLite(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.GRU(input_size=in_dim, hidden_size=hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, use_dropout=False):
        if use_dropout: self.train(True)
        else: self.train(False)
        out,_ = self.rnn(x)
        out = self.dropout(out) if use_dropout else out
        return self.head(out).squeeze(-1)

def rf_predict_quantiles(model, X, q_low=0.1, q_high=0.9):
    all_tree_preds = np.stack([est.predict(X) for est in model.estimators_], axis=0)
    mean_pred = all_tree_preds.mean(axis=0)
    lo = np.quantile(all_tree_preds, q_low, axis=0)
    hi = np.quantile(all_tree_preds, q_high, axis=0)
    return mean_pred, lo, hi

def dss_predict_quantiles(model, X_seq, mc=200, q_low=0.1, q_high=0.9, device="cpu"):
    X_t = torch.from_numpy(X_seq[None,...]).to(device)
    preds = []
    with torch.no_grad():
        for _ in range(mc):
            yhat = model(X_t, use_dropout=True).cpu().numpy()[0]
            preds.append(yhat[-1])
    preds = np.array(preds)
    return float(np.mean(preds)), float(np.quantile(preds, q_low)), float(np.quantile(preds, q_high))

# ---------- API ingest ----------
DATE_INPUT_FORMATS = ["%Y-%m-%d","%Y-%b-%d","%Y-%B-%d","%d-%m-%Y","%d/%m/%Y"]

def normalize_date_input(date_str: str) -> tuple[str,str]:
    for fmt in DATE_INPUT_FORMATS:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%b-%d")
        except ValueError:
            continue
    raise SystemExit("Could not parse --date. Try 2025-05-07 or 2025-May-07.")

def fetch_projections(date_mon: str, api_key: str) -> pd.DataFrame:
    if requests is None:
        raise SystemExit("The 'requests' package is unavailable; install it locally to call the API.")
    base = "https://api.sportsdata.io/api/nhl/fantasy/json/PlayerGameProjectionStatsByDate"
    url = f"{base}/{date_mon}?key={api_key}"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise SystemExit(f"SportsData.io API returned {r.status_code}: {r.text[:300]}")
    data = r.json()
    if not isinstance(data, list):
        raise SystemExit("Unexpected payload shape: expected a JSON array.")
    return pd.DataFrame(data)

def load_offline_json(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Offline JSON not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("Offline JSON must be a list of objects.")
    return pd.DataFrame(data)

def projections_to_rows(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "PlayerID": "player_id",
        "Name": "name",
        "Team": "team",
        "Opponent": "opponent",
        "HomeOrAway": "home_away",
        "Position": "position",
        "DateTime": "datetime",
        "Day": "day"
    }
    keep = [c for c in cols if c in df.columns]
    out = df[keep].rename(columns=cols).copy()
    out["HomeOrAway_bool"] = (out["home_away"].astype(str).str.upper() == "HOME").astype(int)
    if "day" in out and out["day"].notna().any():
        out["target_date"] = pd.to_datetime(out["day"], errors="coerce")
    else:
        out["target_date"] = pd.to_datetime(out["datetime"], errors="coerce")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="Single, already-unioned history CSV") #not used after rollback
    ap.add_argument("--csv_ytd", default=None, help="This season rolling file (will be unioned if provided)")
    ap.add_argument("--csv_last", default=None, help="Last season static file (will be unioned if provided)")
    ap.add_argument("--head_to_head", required=True, help="per_player_head_to_head.csv (winner per player)")
    ap.add_argument("--rf_model", required=True)
    ap.add_argument("--dss_model", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--diagnostics_csv", required=False, default=None, help="CSV listing API players missing from history")
    ap.add_argument("--date", default=None, help="Prediction date, e.g., 2025-05-07 or 2025-May-07")
    ap.add_argument("--key", default=None, help="SportsData.io API key")
    ap.add_argument("--offline_json", default=None, help="If provided, load projections from file instead of API")
    ap.add_argument("--team_filter", default=None, help="Only predict players with Team==this value (e.g., FLA)")
    ap.add_argument("--q_low", type=float, default=0.1)
    ap.add_argument("--q_high", type=float, default=0.9)
    ap.add_argument("--mc", type=int, default=200)
    args = ap.parse_args()

    hist_df, feat_cols_hist = prepare_history_union(args.csv, args.csv_ytd, args.csv_last, lag_k=3)

    hh = pd.read_csv(args.head_to_head)
    if "winner" not in hh.columns or "name" not in hh.columns:
        raise SystemExit("head_to_head must include 'name' and 'winner'.")
    winner_map = {row["name"]: row["winner"] for _, row in hh.iterrows()}

    rf = joblib.load(args.rf_model)
    ckpt = torch.load(args.dss_model, map_location="cpu")
    if "feat_cols" not in ckpt: raise SystemExit("Bad DSS checkpoint: missing feat_cols.")
    feat_cols = ckpt["feat_cols"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dss = DSSLite(in_dim=len(feat_cols)).to(device)
    dss.load_state_dict(ckpt["state_dict"])

    if args.offline_json:
        proj_raw = load_offline_json(args.offline_json)
        iso_date = None
    else:
        if not args.date or not args.key:
            raise SystemExit("--date and --key required for live API (or use --offline_json).")
        iso_date, mon_date = normalize_date_input(args.date)
        proj_raw = fetch_projections(mon_date, args.key)

    proj = projections_to_rows(proj_raw)
    if args.team_filter is not None and "Team" in proj_raw.columns:
        proj = proj[proj["team"].astype(str).str.upper() == args.team_filter.upper()].copy()

    names_in_hist = set(hist_df["name"].unique().tolist())
    missing_rows = proj[~proj["name"].isin(names_in_hist)].copy()
    if args.diagnostics_csv:
        Path(args.diagnostics_csv).parent.mkdir(parents=True, exist_ok=True)
        missing_rows.to_csv(args.diagnostics_csv, index=False)

    records = []
    for _, pr in proj.iterrows():
        name = pr.get("name", "")
        if name not in names_in_hist:
            continue
        hoa_flag = int(pr.get("HomeOrAway_bool", 0))
        opponent = pr.get("opponent", None)
        next_date = pr.get("target_date", pd.NaT)

        g = hist_df[hist_df["name"] == name].copy()
        g = g.sort_values("parsed_date" if "parsed_date" in g.columns else "_row")
        last = g.iloc[-1].copy()

        last["home_or_away"] = float(hoa_flag)
        if pd.notna(next_date) and "parsed_date" in g.columns and not g["parsed_date"].isna().all():
            ld = g["parsed_date"].dropna().iloc[-1]
            dr = (next_date - ld).days
            last["days_rest_capped"] = float(min(dr, 7))
            last["rest_anomaly"] = int(dr > 7)
        if opponent is not None:
            last["opponent"] = opponent

        for c in feat_cols:
            if c not in g.columns:
                g[c] = np.nan
        x_next = last[feat_cols].values.astype(np.float32)

        chosen = winner_map.get(name, "DSS")
        if chosen == "RF":
            mean_pred, lo, hi = rf_predict_quantiles(rf, x_next.reshape(1,-1), args.q_low, args.q_high)
            mean_pred, lo, hi = float(mean_pred[0]), float(lo[0]), float(hi[0])
            model_used = "RF"
        else:
            X_hist = g[feat_cols].values.astype(np.float32)
            X_seq = np.vstack([X_hist, x_next[None,:]])
            mean_pred, lo, hi = dss_predict_quantiles(dss, X_seq, args.mc, args.q_low, args.q_high, device=device)
            model_used = "DSS"

        prob_score = float(1.0 - np.exp(-float(mean_pred)))
        if prob_score < 0.0: prob_score = 0.0
        if prob_score > 1.0: prob_score = 1.0

        records.append({
            "name": name,
            "team": pr.get("team",""),
            "opponent": opponent,
            "home_or_away": "HOME" if hoa_flag==1 else "AWAY",
            "date": (iso_date if isinstance(iso_date, str) else (str(next_date.date()) if pd.notna(next_date) else "")),
            "chosen_model": model_used,
            "pred_mean": round(mean_pred, 6),
            "prob_score": round(prob_score, 6),
            f"q{int(args.q_low*100)}": round(lo, 6),
            f"q{int(args.q_high*100)}": round(hi, 6),
        })

    out_df = pd.DataFrame(records).sort_values(["team","name"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} with {len(out_df)} rows.")
    if args.diagnostics_csv:
        print(f"Diagnostics (missing in history): {args.diagnostics_csv} ({len(missing_rows)} rows)")

if __name__ == "__main__":
    main()