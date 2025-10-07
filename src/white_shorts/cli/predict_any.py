#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_any.py — v0.3
Predict next-game points using either the RF feature model or DSS-lite sequence model,
with opponent-aware features and quantile intervals.
"""

import argparse, numpy as np, pandas as pd, joblib, torch, torch.nn as nn
from datetime import datetime

# ---- (helpers identical to training script) ----
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
    for k in ["date","game_date","match_date","timestamp","time"]:
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

def prepare_data(csv_path: str, lag_k=3):
    df = pd.read_csv(csv_path)
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
    return df, feat_cols

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

def dss_predict_quantiles(ckpt, X_seq, mc=200, q_low=0.1, q_high=0.9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_cols = ckpt["feat_cols"]
    model = DSSLite(in_dim=len(feat_cols))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    X_t = torch.from_numpy(X_seq[None,...]).to(device)
    preds = []
    with torch.no_grad():
        for _ in range(mc):
            yhat = model(X_t, use_dropout=True).cpu().numpy()[0]
            preds.append(yhat[-1])
    preds = np.array(preds)
    return float(np.mean(preds)), float(np.quantile(preds, q_low)), float(np.quantile(preds, q_high))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", required=True, help="Path to rf_points_forecaster.pkl or dss_model.pt")
    ap.add_argument("--model_type", choices=["rf","dss"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--home_or_away", required=True)
    ap.add_argument("--next_date", default=None)
    ap.add_argument("--opponent", default=None, help="Opponent team name (optional)")
    ap.add_argument("--q_low", type=float, default=0.1)
    ap.add_argument("--q_high", type=float, default=0.9)
    ap.add_argument("--mc", type=int, default=200, help="MC samples for DSS")
    args = ap.parse_args()

    df, feat_cols = prepare_data(args.csv, lag_k=3)
    g = df[df["name"] == args.name].copy()
    if g.empty: raise SystemExit(f"No history for player '{args.name}'.")
    g = g.sort_values("parsed_date" if "parsed_date" in g.columns else "_row")

    last = g.iloc[-1].copy()
    hoa = to_home_flag(args.home_or_away)
    if pd.isna(hoa): raise SystemExit("Bad Home/Away value.")
    last["home_or_away"] = float(hoa)

    if args.next_date is not None and "parsed_date" in g.columns and not g["parsed_date"].isna().all():
        ld = g["parsed_date"].dropna().iloc[-1]
        import pandas as pd
        nd = pd.to_datetime(args.next_date)
        dr = (nd - ld).days
        last["days_rest_capped"] = float(min(dr, 7))
        last["rest_anomaly"] = int(dr > 7)

    if args.opponent is not None:
        last["opponent"] = args.opponent

    x_next = last[feat_cols].values.astype(np.float32)

    if args.model_type == "rf":
        model = joblib.load(args.model)
        mean_pred, lo, hi = rf_predict_quantiles(model, x_next.reshape(1,-1), args.q_low, args.q_high)
        print(f"RF forecast for {args.name} vs {args.opponent} (H/A={int(hoa)}): "
              f"mean={mean_pred[0]:.3f}, q{int(args.q_low*100)}={lo[0]:.3f}, q{int(args.q_high*100)}={hi[0]:.3f}")
    else:
        ckpt = torch.load(args.model, map_location="cpu")
        if "feat_cols" not in ckpt: raise SystemExit("Bad DSS checkpoint (missing feat_cols).")
        X_hist = g[feat_cols].values.astype(np.float32)
        X_seq = np.vstack([X_hist, x_next[None,:]])
        mean_pred, lo, hi = dss_predict_quantiles(ckpt, X_seq, args.mc, args.q_low, args.q_high)
        print(f"DSS forecast for {args.name} vs {args.opponent} (H/A={int(hoa)}): "
              f"mean={mean_pred:.3f}, q{int(args.q_low*100)}={lo:.3f}, q{int(args.q_high*100)}={hi:.3f}")

if __name__ == "__main__":
    main()
