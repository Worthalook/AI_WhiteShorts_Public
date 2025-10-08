#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_compare_season.py — v0.3.2 (YTD + Last-Season weighting)
A/B compare RandomForest (feature model) vs. DSS-lite (GRU). Adds support for
two training sources with sample-weighting so newer games matter more.

New flags (optional):
  --csv_ytd /path/YTD.csv
  --csv_last /path/LastSeason.csv
  --last_weight 0.5         # weight applied to 'last' rows (YTD rows use 1.0)
You can still use the legacy --csv flag alone (no weighting).
"""
import argparse, json, os, math
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict

# --------------------- Feature engineering (opponent-aware) ---------------------
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
    for k in ["date","game_date","match_date","timestamp","time"]:
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

def infer_season(df: pd.DataFrame, season_col: str | None):
    if season_col and season_col in df.columns:
        return df[season_col].astype(str)
    if "date" in df.columns:
        return pd.to_datetime(df["date"], errors="coerce").dt.year.astype("Int64").astype(str)
    return pd.Series(["unknown"]*len(df))

def prepare_data(csv_path: str, lag_k=3, season_col: str | None = None) -> Tuple[pd.DataFrame, list]:
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
        **({cm["date"]:"date"} if "date" in cm else {}),
        **({cm["season"]:"season"} if "season" in cm else {})
    })
    # REPLACE WITH BELOW TO GUARD AGAINST LIST vs BOOL H&A    
    # OLD-----df["home_or_away"] = df["home_or_away"].apply(to_home_flag).astype(float)
    # --- robust normalization of home_or_away to {0.0, 1.0} --------------------
    if "home_or_away" not in df.columns:
        raise ValueError("Missing required column: home_or_away")

  
    # 0) Flatten any nested structures to a scalar (first element) per row
    def _scalarize(v):
        try:
            import numpy as _np
            import pandas as _pd
       
            # list / tuple
            if isinstance(v, (list, tuple)) and len(v):
                return v[0]
        except Exception:
            pass
        # numpy array
        try:
            import numpy as _np
            if isinstance(v, _np.ndarray) and v.size:
                return v.flat[0]
        except Exception:
            pass
        # pandas Series
        try:
            import pandas as _pd
            if isinstance(v, _pd.Series) and not v.empty:
                return v.iloc[0]
        except Exception:
            pass
        return v

    s = df["home_or_away"]#.map(_scalarize)

    # 1) try direct numeric coercion (handles 0/1, booleans True/False -> 1/0)
    try:
        num = pd.to_numeric(s, errors="coerce")
    except Exception:
        print(f"home_or_away NOT NUMERIC. Actual: {s} ---")
        num = 0
        pass
            
    try:
        # 2) map common string variants (HOME/AWAY, H/A, TRUE/FALSE, '1'/'0', etc.)
        mapped = (
            s.astype(str).str.strip().str.upper().map({
                "HOME": 1, "H": 1, "TRUE": 1, "T": 1, "1": 1, "YES": 1, "Y": 1,
                "AWAY": 0, "A": 0, "FALSE": 0, "F": 0, "0": 0, "NO": 0, "N": 0,
            })
        )

        # 3) prefer mapped string results, else numeric, else NaN
        df["home_or_away"] = mapped.fillna(num).astype(float)
        # ---------------------------------------------------------------------------
    except Exception:
        print(f"home_or_away NOT MAPPED. Actual: {s} ---")
        #df["home_or_away"] = 
        pass

    # END REPLACE WITH BELOW TO GUARD AGAINST LIST vs BOOL H&A


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

# --------------------- Combine with weights ---------------------
def load_weighted_union(csv: str | None, csv_ytd: str | None, csv_last: str | None, last_weight: float, season_col: str | None):
    if csv and (csv_ytd or csv_last):
        raise SystemExit("Use either --csv OR (--csv_ytd and/or --csv_last), not both.")
    if csv:
        df, feat_cols = prepare_data(csv, lag_k=3, season_col=season_col)
        df["row_weight"] = 1.0
        return df, feat_cols
    if not csv_ytd and not csv_last:
        raise SystemExit("Provide --csv or at least one of --csv_ytd / --csv_last.")
    frames = []
    feat_cols_ref = None
    if csv_ytd:
        d1, feat_cols_ref = prepare_data(csv_ytd, lag_k=3, season_col=season_col)
        d1["row_weight"] = 1.0
        frames.append(d1)
    if csv_last:
        d2, feat_cols2 = prepare_data(csv_last, lag_k=3, season_col=season_col)
        d2["row_weight"] = float(last_weight)
        frames.append(d2)
        if feat_cols_ref is None:
            feat_cols_ref = feat_cols2
    common_cols = set.intersection(*[set(f.columns) for f in frames]) if len(frames) > 1 else set(frames[0].columns)
    frames = [f[list(common_cols)] for f in frames]
    df = pd.concat(frames, ignore_index=True).sort_values(["name","parsed_date" if "parsed_date" in common_cols else "points"])
    feat_cols = [c for c in feat_cols_ref if c in df.columns]
    return df, feat_cols

# --------------------- Splits ---------------------
def time_split_mask(df: pd.DataFrame, train_frac=0.8) -> np.ndarray:
    mask = np.zeros(len(df), dtype=bool)
    for name, g in df.groupby("name"):
        idx = g.index.to_list()
        n = len(idx)
        if n < 5: mask[idx] = True
        else:
            cut = int(math.floor(train_frac * n))
            mask[idx[:cut]] = True
    return mask

def season_walk_splits(df: pd.DataFrame):
    seasons = [s for s in df["season_label"].dropna().unique().tolist() if s != "nan"]
    try: seasons_sorted = sorted(seasons, key=lambda x: int(x))
    except: seasons_sorted = sorted(seasons)
    splits = []
    for i in range(1, len(seasons_sorted)):
        train_seasons = set(seasons_sorted[:i])
        test_season = seasons_sorted[i]
        tr_mask = df["season_label"].isin(train_seasons).values
        te_mask = (df["season_label"] == test_season).values
        splits.append((tr_mask, te_mask, test_season))
    return splits

# --------------------- Models ---------------------
def train_rf(df: pd.DataFrame, feat_cols: List[str], train_mask: np.ndarray):
    Xtr = df.loc[train_mask, feat_cols].values
    ytr = df.loc[train_mask, "points"].values
    wtr = df.loc[train_mask, "row_weight"].values if "row_weight" in df.columns else None
    model = RandomForestRegressor(n_estimators=800, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr, sample_weight=wtr)
    return model

def eval_rf(model, df: pd.DataFrame, feat_cols: List[str], test_mask: np.ndarray):
    Xte = df.loc[test_mask, feat_cols].values
    yte = df.loc[test_mask, "points"].values
    preds = model.predict(Xte) if len(Xte) else np.array([])
    overall_mae = float(mean_absolute_error(yte, preds)) if len(Xte) else float("nan")
    per_player = []
    te_idx = df.index[test_mask]
    for name, g in df.loc[te_idx].groupby("name"):
        if not len(g): continue
        pp = model.predict(g[feat_cols].values)
        per_player.append({"name": name, "n_test": int(len(g)), "mae": float(mean_absolute_error(g["points"].values, pp))})
    return overall_mae, pd.DataFrame(per_player)

class PlayerSeqDataset(Dataset):
    def __init__(self, dfi: pd.DataFrame, feat_cols: List[str], mask: np.ndarray, min_len: int = 5):
        self.samples = []
        dfx = dfi.loc[mask].copy()
        for _, g in dfx.groupby("name"):
            g = g.sort_values("parsed_date" if "parsed_date" in g.columns else "_row")
            X = g[feat_cols].values.astype(np.float32)
            y = g["points"].values.astype(np.float32)
            if "row_weight" in g.columns:
                w = g["row_weight"].values.astype(np.float32)
            else:
                w = np.ones_like(y, dtype=np.float32)
            if len(X) < min_len: continue
            for end in range(min_len, len(X)+1):
                xs = X[:end][-64:]
                ys = y[:end][-64:]
                ws = w[:end][-64:]
                self.samples.append((xs, ys, ws))
        self.feat_dim = len(feat_cols)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        xs, ys, ws = self.samples[i]
        return torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ws)

def collate_pad(batch):
    L = max(len(x[0]) for x in batch)
    F = batch[0][0].shape[1]
    X = torch.zeros(len(batch), L, F)
    Y = torch.zeros(len(batch), L)
    M = torch.zeros(len(batch), L)
    W = torch.ones(len(batch), L)
    for i,(xs,ys,ws) in enumerate(batch):
        l = len(xs)
        X[i,-l:,:] = xs
        Y[i,-l:]   = ys
        M[i,-l:]   = 1.0
        W[i,-l:]   = ws
    return X,Y,M,W

class DSSLite(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.GRU(input_size=in_dim, hidden_size=hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        out,_ = self.rnn(x)
        return self.head(out).squeeze(-1)

def train_dss(df: pd.DataFrame, feat_cols: List[str], train_mask: np.ndarray, epochs=8, batch_size=32, lr=3e-3, max_batches=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = PlayerSeqDataset(df, feat_cols, train_mask, min_len=5)
    if len(ds) == 0: raise SystemExit("Not enough sequence data to train DSS-lite.")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pad)
    model = DSSLite(in_dim=len(feat_cols)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.L1Loss(reduction='none')
    for ep in range(1, epochs+1):
        model.train(); iters=0
        for b,(X,Y,M,W) in enumerate(dl):
            X=X.to(device); Y=Y.to(device); M=M.to(device); W=W.to(device)
            pred=model(X)
            loss=((loss_fn(pred,Y)*M*W).sum())/((M*W).sum()+1e-8)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            iters+=1
            if max_batches and iters>=max_batches: break
    return model

def eval_dss(model, df: pd.DataFrame, feat_cols: List[str], test_mask: np.ndarray):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    te_idx = df.index[test_mask]
    preds, trues, names = [], [], []
    model.eval()
    with torch.no_grad():
        for name, g in df.loc[te_idx].groupby("name"):
            full_g = df[df["name"] == name].sort_values("parsed_date" if "parsed_date" in df.columns else "_row")
            X_full = full_g[feat_cols].values.astype(np.float32)
            y_full = full_g["points"].values.astype(np.float32)
            test_positions = full_g.index.isin(g.index).nonzero()[0]
            for r in test_positions:
                X_seq = torch.from_numpy(X_full[:r+1][None,...]).to(device)
                y_hat_seq = model(X_seq).cpu().numpy()[0]
                preds.append(float(y_hat_seq[-1]))
                trues.append(float(y_full[r]))
                names.append(name)
    overall_mae = float(mean_absolute_error(trues, preds)) if len(trues) else float("nan")
    per_player = []
    if len(trues):
        tmp = pd.DataFrame({"name": names, "true": trues, "pred": preds})
        for name, gg in tmp.groupby("name"):
            per_player.append({"name": name, "n_test": int(len(gg)), "mae": float(mean_absolute_error(gg["true"], gg["pred"]))})
    return overall_mae, pd.DataFrame(per_player)

# --------------------- Main ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--csv_ytd", default=None, help="This season rolling file")
    ap.add_argument("--csv_last", default=None, help="Last season static file")
    ap.add_argument("--last_weight", type=float, default=0.5, help="Weight applied to last-season rows (YTD uses 1.0)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--split", choices=["time_per_player","season_walk"], default="time_per_player")
    ap.add_argument("--season_col", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df, feat_cols = load_weighted_union(args.csv, args.csv_ytd, args.csv_last, args.last_weight, args.season_col)

    results = []
    rf_pp_all = []; dss_pp_all = []
    season_rows = []

    if args.split == "time_per_player":
        train_mask = time_split_mask(df, 0.8)
        rf_model = train_rf(df, feat_cols, train_mask)
        rf_mae, rf_pp = eval_rf(rf_model, df, feat_cols, ~train_mask)
        dss_model = train_dss(df, feat_cols, train_mask, args.epochs, args.batch_size, args.lr, args.max_batches)
        dss_mae, dss_pp = eval_dss(dss_model, df, feat_cols, ~train_mask)

        joblib.dump(rf_model, os.path.join(args.out_dir, "rf_points_forecaster.pkl"))
        with open(os.path.join(args.out_dir, "rf_meta.json"), "w") as f:
            json.dump({"feature_cols": feat_cols}, f, indent=2)
        torch.save({"state_dict": dss_model.state_dict(), "feat_cols": feat_cols}, os.path.join(args.out_dir, "dss_model.pt"))
        with open(os.path.join(args.out_dir, "dss_meta.json"), "w") as f:
            json.dump({"epochs": args.epochs, "feat_cols": feat_cols}, f, indent=2)

        results.append({"model":"RandomForest (feature)","overall_test_MAE": rf_mae, "artifact":"rf_points_forecaster.pkl"})
        results.append({"model":"DSS-lite (GRU)","overall_test_MAE": dss_mae, "artifact":"dss_model.pt"})
        rf_pp_all.append(rf_pp); dss_pp_all.append(dss_pp)

    else:
        splits = season_walk_splits(df)
        if not splits:
            raise SystemExit("Need at least 2 distinct seasons for season_walk split.")
        for tr_mask, te_mask, season in splits:
            rf_model = train_rf(df, feat_cols, tr_mask)
            rf_mae, rf_pp = eval_rf(rf_model, df, feat_cols, te_mask)
            rf_pp["season"] = season

            dss_model = train_dss(df, feat_cols, tr_mask, args.epochs, args.batch_size, args.lr, args.max_batches)
            dss_mae, dss_pp = eval_dss(dss_model, df, feat_cols, te_mask)
            dss_pp["season"] = season

            season_rows.append({"season": season, "rf_mae": rf_mae, "dss_mae": dss_mae})

            last = (season == splits[-1][2])
            if last:
                joblib.dump(rf_model, os.path.join(args.out_dir, "rf_points_forecaster.pkl"))
                with open(os.path.join(args.out_dir, "rf_meta.json"), "w") as f:
                    json.dump({"feature_cols": feat_cols}, f, indent=2)
                torch.save({"state_dict": dss_model.state_dict(), "feat_cols": feat_cols}, os.path.join(args.out_dir, "dss_model.pt"))
                with open(os.path.join(args.out_dir, "dss_meta.json"), "w") as f:
                    json.dump({"epochs": args.epochs, "feat_cols": feat_cols}, f, indent=2)

            rf_pp_all.append(rf_pp); dss_pp_all.append(dss_pp)

        overall_rf = float(np.nanmean([r["rf_mae"] for r in season_rows]))
        overall_dss = float(np.nanmean([r["dss_mae"] for r in season_rows]))
        results.append({"model":"RandomForest (feature)","overall_test_MAE": overall_rf, "artifact":"rf_points_forecaster.pkl"})
        results.append({"model":"DSS-lite (GRU)","overall_test_MAE": overall_dss, "artifact":"dss_model.pt"})
        pd.DataFrame(season_rows).to_csv(os.path.join(args.out_dir, "overall_by_season.csv"), index=False)

    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(args.out_dir, "comparison_overall.csv"), index=False)
    rf_pp = pd.concat(rf_pp_all, ignore_index=True) if rf_pp_all else pd.DataFrame(columns=["name","n_test","mae"])
    dss_pp = pd.concat(dss_pp_all, ignore_index=True) if dss_pp_all else pd.DataFrame(columns=["name","n_test","mae"])
    rf_pp.to_csv(os.path.join(args.out_dir, "rf_per_player.csv"), index=False)
    dss_pp.to_csv(os.path.join(args.out_dir, "dss_per_player.csv"), index=False)

    hh = rf_pp.merge(dss_pp, on="name", how="outer", suffixes=("_rf","_dss")).fillna({"mae_rf": np.nan, "mae_dss": np.nan})
    hh["delta_rf_minus_dss"] = hh["mae_rf"] - hh["mae_dss"]
    hh["winner"] = np.where(hh["mae_rf"] < hh["mae_dss"], "RF", np.where(hh["mae_rf"] > hh["mae_dss"], "DSS", "tie"))
    hh = hh[["name","n_test_rf","mae_rf","n_test_dss","mae_dss","delta_rf_minus_dss","winner"]]
    hh.to_csv(os.path.join(args.out_dir, "per_player_head_to_head.csv"), index=False)

    print("\n=== Overall comparison (MAE) ===")
    print(summary.to_string(index=False))
    if os.path.exists(os.path.join(args.out_dir, "overall_by_season.csv")):
        print("\n=== Season-wise overall MAE ===")
        print(pd.read_csv(os.path.join(args.out_dir, "overall_by_season.csv")).to_string(index=False))
    print(f"\nArtifacts in: {args.out_dir}")

if __name__ == "__main__":
    main()
