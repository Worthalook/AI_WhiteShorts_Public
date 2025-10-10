#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# train_flexi_forecast.py
import argparse, pandas as pd
import yaml as yaml
from trainer import GenericTrainer


p = argparse.ArgumentParser()
p.add_argument("--task", choices=["points","goals","assists"], required=True)
p.add_argument("--cfg", default="config.yaml")
p.add_argument("--csv_ytd", required=True)
p.add_argument("--csv_last", required=False)
p.add_argument("--out_dir", required=True)
p.add_argument("--split", required=True)
p.add_argument("--epochs", required=True)
args = p.parse_args()

def load_weighted_union( csv_ytd: str | None, csv_last: str | None, last_weight: float, season_col: str | None):

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


cfg = yaml.safe_load(open(args.cfg))
df = pd.read_csv(args.train_csv)
#join all last year with rolling ytd
df, feat_cols = load_weighted_union(args.csv_ytd, args.csv_last, args.last_weight, args.season_col)

trainer = GenericTrainer(args.task, cfg)
trainer.fit(df)


