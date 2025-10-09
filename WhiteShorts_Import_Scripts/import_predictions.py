#!/usr/bin/env python
import argparse, sqlite3, pandas as pd, json, re
from baseline_elo import DB_PATH

def canonical_event_id(date_str, home, away):
    def clean(x): return re.sub(r'[^A-Z0-9]+', '', str(x).upper())
    return f"{date_str}-{clean(home)}-V-{clean(away)}"

def main():
    p = argparse.ArgumentParser(description="Import predictions CSV into predictions table without duplicates")
    p.add_argument("--csv_path", required=True)
    p.add_argument("--db_path", default=DB_PATH)
    p.add_argument("--date_col", default="game_date")
    p.add_argument("--home_team_col", default="home_team")
    p.add_argument("--away_team_col", default="away_team")
    p.add_argument("--prob_col", default="home_win_prob")
    p.add_argument("--prob_is_home", action="store_true")
    p.add_argument("--date_fmt_in", default=None)
    p.add_argument("--model_name", default="whiteshorts")
    p.add_argument("--model_version", default="unknown")
    p.add_argument("--features_cols", default="")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.csv_path)
    cols = {c.lower(): c for c in df.columns}
    def get_col(name): return cols.get(name.lower(), name)
    dcol = get_col(args.date_col); hcol = get_col(args.home_team_col); acol = get_col(args.away_team_col); pcol = get_col(args.prob_col)

    if args.date_fmt_in:
        dates = pd.to_datetime(df[dcol], format=args.date_fmt_in, errors="coerce")
    else:
        dates = pd.to_datetime(df[dcol], errors="coerce")
    df["_event_date"] = dates.dt.strftime("%Y-%m-%d")
    df["_home"] = df[hcol]; df["_away"] = df[acol]
    probs = pd.to_numeric(df[pcol], errors="coerce")
    if not args.prob_is_home: probs = 1.0 - probs
    df["_p_home"] = probs.clip(0,1)
    df["_event_id"] = [canonical_event_id(d, h, a) for d,h,a in zip(df["_event_date"], df["_home"], df["_away")]

    feat_cols = [c.strip() for c in args.features_cols.split(",") if c.strip()]

    conn = sqlite3.connect(args.db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS predictions (pred_id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL, event_date TEXT NOT NULL, home TEXT NOT NULL, away TEXT NOT NULL, model_name TEXT NOT NULL, model_version TEXT NOT NULL, home_win_prob REAL NOT NULL, created_at TEXT NOT NULL, input_hash TEXT NOT NULL, features_json TEXT)")

    inserted, skipped = 0, 0
    for row in df.itertuples(index=False):
        event_id = getattr(row, "_event_id")
        event_date = getattr(row, "_event_date")
        home, away = getattr(row, "_home"), getattr(row, "_away")
        p_home = getattr(row, "_p_home")
        if pd.isna(p_home): continue
        if cur.execute("SELECT 1 FROM predictions WHERE event_id=? AND model_name=? AND model_version=?", (event_id, args.model_name, args.model_version)).fetchone():
            skipped += 1; continue
        features_json = {}
        for c in feat_cols:
            if c in df.columns: features_json[c] = getattr(row, c)
        if not args.dry_run:
            cur.execute("INSERT INTO predictions(event_id, event_date, home, away, model_name, model_version, home_win_prob, created_at, input_hash, features_json) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, ?)",
                        (event_id, event_date, home, away, args.model_name, args.model_version, float(p_home), f"{event_id}|{args.model_name}|{args.model_version}", json.dumps(features_json)))
            inserted += 1

    if not args.dry_run: conn.commit()
    conn.close()
    print(f"Predictions import complete. Inserted: {inserted}, skipped: {skipped}")

if __name__ == "__main__":
    main()
