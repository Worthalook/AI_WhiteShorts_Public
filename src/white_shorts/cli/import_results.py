#!/usr/bin/env python
import argparse, sqlite3, pandas as pd, re
from baseline_elo import DB_PATH, update_ratings_after_outcome

def canonical_event_id(date_str, home, away):
    def clean(x): return re.sub(r'[^A-Z0-9]+', '', str(x).upper())
    return f"{date_str}-{clean(home)}-V-{clean(away)}"

def main():
    p = argparse.ArgumentParser(description="Import historical/actual results (YTD) into outcomes table")
    p.add_argument("--csv_path", required=True)
    p.add_argument("--db_path", default=DB_PATH)
    p.add_argument("--date_col", default="Date")
    p.add_argument("--home_team_col", default="Home")
    p.add_argument("--away_team_col", default="Away")
    p.add_argument("--home_score_col", default="HomeGoals")
    p.add_argument("--away_score_col", default="AwayGoals")
    p.add_argument("--date_fmt_in", default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--update_elo", action="store_true")
    p.add_argument("--print_cmds", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.csv_path)
    cols = {c.lower(): c for c in df.columns}
    def get_col(name): return cols.get(name.lower(), name)
    dcol = get_col(args.date_col); hcol = get_col(args.home_team_col); acol = get_col(args.away_team_col)
    hscol = get_col(args.home_score_col); ascol = get_col(args.away_score_col)

    if args.date_fmt_in:
        dates = pd.to_datetime(df[dcol], format=args.date_fmt_in, errors="coerce")
    else:
        dates = pd.to_datetime(df[dcol], errors="coerce")
    df["_event_date"] = dates.dt.strftime("%Y-%m-%d")
    df["_home"] = df[hcol]; df["_away"] = df[acol]
    df["_home_score"] = pd.to_numeric(df[hscol], errors="coerce").astype("Int64")
    df["_away_score"] = pd.to_numeric(df[ascol], errors="coerce").astype("Int64")
    df["_event_id"] = [canonical_event_id(d, h, a) for d,h,a in zip(df["_event_date"], df["_home"], df["_away"])]

    conn = sqlite3.connect(args.db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS outcomes (event_id TEXT PRIMARY KEY, event_date TEXT, home TEXT, away TEXT, home_score INTEGER, away_score INTEGER, result TEXT, ingested_at TEXT NOT NULL)")

    inserted, skipped = 0, 0
    for row in df.itertuples(index=False):
        event_id = getattr(row, "_event_id")
        event_date = getattr(row, "_event_date")
        home, away = getattr(row, "_home"), getattr(row, "_away")
        hs, as_ = getattr(row, "_home_score"), getattr(row, "_away_score")
        if pd.isna(hs) or pd.isna(as_): continue
        hs, as_ = int(hs), int(as_)
        result = "H" if hs > as_ else ("A" if hs < as_ else "D")
        if cur.execute("SELECT 1 FROM outcomes WHERE event_id=?", (event_id,)).fetchone():
            skipped += 1; continue
        if args.print_cmds:
            print(f"python ingest_outcome.py --event_id {event_id} --home_score {hs} --away_score {as_} --event_date {event_date} --home {home} --away {away}")
        if not args.dry_run:
            cur.execute("INSERT OR REPLACE INTO outcomes(event_id, event_date, home, away, home_score, away_score, result, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                        (event_id, event_date, home, away, hs, as_, result))
            inserted += 1
            if args.update_elo:
                try: update_ratings_after_outcome(home, away, hs, as_)
                except Exception: pass

    if not args.dry_run: conn.commit()
    conn.close()
    print(f"Outcomes import complete. Inserted: {inserted}, skipped: {skipped}")

if __name__ == "__main__":
    main()
