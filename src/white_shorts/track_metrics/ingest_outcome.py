#!/usr/bin/env python
import argparse, sqlite3, datetime as dt
from baseline_elo import DB_PATH, update_ratings_after_outcome

p = argparse.ArgumentParser(description="Ingest an outcome and update Elo ratings")
p.add_argument("--event_id", required=True)
p.add_argument("--home_score", type=int, required=True)
p.add_argument("--away_score", type=int, required=True)
p.add_argument("--event_date")
p.add_argument("--home")
p.add_argument("--away")
args = p.parse_args()

conn = sqlite3.connect(DB_PATH)
row = conn.execute("SELECT home, away, event_date FROM predictions WHERE event_id=? ORDER BY created_at DESC LIMIT 1", (args.event_id,)).fetchone()
home = args.home or (row[0] if row else None)
away = args.away or (row[1] if row else None)
event_date = args.event_date or (row[2] if row else None)

result = "H" if args.home_score > args.away_score else ("A" if args.home_score < args.away_score else "D")
conn.execute(
    "INSERT OR REPLACE INTO outcomes(event_id, event_date, home, away, home_score, away_score, result, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    (args.event_id, event_date, home, away, args.home_score, args.away_score, result, dt.datetime.utcnow().isoformat())
)
conn.commit()
conn.close()

update_ratings_after_outcome(home, away, args.home_score, args.away_score)
print(f"Ingested outcome for {args.event_id}: {home} {args.home_score} - {args.away_score} {away} (result={result})")
