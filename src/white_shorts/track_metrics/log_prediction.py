#!/usr/bin/env python
import argparse, sqlite3, json, hashlib, datetime as dt
from baseline_elo import predict, DB_PATH

def input_hash(payload: dict) -> str:
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

p = argparse.ArgumentParser(description="Log a prediction into the monitoring DB")
p.add_argument("--event_id", required=True)
p.add_argument("--event_date", required=True)
p.add_argument("--home", required=True)
p.add_argument("--away", required=True)
p.add_argument("--model_name", default="baseline_elo")
p.add_argument("--model_version", default="0.1")
p.add_argument("--features_json", default="{}")
args = p.parse_args()

prob_home = predict(args.home, args.away)
payload = {
    "event_id": args.event_id,
    "event_date": args.event_date,
    "home": args.home,
    "away": args.away,
    "model_name": args.model_name,
    "model_version": args.model_version,
    "home_win_prob": prob_home,
    "created_at": dt.datetime.utcnow().isoformat(),
    "features_json": json.loads(args.features_json)
}
ihash = input_hash(payload)

conn = sqlite3.connect(DB_PATH)
conn.execute(
    "INSERT INTO predictions(event_id, event_date, home, away, model_name, model_version, home_win_prob, created_at, input_hash, features_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    (payload["event_id"], payload["event_date"], payload["home"], payload["away"], payload["model_name"], payload["model_version"], payload["home_win_prob"], payload["created_at"], ihash, json.dumps(payload["features_json"]))
)
conn.commit()
conn.close()
print(f"Logged prediction for {args.event_id}: P(home)={prob_home:.3f}")
