#!/usr/bin/env python
import sqlite3, json, math, datetime as dt
CONFIG_PATH = "config.json"
DB_PATH = "whiteshorts.db"

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_conn():
    return sqlite3.connect(DB_PATH)

def ensure_team(conn, team, init_rating):
    cur = conn.cursor()
    row = cur.execute("SELECT rating FROM ratings WHERE team=?", (team,)).fetchone()
    if row is None:
        cur.execute("INSERT INTO ratings(team, rating, updated_at) VALUES (?, ?, ?)", (team, init_rating, dt.datetime.utcnow().isoformat()))
        conn.commit()
        return init_rating
    return row[0]

def expected_score(ra, rb):
    return 1.0 / (1.0 + math.pow(10.0, (rb - ra)/400.0))

def predict(home, away):
    cfg = load_config()
    h_adv = cfg["elo"]["home_advantage"]
    init = cfg["elo"]["init_rating"]
    with get_conn() as conn:
        r_home = ensure_team(conn, home, init)
        r_away = ensure_team(conn, away, init)
        p_home = expected_score(r_home + h_adv, r_away)
    return p_home

def update_ratings_after_outcome(home, away, home_score, away_score):
    cfg = load_config()
    k = cfg["elo"]["k_factor"]
    h_adv = cfg["elo"]["home_advantage"]
    init = cfg["elo"]["init_rating"]
    with get_conn() as conn:
        cur = conn.cursor()
        r_home = ensure_team(conn, home, init)
        r_away = ensure_team(conn, away, init)
        if home_score > away_score:
            s_home, s_away = 1.0, 0.0
        elif home_score < away_score:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5
        e_home = 1.0 / (1.0 + math.pow(10.0, (r_away - (r_home + h_adv))/400.0))
        e_away = 1.0 - e_home
        new_home = r_home + k * (s_home - e_home)
        new_away = r_away + k * (s_away - e_away)
        cur.execute("UPDATE ratings SET rating=?, updated_at=? WHERE team=?", (new_home, dt.datetime.utcnow().isoformat(), home))
        cur.execute("UPDATE ratings SET rating=?, updated_at=? WHERE team=?", (new_away, dt.datetime.utcnow().isoformat(), away))
        conn.commit()
