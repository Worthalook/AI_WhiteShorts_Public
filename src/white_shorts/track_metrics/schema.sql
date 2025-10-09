
CREATE TABLE IF NOT EXISTS predictions (
    pred_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    event_date TEXT NOT NULL,
    home TEXT NOT NULL,
    away TEXT NOT NULL,
    model_name TEXT NOT NULL DEFAULT 'baseline_elo',
    model_version TEXT NOT NULL DEFAULT '0.1',
    home_win_prob REAL NOT NULL,
    created_at TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    features_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_event ON predictions(event_id);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(event_date);

CREATE TABLE IF NOT EXISTS outcomes (
    event_id TEXT PRIMARY KEY,
    event_date TEXT,
    home TEXT,
    away TEXT,
    home_score INTEGER,
    away_score INTEGER,
    result TEXT,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics_daily (
    date TEXT PRIMARY KEY,
    window_days INTEGER NOT NULL,
    n INTEGER NOT NULL,
    accuracy REAL,
    brier REAL,
    logloss REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS calibration_bins (
    date TEXT NOT NULL,
    window_days INTEGER NOT NULL,
    bin_lower REAL NOT NULL,
    bin_upper REAL NOT NULL,
    n INTEGER NOT NULL,
    avg_pred REAL NOT NULL,
    avg_outcome REAL NOT NULL,
    PRIMARY KEY (date, window_days, bin_lower, bin_upper)
);

CREATE TABLE IF NOT EXISTS ratings (
    team TEXT PRIMARY KEY,
    rating REAL NOT NULL,
    updated_at TEXT NOT NULL
);
