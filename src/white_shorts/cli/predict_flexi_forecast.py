# src/white_shorts/cli/predict_flexi_forecast.py
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import typer

app = typer.Typer(help="Predictor for models trained by train_flexi_forecast.py (CSV or API projections).")

# ------------------------------- utils --------------------------------------- #

LABEL_LIKE = {"points", "goals", "assists", "label", "target", "y"}
ID_COL_CANDIDATES = ("player", "name", "team", "opponent", "game_id", "date")


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".json":
        obj = json.loads(path.read_text())
        if isinstance(obj, dict) and "data" in obj:
            return pd.DataFrame(obj["data"])
        return pd.DataFrame(obj)
    raise ValueError(f"Unsupported input format: {suf}")


def _infer_features(df: pd.DataFrame, explicit: Optional[List[str]]) -> pd.DataFrame:
    if explicit:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return df[explicit]
    # heuristic: all numeric cols not obviously labels
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric if c.lower() not in LABEL_LIKE]
    if not feats:
        raise ValueError("No numeric feature columns detected; pass --features.")
    return df[feats]


def _keep_id_cols(df_in: pd.DataFrame, extra_ids: Optional[List[str]] = None) -> pd.DataFrame:
    keep = [c for c in ID_COL_CANDIDATES if c in df_in.columns]
    if extra_ids:
        keep.extend([c for c in extra_ids if c in df_in.columns and c not in keep])
    return df_in[keep].copy() if keep else pd.DataFrame(index=df_in.index)


def _load_pickle(p: Path):
    with p.open("rb") as f:
        return pickle.load(f)


def _find_model(model_dir: Path, task_lower: str) -> Path:
    """
    Default artifact naming assumed by this predictor:
      - model_{task}.pkl  (e.g., model_goals.pkl)
    Fallbacks:
      - any *.pkl in the dir with task name inside
      - single *.pkl in the dir (if exactly one)
    """
    preferred = model_dir / f"model_{task_lower}.pkl"
    if preferred.exists():
        return preferred

    # Look for a close match
    cands = list(model_dir.glob("*.pkl"))
    cands_task = [p for p in cands if task_lower in p.name.lower()]
    if cands_task:
        return sorted(cands_task, key=lambda x: len(x.name))[0]

    if len(cands) == 1:
        return cands[0]

    raise FileNotFoundError(
        f"Could not locate model for task='{task_lower}' in {model_dir}. "
        f"Expected {preferred.name} or a single *.pkl."
    )


def _maybe_load_feature_list(model_dir: Path, task_lower: str) -> Optional[List[str]]:
    """
    If the trainer exported a feature list alongside the model, try to use it.
    Supported names:
      - features_{task}.json
      - features.json
    """
    for name in (f"features_{task_lower}.json", "features.json"):
        p = model_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return None


def _predict_df(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DF with at least 'prediction'.
    If classifier exposes predict_proba, include 'proba_1' (class-1).
    If model exposes predict_std, include 'pred_std'.
    """
    out = {}
    yhat = model.predict(X)
    out["prediction"] = np.asarray(yhat)

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            if isinstance(proba, (list, tuple)):
                proba = proba[-1]
            if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
                out["proba_1"] = proba[:, 1]
        except Exception:
            pass

    if hasattr(model, "predict_std"):
        try:
            out["pred_std"] = np.asarray(model.predict_std(X))
        except Exception:
            pass

    return pd.DataFrame(out)


# ------------------------- projections (API) -------------------------------- #

def _fetch_projections_via_import(date_str: str, api_key: str) -> Optional[pd.DataFrame]:
    """
    Try to reuse your existing implementation from:
      white_shorts.cli.batch_predict_by_player.fetch_projections
    """
    try:
        from white_shorts.cli.batch_predict_by_player import fetch_projections  # type: ignore
        df = fetch_projections(date_str, api_key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass
    return None


def _fetch_projections_direct(date_str: str, api_key: str) -> pd.DataFrame:
    """
    Minimal, dependency-light fallback to hit the same SportsData.io endpoint
    used in the legacy script.
    """
    import urllib.request
    import ssl

    base = "https://api.sportsdata.io/api/nhl/fantasy/json/PlayerGameProjectionStatsByDate"
    url = f"{base}/{date_str}?key={api_key}"

    # SportsData uses TLS; create permissive context for GHA runners if needed.
    ctx = ssl.create_default_context()

    with urllib.request.urlopen(url, context=ctx, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
        data = json.loads(raw)

    # Expect a list of dicts
    if not isinstance(data, list):
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Light normalization: keep common identifiers if present
    # (Your existing fetch_projections likely does richer shaping; we keep it minimal here.)
    # Downstream feature selection will pick numeric columns anyway.
    # Ensure 'date' column exists for ID join semantics
    if "Day" in df.columns and "DateTime" not in df.columns:
        # Older payloads sometimes use 'Day' for the date
        df["date"] = pd.to_datetime(df["Day"]).dt.date.astype(str)
    elif "DateTime" in df.columns:
        df["date"] = pd.to_datetime(df["DateTime"]).dt.date.astype(str)
    else:
        df["date"] = date_str

    # Best-effort rename to match your ID columns if available
    rename_map = {
        "Name": "name",
        "Team": "team",
        "Opponent": "opponent",
        "GameID": "game_id",
        "GameId": "game_id",
        "PlayerID": "player_id",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing:
        df = df.rename(columns=existing)

    return df


def _load_projections(date_str: str, api_key: str) -> pd.DataFrame:
    """
    First try your project's fetch_projections(); if not available, fallback.
    """
    df = _fetch_projections_via_import(date_str, api_key)
    if df is None:
        df = _fetch_projections_direct(date_str, api_key)
    return df


# ------------------------------- CLI ---------------------------------------- #

@app.command()
def run(
    csv_features: Optional[Path] = typer.Argument(None, help="Feature table (CSV/JSON). If omitted and --date/--key provided, will fetch projections from API."),
    out_csv: Path = typer.Option(Path("predictions.csv"), "--out_csv", "-o", help="Output CSV path."),
    model_dir: Path = typer.Option(..., "--model_dir", "-m", help="Directory produced by train_flexi_forecast.py (contains model .pkl)."),
    task: str = typer.Option("Points", "--task", "-t", help="Task used during training, e.g., 'Goals', 'Assists', or 'Points'."),
    features: Optional[str] = typer.Option(None, "--features", "-f", help="Comma-separated list OR path to a JSON/CSV list of feature names."),
    id_cols: Optional[str] = typer.Option(None, "--id_cols", help="Optional comma list of identifier columns to preserve in output."),
    date: Optional[str] = typer.Option(None, "--date", "-d", help="Projection date for API (e.g., 2025-Oct-13 or 2025-10-13). If provided with --key, will fetch from API."),
    key: Optional[str] = typer.Option(None, "--key", "-k", help="SportsData.io API key, required when using --date."),
):
    """
    Predict using a model trained by train_flexi_forecast.py.

    Input options:
      - CSV/JSON features file (pass as first positional arg), OR
      - API projections via --date and --key.

    Model artifacts:
      - Will look for model_{task_lower}.pkl inside --model_dir (with fallbacks).

    Output columns:
      [id cols...] + prediction (+ optional proba_1, pred_std).
    """
    task_lower = task.strip().lower()

    # -------------------- Load inputs: CSV or API -------------------- #
    if date and key:
        df_in = _load_projections(date, key)
        if df_in.empty:
            raise typer.Exit(code=2)
    else:
        if csv_features is None:
            raise typer.BadParameter("Provide a features file OR use --date and --key to fetch from API.")
        df_in = _read_table(csv_features)

    # -------------------- Feature selection -------------------------- #
    explicit_features: Optional[List[str]] = None
    if features:
        p = Path(features)
        if p.exists():
            if p.suffix.lower() == ".json":
                explicit_features = json.loads(p.read_text())
            elif p.suffix.lower() == ".csv":
                explicit_features = pd.read_csv(p, header=None).iloc[:, 0].astype(str).tolist()
            else:
                raise ValueError(f"--features path must be .json or .csv, got {p.suffix}")
        else:
            explicit_features = [x.strip() for x in features.split(",") if x.strip()]

    model_features = _maybe_load_feature_list(model_dir, task_lower)
    feat_list = explicit_features or model_features

    X = _infer_features(df_in, feat_list)

    # -------------------- Load model & predict ----------------------- #
    model_path = _find_model(model_dir, task_lower)
    model = _load_pickle(model_path)
    df_pred = _predict_df(model, X)

    # -------------------- Build & save output ------------------------ #
    id_list = [c.strip() for c in id_cols.split(",")] if id_cols else None
    ids = _keep_id_cols(df_in, id_list)
    out = pd.concat([ids.reset_index(drop=True), df_pred.reset_index(drop=True)], axis=1)

    _ensure_parent(out_csv)
    out.to_csv(out_csv, index=False)
    src = "API" if (date and key) else str(csv_features)
    typer.echo(f"[flexi-predict] task={task} | model={model_path.name} | src={src} → {out_csv}")


if __name__ == "__main__":
    app()
