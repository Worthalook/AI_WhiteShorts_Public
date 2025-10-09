#!/usr/bin/env python
import sqlite3, pandas as pd, numpy as np
import streamlit as st
DB_PATH = "whiteshorts.db"
st.set_page_config(page_title="WhiteShorts Performance", layout="wide")

@st.cache_data
def load_table(name):
    with sqlite3.connect(DB_PATH) as conn:
        try: return pd.read_sql_query(f"SELECT * FROM {name}", conn)
        except: return pd.DataFrame()

st.title("WhiteShorts Performance Tracker")
md = load_table("metrics_daily").sort_values(["date","window_days"])
if md.empty:
    st.info("No metrics yet. Log predictions, ingest outcomes, then run compute_metrics.py")
else:
    col1, col2, col3 = st.columns(3)
    for w, col in zip([7,30,90],[col1,col2,col3]):
        if (md["window_days"]==w).any():
            latest = md[md["window_days"]==w].tail(1).iloc[0]
            col.metric(f"{w}d Accuracy", f"{latest['accuracy']*100:.1f}%")
    st.subheader("Rolling Metrics")
    for w in sorted(md["window_days"].unique()):
        sub = md[md["window_days"]==w].copy()
        sub["date"] = pd.to_datetime(sub["date"])
        st.line_chart(sub.set_index("date")[["accuracy","brier","logloss"]])
    st.subheader("Calibration (Last window)")
    cb = load_table("calibration_bins")
    if not cb.empty:
        sel_w = st.selectbox("Window", sorted(cb["window_days"].unique()))
        last_date = cb[cb["window_days"]==sel_w]["date"].max()
        calib = cb[(cb["window_days"]==sel_w) & (cb["date"]==last_date)].sort_values("bin_lower")
        st.bar_chart(calib.set_index("bin_lower")[["avg_pred","avg_outcome"]])
st.subheader("Recent Predictions")
preds = load_table("predictions").sort_values("created_at", ascending=False).head(50)
st.dataframe(preds)
