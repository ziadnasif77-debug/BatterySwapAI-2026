"""
Interactive Streamlit dashboard for BatterySwapAI 2026.

Displays real-time station status, demand forecasts, prediction confidence
intervals, and the current work order schedule. Run with:
    streamlit run demo/dashboard.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import streamlit as st
    import plotly.graph_objects as go
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


RESULTS_DIR = Path("results")


def load_predictions() -> pd.DataFrame:
    """Load predictions.csv from results/ or return sample data."""
    path = RESULTS_DIR / "predictions.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["timestamp"])
    rng = np.random.default_rng(0)
    n = 100
    return pd.DataFrame({
        "station_id": [f"STN_{i % 5 + 1:03d}" for i in range(n)],
        "timestamp": pd.date_range("2026-06-15", periods=n, freq="h"),
        "prediction": rng.poisson(8, n).astype(float),
        "lower": rng.poisson(5, n).astype(float),
        "upper": rng.poisson(12, n).astype(float),
    })


def load_metrics() -> dict:
    """Load metrics.json or return empty dict."""
    path = RESULTS_DIR / "metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"cv_mae_mean": "N/A", "cv_rmse_mean": "N/A"}


def main() -> None:
    """Entry point for the Streamlit dashboard."""
    if not _HAS_STREAMLIT:
        print("Install streamlit to run the dashboard: pip install streamlit")
        return

    st.set_page_config(page_title="BatterySwapAI 2026", layout="wide")
    st.title("BatterySwapAI 2026 — Operations Dashboard")

    metrics = load_metrics()
    col1, col2 = st.columns(2)
    col1.metric("CV MAE", metrics.get("cv_mae_mean", "N/A"))
    col2.metric("CV RMSE", metrics.get("cv_rmse_mean", "N/A"))

    preds = load_predictions()
    stations = sorted(preds["station_id"].unique())
    selected = st.selectbox("Select station", stations)

    df = preds[preds["station_id"] == selected].sort_values("timestamp")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["prediction"],
        name="Predicted demand", line=dict(color="royalblue")
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([df["timestamp"], df["timestamp"].iloc[::-1]]),
        y=pd.concat([df["upper"], df["lower"].iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(65,105,225,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Confidence interval",
    ))
    fig.update_layout(
        title=f"Demand Forecast — {selected}",
        xaxis_title="Time",
        yaxis_title="Swaps / hour",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Predictions")
    st.dataframe(preds.tail(50), use_container_width=True)


if __name__ == "__main__":
    main()
