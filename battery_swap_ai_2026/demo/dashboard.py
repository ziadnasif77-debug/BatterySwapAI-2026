"""
demo/dashboard.py
BatterySwapAI 2026 — Live Monitor Streamlit dashboard.
"""

import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE        = Path(__file__).parent.parent
RESULTS_DIR = BASE / "results"
DATA_DIR    = BASE / "data" / "raw"
TODAY       = datetime(2026, 6, 16)

st.set_page_config(
    page_title="BatterySwapAI 2026 — Live Monitor",
    page_icon="🔋",
    layout="wide",
)


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    prio_df      = pd.read_csv(RESULTS_DIR / "prioritized_sensors.csv")
    full_preds   = pd.read_csv(RESULTS_DIR / "full_predictions.csv",
                               parse_dates=["timestamp"])
    buildings_df = pd.read_csv(DATA_DIR    / "buildings.csv")
    readings_df  = pd.read_csv(DATA_DIR    / "sensor_readings.csv",
                               parse_dates=["timestamp"])
    cal_df       = pd.read_csv(RESULTS_DIR / "calibrated_predictions.csv")

    work_orders_df = pd.DataFrame()
    wo_path = RESULTS_DIR / "work_orders.csv"
    if wo_path.exists():
        work_orders_df = pd.read_csv(wo_path)

    with open(RESULTS_DIR / "scenario_comparison.json") as f:
        scenarios = json.load(f)

    prio_df = prio_df.merge(
        buildings_df[["building_id", "building_name", "latitude", "longitude"]],
        on="building_id", how="left",
    )

    return prio_df, work_orders_df, full_preds, readings_df, buildings_df, scenarios, cal_df


prio_df, work_orders_df, full_preds, readings_df, buildings_df, scenarios, cal_df = load_data()


# ── Page header ────────────────────────────────────────────────────────────────

st.title("🔋 BatterySwapAI 2026 — Live Monitor")
st.caption(f"Showing latest predictions · Simulated date: {TODAY.strftime('%A, %d %B %Y')}")
st.divider()


# ── Component 1: Status bar ────────────────────────────────────────────────────

n_dead     = int((prio_df["risk_score"] >= 100).sum())
n_critical = int(((prio_df["risk_score"] > 70) & (prio_df["risk_score"] < 100)).sum())
n_warning  = int(((prio_df["risk_score"] > 40) & (prio_df["risk_score"] <= 70)).sum())
n_safe     = int((prio_df["risk_score"] <= 40).sum())
total      = len(prio_df)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("💀 Dead",     n_dead,     delta=None, help="Battery at 2.5 V floor")
c2.metric("🔴 Critical", n_critical, delta=None, help="Risk score > 70")
c3.metric("🟠 Warning",  n_warning,  delta=None, help="Risk score 40–70")
c4.metric("🟢 Safe",     n_safe,     delta=None, help="Risk score ≤ 40")
c5.metric("📡 Total",    total,      delta=None, help="Sensors monitored")

st.divider()


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Map", "📊 Analytics", "📋 Work Orders", "📈 Model Performance"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Map
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    map_path = Path(__file__).parent / "battery_map.html"
    if map_path.exists():
        html_content = map_path.read_text(encoding="utf-8")
        components.html(html_content, height=620, scrolling=False)
        st.caption("Click any sensor marker for details. Use layer control (top-right) to toggle routes.")
    else:
        st.warning("Map not found. Run `python demo/map_builder.py` first.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analytics
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    col_left, col_right = st.columns([3, 2])

    # ── Voltage decline chart ──────────────────────────────────────────────────
    with col_left:
        st.subheader("Voltage Decline Forecast")

        sensor_ids = sorted(prio_df["sensor_id"].unique())
        default_ix = sensor_ids.index("SEN_012") if "SEN_012" in sensor_ids else 0
        selected   = st.selectbox("Select sensor", sensor_ids, index=default_ix)

        sensor_row  = prio_df[prio_df["sensor_id"] == selected].iloc[0]
        history     = (readings_df[readings_df["sensor_id"] == selected]
                       .sort_values("timestamp"))

        rul_pred  = float(sensor_row.get("rul_predicted", 30))
        rul_lower = float(sensor_row.get("rul_lower_90",  rul_pred))
        rul_upper = float(sensor_row.get("rul_upper_90",  rul_pred))
        p_fail_7d = float(sensor_row.get("p_fail_7d", 0))
        risk_cat  = str(sensor_row.get("risk_category", "SAFE"))
        bid       = str(sensor_row.get("building_id", ""))

        last_ts      = history["timestamp"].iloc[-1] if len(history) else TODAY
        last_voltage = float(history["voltage"].iloc[-1]) if len(history) else 2.5

        eol_date   = last_ts + timedelta(days=max(0, rul_pred))
        eol_lower  = last_ts + timedelta(days=max(0, rul_lower))
        eol_upper  = last_ts + timedelta(days=max(0, rul_upper))
        dead_thresh = 2.5

        # Forecast line: linear from last_voltage toward dead_thresh
        n_forecast = max(int(rul_upper) + 5, 10)
        fc_dates   = [last_ts + timedelta(days=d) for d in range(n_forecast + 1)]
        if rul_pred > 0:
            slope = (dead_thresh - last_voltage) / rul_pred
            fc_volts = [max(dead_thresh, last_voltage + slope * d) for d in range(n_forecast + 1)]
        else:
            fc_volts = [dead_thresh] * (n_forecast + 1)

        # CI band voltages (project from lower/upper RUL)
        def _project(days_to_eol):
            if days_to_eol <= 0:
                return [dead_thresh] * (n_forecast + 1)
            slope = (dead_thresh - last_voltage) / days_to_eol
            return [max(dead_thresh, last_voltage + slope * d) for d in range(n_forecast + 1)]

        ci_lower_volts = _project(rul_lower)
        ci_upper_volts = _project(rul_upper)

        fig = go.Figure()

        # Historical
        fig.add_trace(go.Scatter(
            x=history["timestamp"], y=history["voltage"],
            mode="lines", name="Historical voltage",
            line=dict(color="#4C9BE8", width=2),
        ))

        # 90% CI band
        fig.add_trace(go.Scatter(
            x=fc_dates + fc_dates[::-1],
            y=ci_upper_volts + ci_lower_volts[::-1],
            fill="toself",
            fillcolor="rgba(255,165,0,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="90% CI band",
            showlegend=True,
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=fc_dates, y=fc_volts,
            mode="lines", name="Predicted decline",
            line=dict(color="#FF8C00", width=2, dash="dash"),
        ))

        # Dead threshold
        x_min = history["timestamp"].min() if len(history) else last_ts - timedelta(days=30)
        fig.add_hline(
            y=dead_thresh, line_color="red", line_dash="dot", line_width=1.5,
            annotation_text="Dead threshold (2.5 V)",
            annotation_position="top left",
            annotation_font_color="red",
        )

        # EOL vertical line
        fig.add_vline(
            x=eol_date.timestamp() * 1000,
            line_color="red", line_dash="dash", line_width=1.5,
            annotation_text=f"EOL {eol_date.strftime('%Y-%m-%d')}",
            annotation_position="top right",
            annotation_font_color="red",
        )

        fig.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#e0e0e0",
            xaxis=dict(title="Date", gridcolor="#222"),
            yaxis=dict(title="Voltage (V)", gridcolor="#222", range=[2.3, 4.3]),
            legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Quick stats row under chart
        s1, s2, s3 = st.columns(3)
        s1.metric("RUL Estimate",   f"{rul_pred:.1f} d")
        s2.metric("P(fail 7 days)", f"{p_fail_7d:.1f}%")
        s3.metric("Status",         risk_cat)

    # ── Risk distribution pie ──────────────────────────────────────────────────
    with col_right:
        st.subheader("Risk Distribution")

        labels = ["Dead", "Critical", "Warning", "Safe"]
        values = [n_dead, n_critical, n_warning, n_safe]
        colors = ["#808080", "#FF0000", "#FF8C00", "#00CC44"]

        pie = go.Figure(go.Pie(
            labels=labels, values=values,
            marker_colors=colors,
            hole=0.5,
            textinfo="label+value",
            textfont_size=13,
        ))
        pie.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="#0e1117",
            font_color="#e0e0e0",
            showlegend=False,
        )
        st.plotly_chart(pie, use_container_width=True)

        st.subheader("Scenario Comparison")
        for sc in scenarios:
            label    = sc["label"]
            rec      = sc.get("recommended", False)
            badge    = " ✅" if rec else ""
            bg_color = "#0a3d1f" if rec else "#1a1a2e"
            border   = "#00AA44" if rec else "#333"

            st.markdown(f"""
<div style="
    background:{bg_color};border:1px solid {border};
    border-radius:8px;padding:10px 14px;margin-bottom:8px;
">
  <b style="font-size:14px">{label}{badge}</b><br>
  <small style="color:#aaa">threshold: p_fail_7d > {sc['threshold']}%</small>
  <table style="width:100%;margin-top:6px;font-size:12px;color:#ccc">
    <tr><td>Sensors qualified</td><td style="text-align:right">{sc['n_candidates']}</td></tr>
    <tr><td>Sensors saved</td><td style="text-align:right">{sc['pct_saved']}%</td></tr>
    <tr><td>Travel hours</td><td style="text-align:right">{sc['travel_hours']}</td></tr>
    <tr><td><b style="color:#fff">Total cost</b></td>
        <td style="text-align:right"><b style="color:#fff">{sc['total_cost']:,} NOK</b></td></tr>
  </table>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Work Orders
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Today's Work Schedule")
    st.caption(f"Shift: 08:00 – 16:00 · {TODAY.strftime('%d %B %Y')} · Oslo depot (B001)")

    if work_orders_df.empty:
        st.info("No work orders scheduled for today.")
    else:
        # Summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Stops",    len(work_orders_df))
        m2.metric("Workers Active", work_orders_df["worker_id"].nunique())
        m3.metric("Batteries",      int(work_orders_df["n_batteries_to_replace"].sum()))

        st.markdown("")

        WORKER_COLORS_CSS = {1: "#0066FF", 2: "#00AA44", 3: "#9900CC"}

        for wid in sorted(work_orders_df["worker_id"].unique()):
            color = WORKER_COLORS_CSS.get(wid, "#888")
            wo    = work_orders_df[work_orders_df["worker_id"] == wid].sort_values("stop_number")

            st.markdown(
                f'<div style="border-left:4px solid {color};padding-left:12px;'
                f'margin-bottom:4px"><b style="color:{color}">Worker {wid}</b></div>',
                unsafe_allow_html=True,
            )

            display = wo[[
                "stop_number", "arrival_time", "departure_time",
                "building_name", "building_id",
                "n_batteries_to_replace", "sensor_ids",
            ]].rename(columns={
                "stop_number":            "Stop",
                "arrival_time":           "Arrival",
                "departure_time":         "Departure",
                "building_name":          "Building",
                "building_id":            "ID",
                "n_batteries_to_replace": "Batteries",
                "sensor_ids":             "Sensors",
            })
            st.dataframe(display, use_container_width=True, hide_index=True)

        st.download_button(
            label="⬇ Download Work Orders CSV",
            data=work_orders_df.to_csv(index=False).encode("utf-8"),
            file_name=f"work_orders_{TODAY.strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Performance
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Model Performance")

    valid_cal = cal_df.dropna(subset=["actual_rul", "raw_predictions", "calibrated_by_type"])

    raw_mae  = float(np.abs(valid_cal["actual_rul"] - valid_cal["raw_predictions"]).mean())
    raw_rmse = float(np.sqrt(((valid_cal["actual_rul"] - valid_cal["raw_predictions"])**2).mean()))
    cal_mae  = float(np.abs(valid_cal["actual_rul"] - valid_cal["calibrated_by_type"]).mean())
    cal_rmse = float(np.sqrt(((valid_cal["actual_rul"] - valid_cal["calibrated_by_type"])**2).mean()))
    improvement = (raw_mae - cal_mae) / raw_mae * 100

    valid_cov = full_preds.dropna(subset=["actual_rul", "rul_lower_90", "rul_upper_90"])
    coverage  = float(
        ((valid_cov["actual_rul"] >= valid_cov["rul_lower_90"]) &
         (valid_cov["actual_rul"] <= valid_cov["rul_upper_90"])).mean() * 100
    )
    avg_width = float(full_preds["interval_width"].mean())

    conf_counts = full_preds["confidence"].value_counts()

    # Metrics row 1: raw model
    st.markdown("##### LightGBM — Raw Predictions")
    r1, r2, r3 = st.columns(3)
    r1.metric("MAE",  f"{raw_mae:.1f} days",  delta=f"vs baseline 2.7 d", delta_color="inverse")
    r2.metric("RMSE", f"{raw_rmse:.1f} days")
    r3.metric("Baseline MAE", "2.7 days", help="Linear extrapolation from 14-day slope")

    st.markdown("##### After Building-Type Calibration")
    c1, c2, c3 = st.columns(3)
    c1.metric("Calibrated MAE",  f"{cal_mae:.1f} days",
              delta=f"−{improvement:.1f}% vs raw", delta_color="normal")
    c2.metric("Calibrated RMSE", f"{cal_rmse:.1f} days")
    c3.metric("MAE Improvement", f"{improvement:.1f}%")

    st.divider()

    # Metrics row 2: uncertainty
    st.markdown("##### Uncertainty Quantification (90% Confidence Intervals)")
    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Interval Coverage", f"{coverage:.1f}%",
              delta="target: 90%", delta_color="inverse" if coverage < 80 else "normal",
              help="% of actual RUL values falling within predicted 90% CI")
    u2.metric("Avg Interval Width",  f"{avg_width:.1f} days")
    u3.metric("HIGH Confidence",  f"{conf_counts.get('HIGH', 0)} sensors")
    u4.metric("LOW Confidence",   f"{conf_counts.get('LOW', 0)} sensors")

    if coverage < 80:
        st.warning(
            f"⚠ Interval coverage is {coverage:.1f}% — well below the 90% target. "
            "This is expected due to temporal distribution shift: the model was trained on "
            "Jan–Sep 2025 data but tested on Oct 2025–Mar 2026, where voltage decay "
            "patterns differ. A larger training dataset with more seasonal cycles "
            "would narrow this gap."
        )

    st.divider()

    # Scatter: predicted vs actual RUL
    st.markdown("##### Predicted vs Actual RUL")
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=valid_cal["actual_rul"], y=valid_cal["raw_predictions"],
        mode="markers", name="Raw",
        marker=dict(color="#4C9BE8", size=6, opacity=0.6),
    ))
    fig2.add_trace(go.Scatter(
        x=valid_cal["actual_rul"], y=valid_cal["calibrated_by_type"],
        mode="markers", name="Calibrated",
        marker=dict(color="#FF8C00", size=6, opacity=0.7),
    ))

    # Perfect prediction line
    max_val = float(max(valid_cal["actual_rul"].max(), valid_cal["raw_predictions"].max()))
    fig2.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", name="Perfect prediction",
        line=dict(color="#444", dash="dash", width=1),
    ))

    fig2.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#e0e0e0",
        xaxis=dict(title="Actual RUL (days)", gridcolor="#222"),
        yaxis=dict(title="Predicted RUL (days)", gridcolor="#222"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig2, use_container_width=True)
