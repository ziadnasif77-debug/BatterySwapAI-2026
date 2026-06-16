"""
demo/map_builder.py
Interactive Norway map showing all sensors with colored markers and worker routes.
"""

import sys
import json
import folium
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.priority import assign_risk_color

RESULTS_DIR    = Path(__file__).parent.parent / "results"
DATA_DIR       = Path(__file__).parent.parent / "data" / "raw"

PRIO_PATH      = RESULTS_DIR / "prioritized_sensors.csv"
WORK_ORDERS    = RESULTS_DIR / "work_orders.csv"
BUILDINGS_PATH = DATA_DIR    / "buildings.csv"
READINGS_PATH  = DATA_DIR    / "sensor_readings.csv"

WORKER_COLORS = {
    1: "#0066FF",   # blue
    2: "#00AA44",   # green
    3: "#9900CC",   # purple
}

DEPOT_ID = "B001"
TODAY    = datetime(2026, 6, 16)


def create_base_map() -> folium.Map:
    return folium.Map(
        location=[62.0, 10.0],
        zoom_start=6,
        tiles="CartoDB dark_matter",
    )


def _top_risk_factors(row: pd.Series) -> list[str]:
    """
    Derive top 3 risk factors from prediction columns.
    Used when SHAP values are unavailable.
    """
    factors = []

    voltage_pct   = float(row.get("voltage_pct",   100))
    rul_predicted = float(row.get("rul_predicted",  999))
    p_fail_7d     = float(row.get("p_fail_7d",      0.0))
    p_fail_3d     = float(row.get("p_fail_3d",      0.0))
    building_type = str(row.get("building_type",  "office"))
    interval_width= float(row.get("interval_width", 0.0))

    # Ordered by severity
    if voltage_pct <= 0:
        factors.append("Battery depleted (voltage at 2.5 V floor)")
    elif voltage_pct < 10:
        factors.append(f"Critical battery level ({voltage_pct:.1f}% remaining)")
    elif voltage_pct < 30:
        factors.append(f"Low battery level ({voltage_pct:.1f}% remaining)")

    if rul_predicted <= 3:
        factors.append(f"Imminent failure — RUL {rul_predicted:.1f} days")
    elif rul_predicted <= 7:
        factors.append(f"Failure within 7 days (RUL {rul_predicted:.1f} d)")
    elif rul_predicted <= 30:
        factors.append(f"Failure within 30 days (RUL {rul_predicted:.1f} d)")

    if p_fail_7d > 20:
        factors.append(f"Very high 7-day failure probability ({p_fail_7d:.1f}%)")
    elif p_fail_7d > 5:
        factors.append(f"Elevated 7-day failure probability ({p_fail_7d:.1f}%)")

    if p_fail_3d > 5:
        factors.append(f"High 3-day failure risk ({p_fail_3d:.1f}%)")

    if building_type == "hospital":
        factors.append("Critical site — hospital building (×1.5 weight)")

    if interval_width > 30:
        factors.append(f"High prediction uncertainty (±{interval_width:.0f} day CI)")

    # Fallback
    if not factors:
        factors.append("Routine monitoring — no critical indicators")

    return factors[:3]


def add_sensor_markers(
    map_obj:               folium.Map,
    prioritized_sensors_df: pd.DataFrame,
    buildings_df:          pd.DataFrame,
) -> None:
    """
    Add a CircleMarker per sensor, colored by risk, with a detailed popup.
    """
    coord_map   = dict(zip(buildings_df["building_id"],
                           zip(buildings_df["latitude"], buildings_df["longitude"])))
    name_map    = dict(zip(buildings_df["building_id"], buildings_df["building_name"]))

    sensor_layer = folium.FeatureGroup(name="Sensors", show=True)

    for _, row in prioritized_sensors_df.iterrows():
        sid          = str(row["sensor_id"])
        bid          = str(row["building_id"])
        coords       = coord_map.get(bid)
        if coords is None:
            continue

        risk_score    = float(row.get("risk_score",    0))
        risk_color    = str(row.get("risk_color",     "#00CC44"))
        voltage_pct   = float(row.get("voltage_pct",  0))
        rul_predicted = float(row.get("rul_predicted", 999))
        rul_lower     = float(row.get("rul_lower_90",  rul_predicted))
        rul_upper     = float(row.get("rul_upper_90",  rul_predicted))
        p_fail_7d     = float(row.get("p_fail_7d",    0))
        risk_category = str(row.get("risk_category",  "SAFE"))
        confidence    = str(row.get("confidence",     "LOW"))
        building_name = name_map.get(bid, bid)

        eol_date = (TODAY + timedelta(days=max(0, rul_predicted))).strftime("%Y-%m-%d")

        radius = 10 if risk_score > 70 else 7
        factors = _top_risk_factors(row)

        factors_html = "".join(
            f"<li style='margin:2px 0'>{i+1}. {f}</li>"
            for i, f in enumerate(factors)
        )

        badge_color = {
            "DEAD": "#808080", "CRITICAL": "#FF0000",
            "WARNING": "#FF8C00", "SAFE": "#00CC44",
        }.get(risk_category, "#00CC44")

        popup_html = f"""
<div style="
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
    color: #e0e0e0;
    background: #1a1a2e;
    border: 1px solid {risk_color};
    border-radius: 8px;
    padding: 14px 16px;
    min-width: 260px;
    max-width: 300px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.6);
">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
    <b style="font-size:14px;color:#ffffff">{sid}</b>
    <span style="
        background:{badge_color};
        color:#fff;
        font-size:10px;
        font-weight:bold;
        padding:2px 7px;
        border-radius:10px;
    ">{risk_category}</span>
  </div>
  <div style="color:#aaa;margin-bottom:10px;font-size:12px">{building_name} ({bid})</div>

  <hr style="border:0;border-top:1px solid #333;margin:8px 0">

  <table style="width:100%;border-collapse:collapse">
    <tr>
      <td style="color:#aaa;padding:2px 0">Battery Level</td>
      <td style="text-align:right;color:{risk_color};font-weight:bold">{voltage_pct:.1f}%</td>
    </tr>
    <tr>
      <td style="color:#aaa;padding:2px 0">Confidence</td>
      <td style="text-align:right;color:#ccc">{confidence}</td>
    </tr>
  </table>

  <hr style="border:0;border-top:1px solid #333;margin:8px 0">

  <table style="width:100%;border-collapse:collapse">
    <tr>
      <td style="color:#aaa;padding:2px 0">RUL Prediction</td>
      <td style="text-align:right;color:#ffffff;font-weight:bold">{rul_predicted:.1f} days</td>
    </tr>
    <tr>
      <td style="color:#aaa;padding:2px 0">Predicted EOL</td>
      <td style="text-align:right;color:#ccc">{eol_date}</td>
    </tr>
    <tr>
      <td style="color:#aaa;padding:2px 0">90% CI Range</td>
      <td style="text-align:right;color:#ccc">{rul_lower:.0f} – {rul_upper:.0f} d</td>
    </tr>
  </table>

  <hr style="border:0;border-top:1px solid #333;margin:8px 0">

  <table style="width:100%;border-collapse:collapse">
    <tr>
      <td style="color:#aaa;padding:2px 0">Risk Score</td>
      <td style="text-align:right;color:{risk_color};font-weight:bold">{risk_score:.0f} / 100</td>
    </tr>
    <tr>
      <td style="color:#aaa;padding:2px 0">P(fail 7 days)</td>
      <td style="text-align:right;color:#ccc">{p_fail_7d:.1f}%</td>
    </tr>
  </table>

  <hr style="border:0;border-top:1px solid #333;margin:8px 0">

  <div style="color:#aaa;font-size:11px;margin-bottom:4px">Top Risk Factors:</div>
  <ul style="margin:0;padding-left:16px;color:#cccccc;font-size:11px;line-height:1.6">
    {factors_html}
  </ul>
</div>
"""

        folium.CircleMarker(
            location=coords,
            radius=radius,
            color=risk_color,
            fill=True,
            fill_color=risk_color,
            fill_opacity=0.85,
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{sid} — Risk {risk_score:.0f}/100 ({risk_category})",
        ).add_to(sensor_layer)

    sensor_layer.add_to(map_obj)


def add_worker_routes(
    map_obj:       folium.Map,
    work_orders_df: pd.DataFrame,
    buildings_df:  pd.DataFrame,
) -> None:
    """
    Draw colored PolyLines for each worker's route, starting and ending at depot.
    Adds numbered stop markers and direction arrows.
    """
    if work_orders_df.empty:
        return

    coord_map = dict(zip(buildings_df["building_id"],
                         zip(buildings_df["latitude"], buildings_df["longitude"])))
    depot_coords = coord_map.get(DEPOT_ID)

    routes_layer = folium.FeatureGroup(name="Worker Routes", show=True)

    for worker_id in sorted(work_orders_df["worker_id"].unique()):
        color = WORKER_COLORS.get(worker_id, "#FFFFFF")
        wo    = (work_orders_df[work_orders_df["worker_id"] == worker_id]
                 .sort_values("stop_number"))

        # Build ordered coordinate list: depot → stops → depot
        route_coords = []
        if depot_coords:
            route_coords.append(depot_coords)

        stop_coords = []
        for _, row in wo.iterrows():
            bid    = str(row["building_id"])
            coords = coord_map.get(bid)
            if coords:
                route_coords.append(coords)
                stop_coords.append((coords, row))

        if depot_coords:
            route_coords.append(depot_coords)

        if len(route_coords) < 2:
            continue

        # Route polyline
        folium.PolyLine(
            locations=route_coords,
            color=color,
            weight=3,
            opacity=0.8,
            dash_array=None,
            tooltip=f"Worker {worker_id} route",
        ).add_to(routes_layer)

        # Midpoint arrows (direction indicators)
        for i in range(len(route_coords) - 1):
            a, b   = route_coords[i], route_coords[i + 1]
            mid_lat = (a[0] + b[0]) / 2
            mid_lon = (a[1] + b[1]) / 2
            folium.RegularPolygonMarker(
                location=[mid_lat, mid_lon],
                number_of_sides=3,
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                rotation=0,
            ).add_to(routes_layer)

        # Numbered stop markers
        for stop_num, (coords, row) in enumerate(stop_coords, start=1):
            popup_html = f"""
<div style="
    font-family:'Segoe UI',Arial,sans-serif;
    font-size:12px;
    background:#1a1a2e;
    color:#e0e0e0;
    border:1px solid {color};
    border-radius:6px;
    padding:10px 14px;
    min-width:200px;
">
  <b style="color:{color}">Worker {worker_id} — Stop {stop_num}</b><br>
  <span style="color:#aaa">{row['building_name']} ({row['building_id']})</span><br><br>
  <table style="width:100%;border-collapse:collapse">
    <tr><td style="color:#aaa">Arrival</td><td style="text-align:right">{row['arrival_time']}</td></tr>
    <tr><td style="color:#aaa">Departure</td><td style="text-align:right">{row['departure_time']}</td></tr>
    <tr><td style="color:#aaa">Batteries</td><td style="text-align:right">{row['n_batteries_to_replace']}</td></tr>
    <tr><td style="color:#aaa">Sensors</td><td style="text-align:right;font-size:11px">{row['sensor_ids']}</td></tr>
  </table>
</div>
"""
            folium.Marker(
                location=coords,
                icon=folium.DivIcon(
                    html=f"""
<div style="
    background:{color};
    color:#fff;
    font-weight:bold;
    font-size:11px;
    width:20px;height:20px;
    border-radius:50%;
    display:flex;
    align-items:center;
    justify-content:center;
    border:2px solid #fff;
    box-shadow:0 2px 6px rgba(0,0,0,0.5);
    margin-left:-10px;margin-top:-10px;
">{stop_num}</div>""",
                    icon_size=(20, 20),
                    icon_anchor=(10, 10),
                ),
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"W{worker_id} Stop {stop_num}: {row['building_name']} @ {row['arrival_time']}",
            ).add_to(routes_layer)

    # Depot marker
    if depot_coords:
        folium.Marker(
            location=depot_coords,
            icon=folium.DivIcon(
                html="""
<div style="
    background:#FFD700;
    color:#000;
    font-weight:bold;
    font-size:10px;
    padding:3px 6px;
    border-radius:4px;
    border:2px solid #fff;
    white-space:nowrap;
    box-shadow:0 2px 6px rgba(0,0,0,0.5);
    margin-left:-24px;margin-top:-12px;
">DEPOT</div>""",
                icon_size=(48, 24),
                icon_anchor=(24, 12),
            ),
            tooltip="Depot (B001 — Oslo Sentrum)",
        ).add_to(routes_layer)

    routes_layer.add_to(map_obj)


def add_legend(map_obj: folium.Map) -> None:
    """Add HTML legend in bottom-right corner explaining marker colors."""
    legend_html = """
<div id="map-legend" style="
    position: fixed;
    bottom: 30px;
    right: 15px;
    z-index: 9999;
    background: rgba(20, 20, 40, 0.92);
    border: 1px solid #444;
    border-radius: 10px;
    padding: 14px 18px;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 12px;
    color: #e0e0e0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.7);
    min-width: 200px;
">
  <div style="font-weight:bold;font-size:13px;color:#fff;margin-bottom:10px;
              border-bottom:1px solid #444;padding-bottom:6px">
    Battery Risk Legend
  </div>

  <div style="margin-bottom:8px;font-weight:bold;color:#bbb;font-size:11px">SENSOR STATUS</div>
  <div style="display:flex;align-items:center;margin-bottom:5px">
    <span style="width:14px;height:14px;border-radius:50%;background:#808080;
                 display:inline-block;margin-right:9px;flex-shrink:0"></span>
    <span>Dead — Battery at floor (100/100)</span>
  </div>
  <div style="display:flex;align-items:center;margin-bottom:5px">
    <span style="width:14px;height:14px;border-radius:50%;background:#FF0000;
                 display:inline-block;margin-right:9px;flex-shrink:0"></span>
    <span>Critical — Risk &gt; 70</span>
  </div>
  <div style="display:flex;align-items:center;margin-bottom:5px">
    <span style="width:14px;height:14px;border-radius:50%;background:#FF8C00;
                 display:inline-block;margin-right:9px;flex-shrink:0"></span>
    <span>Warning — Risk 40–70</span>
  </div>
  <div style="display:flex;align-items:center;margin-bottom:12px">
    <span style="width:14px;height:14px;border-radius:50%;background:#00CC44;
                 display:inline-block;margin-right:9px;flex-shrink:0"></span>
    <span>Safe — Risk &lt; 40</span>
  </div>

  <div style="font-weight:bold;color:#bbb;font-size:11px;margin-bottom:6px">WORKER ROUTES</div>
  <div style="display:flex;align-items:center;margin-bottom:5px">
    <span style="width:22px;height:3px;background:#0066FF;
                 display:inline-block;margin-right:9px;flex-shrink:0"></span>
    <span>Worker 1</span>
  </div>
  <div style="display:flex;align-items:center;margin-bottom:5px">
    <span style="width:22px;height:3px;background:#00AA44;
                 display:inline-block;margin-right:9px;flex-shrink:0"></span>
    <span>Worker 2</span>
  </div>
  <div style="display:flex;align-items:center;margin-bottom:10px">
    <span style="width:22px;height:3px;background:#9900CC;
                 display:inline-block;margin-right:9px;flex-shrink:0"></span>
    <span>Worker 3</span>
  </div>

  <div style="display:flex;align-items:center">
    <span style="width:14px;height:14px;border-radius:3px;background:#FFD700;
                 display:inline-block;margin-right:9px;flex-shrink:0"></span>
    <span>Depot (B001 Oslo)</span>
  </div>

  <div style="margin-top:10px;border-top:1px solid #333;padding-top:8px;
              color:#888;font-size:10px;text-align:center">
    Click any marker for details
  </div>
</div>
"""
    map_obj.get_root().html.add_child(folium.Element(legend_html))


def build_and_save_map(output_path: str = "demo/battery_map.html") -> None:
    # ── Load data ──────────────────────────────────────────────────────────
    prio_df     = pd.read_csv(PRIO_PATH)
    buildings_df= pd.read_csv(BUILDINGS_PATH)

    work_orders_df = pd.DataFrame()
    if WORK_ORDERS.exists():
        work_orders_df = pd.read_csv(WORK_ORDERS)

    # Enrich prioritized sensors with latest raw voltage/temperature
    readings_df = pd.read_csv(READINGS_PATH, parse_dates=["timestamp"])
    latest_raw  = (readings_df.sort_values("timestamp")
                   .groupby("sensor_id")[["voltage", "temperature"]]
                   .last()
                   .reset_index())
    prio_df = prio_df.merge(latest_raw, on="sensor_id", how="left")

    # ── Build map ──────────────────────────────────────────────────────────
    m = create_base_map()
    add_sensor_markers(m, prio_df, buildings_df)
    add_worker_routes(m, work_orders_df, buildings_df)
    add_legend(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # ── Save ───────────────────────────────────────────────────────────────
    out = Path(__file__).parent.parent / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print(f"Map saved to {out}")
    print("Open in browser to view")


if __name__ == "__main__":
    build_and_save_map()
