"""
Station map builder for BatterySwapAI 2026.

Generates an interactive Folium map showing all battery swap stations
color-coded by their current urgency level (green/amber/red), with
popups displaying predicted demand, inventory status, and the next
scheduled technician visit.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

try:
    import folium
    _HAS_FOLIUM = True
except ImportError:
    _HAS_FOLIUM = False


URGENCY_COLORS = {
    "MONITOR": "green",
    "DISPATCH": "orange",
    "EMERGENCY": "red",
}


def build_map(
    stations: pd.DataFrame,
    work_orders: Optional[pd.DataFrame] = None,
    center: tuple = (30.0444, 31.2357),
    zoom: int = 12,
):
    """
    Build a Folium map with station markers.

    Args:
        stations: DataFrame with [station_id, lat, lon, inventory_pct, predicted_demand]
        work_orders: Optional ranked work order DataFrame from priority.rank_orders()
        center: (lat, lon) map center — defaults to Cairo, Egypt
        zoom: Initial zoom level

    Returns:
        folium.Map object (call .save('map.html') to export)
    """
    if not _HAS_FOLIUM:
        raise ImportError("Install folium: pip install folium")

    m = folium.Map(location=list(center), zoom_start=zoom, tiles="CartoDB positron")

    urgency_lookup = {}
    if work_orders is not None:
        urgency_lookup = dict(zip(work_orders["station_id"], work_orders["recommended_action"]))

    for _, row in stations.iterrows():
        urgency = urgency_lookup.get(row["station_id"], "MONITOR")
        color = URGENCY_COLORS.get(urgency, "gray")
        popup_html = (
            f"<b>{row['station_id']}</b><br>"
            f"Inventory: {row.get('inventory_pct', '?')}%<br>"
            f"Predicted demand (4h): {row.get('predicted_demand', '?')}<br>"
            f"Status: <span style='color:{color}'><b>{urgency}</b></span>"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=10,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=row["station_id"],
        ).add_to(m)

    return m


def save_map(m, output_path: str = "results/station_map.html") -> None:
    """Save the Folium map to an HTML file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    m.save(output_path)
    print(f"Map saved to {output_path}")
