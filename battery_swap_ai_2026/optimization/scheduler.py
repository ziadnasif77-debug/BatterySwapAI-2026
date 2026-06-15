"""
Technician scheduling engine for BatterySwapAI 2026.

Assigns prioritized work orders to available field technicians using a
greedy nearest-neighbor algorithm with shift-window constraints and travel
time awareness. Outputs a daily schedule with estimated arrival times and
route sequences per technician.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class Technician:
    """Represents an available field technician."""
    tech_id: str
    current_lat: float
    current_lon: float
    shift_end: datetime
    assigned_orders: list = field(default_factory=list)


@dataclass
class Station:
    """Represents a battery swap station location."""
    station_id: str
    lat: float
    lon: float


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometers between two coordinates."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def travel_time_hours(
    lat1: float, lon1: float, lat2: float, lon2: float, avg_speed_kph: float = 40.0
) -> float:
    """Estimate travel time in hours assuming constant average speed."""
    return haversine_km(lat1, lon1, lat2, lon2) / avg_speed_kph


def assign_orders(
    ranked_orders: pd.DataFrame,
    technicians: list,
    stations: dict,
    service_time_hours: float = 1.0,
    current_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Greedy assignment: assign each order to the nearest available technician.

    Args:
        ranked_orders: Output of priority.rank_orders(), sorted by urgency
        technicians: List of Technician objects with current positions
        stations: Dict mapping station_id -> Station
        service_time_hours: Time to service each station
        current_time: Schedule start time (defaults to now)

    Returns:
        DataFrame with columns [order_id, tech_id, eta, sequence]
    """
    if current_time is None:
        current_time = datetime.now()

    tech_available_at = {t.tech_id: current_time for t in technicians}
    tech_positions = {t.tech_id: (t.current_lat, t.current_lon) for t in technicians}
    assignments = []

    for _, order in ranked_orders.iterrows():
        station = stations.get(order["station_id"])
        if station is None:
            continue

        best_tech, best_eta = None, None
        for tech in technicians:
            travel_h = travel_time_hours(*tech_positions[tech.tech_id], station.lat, station.lon)
            eta = tech_available_at[tech.tech_id] + timedelta(hours=travel_h)
            done = eta + timedelta(hours=service_time_hours)
            if done <= tech.shift_end:
                if best_eta is None or eta < best_eta:
                    best_tech, best_eta = tech, eta

        if best_tech is None:
            assignments.append(
                {"order_id": order["order_id"], "tech_id": None, "eta": None, "sequence": None}
            )
            continue

        tech_available_at[best_tech.tech_id] = best_eta + timedelta(hours=service_time_hours)
        tech_positions[best_tech.tech_id] = (station.lat, station.lon)
        seq = len(best_tech.assigned_orders) + 1
        best_tech.assigned_orders.append(order["order_id"])
        assignments.append(
            {"order_id": order["order_id"], "tech_id": best_tech.tech_id,
             "eta": best_eta.isoformat(), "sequence": seq}
        )

    return pd.DataFrame(assignments)
