"""
Work order priority scoring for BatterySwapAI 2026.

Scores each pending maintenance work order on a 0-100 urgency scale using
a weighted combination of: battery health risk, station inventory level,
SLA time remaining, station criticality (traffic volume), and predicted
demand in the next 4 hours.
"""

import pandas as pd
from dataclasses import dataclass


@dataclass
class WorkOrder:
    """Represents a single pending maintenance work order."""
    order_id: str
    station_id: str
    battery_health_pct: float      # 0-100; lower is more urgent
    inventory_level_pct: float     # 0-100; lower is more urgent
    sla_remaining_hours: float     # hours until SLA breach
    station_traffic_rank: float    # 0-1; higher means more critical station
    predicted_demand_4h: float     # expected swaps in next 4 hours


WEIGHTS = {
    "battery_health": 0.25,
    "inventory": 0.30,
    "sla": 0.25,
    "traffic": 0.10,
    "demand": 0.10,
}


def score_order(order: WorkOrder) -> float:
    """
    Compute a 0-100 priority score for a single work order.

    Higher score means more urgent dispatch needed.
    """
    battery_score = 100 - order.battery_health_pct
    inventory_score = 100 - order.inventory_level_pct
    sla_score = max(0.0, 100 - order.sla_remaining_hours * 4)
    traffic_score = order.station_traffic_rank * 100
    demand_score = min(order.predicted_demand_4h * 5, 100)

    return (
        WEIGHTS["battery_health"] * battery_score
        + WEIGHTS["inventory"] * inventory_score
        + WEIGHTS["sla"] * sla_score
        + WEIGHTS["traffic"] * traffic_score
        + WEIGHTS["demand"] * demand_score
    )


def rank_orders(orders: list) -> pd.DataFrame:
    """
    Score and rank a list of WorkOrder objects by urgency.

    Returns:
        DataFrame sorted by priority_score descending with recommended_action
    """
    records = []
    for order in orders:
        score = score_order(order)
        if score >= 75:
            action = "EMERGENCY"
        elif score >= 50:
            action = "DISPATCH"
        else:
            action = "MONITOR"
        records.append({
            "order_id": order.order_id,
            "station_id": order.station_id,
            "priority_score": round(score, 2),
            "recommended_action": action,
        })

    return (
        pd.DataFrame(records)
        .sort_values("priority_score", ascending=False)
        .reset_index(drop=True)
    )
