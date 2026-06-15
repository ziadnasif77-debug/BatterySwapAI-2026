"""
Station operation simulator for BatterySwapAI 2026.

Runs discrete-event simulations of battery swap station operations to
evaluate scheduling strategies, estimate downtime under different demand
scenarios, and stress-test the optimization pipeline against worst-case
conditions before live deployment.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import heapq


@dataclass
class SimEvent:
    """A discrete simulation event."""
    time: float
    event_type: str          # "swap_request" | "restock" | "failure"
    station_id: str
    payload: dict = field(default_factory=dict)

    def __lt__(self, other):
        return self.time < other.time


@dataclass
class StationState:
    """Runtime state of a single battery swap station."""
    station_id: str
    capacity: int
    inventory: int
    is_operational: bool = True
    total_swaps: int = 0
    total_denied: int = 0


class StationSimulator:
    """
    Discrete-event simulator for battery swap station networks.

    Simulates demand arrivals, battery restocking, and unplanned outages
    over a configurable time horizon (in hours).
    """

    def __init__(self, horizon_hours: float = 24.0, seed: int = 42):
        self.horizon = horizon_hours
        self.rng = np.random.default_rng(seed)
        self._stations: dict = {}
        self._event_queue: list = []

    def add_station(self, station_id: str, capacity: int, initial_inventory: int) -> None:
        """Register a station with the simulator."""
        self._stations[station_id] = StationState(station_id, capacity, initial_inventory)

    def _schedule(self, event: SimEvent) -> None:
        heapq.heappush(self._event_queue, event)

    def _seed_demand(self, station_id: str, mean_rate_per_hour: float) -> None:
        """Generate Poisson-distributed swap arrival events."""
        t = 0.0
        while t < self.horizon:
            inter_arrival = self.rng.exponential(1.0 / max(mean_rate_per_hour, 1e-6))
            t += inter_arrival
            if t < self.horizon:
                self._schedule(SimEvent(t, "swap_request", station_id))

    def run(self, demand_rates: dict, restock_hours: float = 4.0) -> pd.DataFrame:
        """
        Run the full simulation.

        Args:
            demand_rates: Dict mapping station_id -> mean swaps per hour
            restock_hours: Hours between automatic restocking events

        Returns:
            DataFrame with per-station summary statistics
        """
        self._event_queue = []
        for sid, rate in demand_rates.items():
            self._seed_demand(sid, rate)
            for t in np.arange(restock_hours, self.horizon, restock_hours):
                self._schedule(SimEvent(float(t), "restock", sid))

        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            station = self._stations.get(event.station_id)
            if station is None:
                continue

            if event.event_type == "swap_request":
                if station.is_operational and station.inventory > 0:
                    station.inventory -= 1
                    station.total_swaps += 1
                else:
                    station.total_denied += 1

            elif event.event_type == "restock":
                station.inventory = station.capacity

        rows = []
        for s in self._stations.values():
            total = s.total_swaps + s.total_denied
            rows.append({
                "station_id": s.station_id,
                "total_swaps": s.total_swaps,
                "total_denied": s.total_denied,
                "fulfillment_rate": s.total_swaps / total if total > 0 else 1.0,
                "final_inventory": s.inventory,
            })
        return pd.DataFrame(rows)
