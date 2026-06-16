"""
optimization/scheduler.py
Vehicle Routing Problem (VRP) scheduler for battery swap field workers.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

sys.path.insert(0, str(Path(__file__).parent.parent))

PRIORITIZED_PATH = Path(__file__).parent.parent / "results"          / "prioritized_sensors.csv"
TRAVEL_PATH      = Path(__file__).parent.parent / "data" / "raw"      / "travel_times.csv"
BUILDINGS_PATH   = Path(__file__).parent.parent / "data" / "raw"      / "buildings.csv"
RESULTS_DIR      = Path(__file__).parent.parent / "results"

SHIFT_START      = datetime(2026, 6, 16, 8, 0)   # 08:00
SHIFT_MINUTES    = 480                             # 8-hour shift
SERVICE_MINUTES  = 15                              # time per stop


def build_distance_matrix(buildings_to_visit: list, travel_times_df: pd.DataFrame) -> list:
    """
    Converts travel_times_df into OR-Tools format (2D list of integer minutes).
    Index 0 in the returned matrix is the depot (first building in the list).

    Parameters
    ----------
    buildings_to_visit : list of building_id strings, depot first
    travel_times_df    : full 20×20 travel time DataFrame

    Returns
    -------
    2D list[int] — travel times rounded to nearest minute
    """
    n = len(buildings_to_visit)
    idx = {b: i for i, b in enumerate(buildings_to_visit)}

    # Build lookup: (from, to) → minutes
    travel_lookup = {
        (row["from_building_id"], row["to_building_id"]): row["travel_minutes"]
        for _, row in travel_times_df.iterrows()
    }

    matrix = []
    for b_from in buildings_to_visit:
        row = []
        for b_to in buildings_to_visit:
            if b_from == b_to:
                row.append(0)
            else:
                minutes = travel_lookup.get((b_from, b_to), 9999)
                row.append(int(round(minutes)))
        matrix.append(row)

    return matrix


def create_vrp_data(
    prioritized_sensors_df: pd.DataFrame,
    travel_times_df: pd.DataFrame,
    n_workers: int = 3,
) -> dict:
    """
    Packages all inputs for OR-Tools VRP.

    Only buildings with sensors at risk_score > 40 are included as stops.
    Depot = B001 (index 0).

    Penalties:
      risk > 85  → 10000 (must visit)
      risk > 60  → 1000  (should visit)
      else       → 100   (optional)
    """
    df = prioritized_sensors_df.copy()

    # Required + optional stops (risk > 40)
    stops_df = df[df["risk_score"] > 40].copy()

    # Unique buildings to visit
    visit_buildings = stops_df["building_id"].unique().tolist()

    # Depot: B001 (or first available building if B001 not in list)
    depot_id = "B001"
    if depot_id not in visit_buildings:
        visit_buildings = [depot_id] + visit_buildings
    else:
        visit_buildings.remove(depot_id)
        visit_buildings = [depot_id] + visit_buildings

    n_locs = len(visit_buildings)
    building_index = {b: i for i, b in enumerate(visit_buildings)}

    # Distance matrix
    distance_matrix = build_distance_matrix(visit_buildings, travel_times_df)

    # Time windows: full 8-hour shift for every location
    time_windows = [(0, SHIFT_MINUTES)] * n_locs

    # Penalties per location (skip depot — it's index 0)
    penalties = {}
    for _, row in stops_df.iterrows():
        b = row["building_id"]
        if b == depot_id:
            continue
        idx = building_index[b]
        risk = float(row["risk_score"])
        if risk > 85:
            penalties[idx] = 10000
        elif risk > 60:
            penalties[idx] = 1000
        else:
            penalties[idx] = 100

    # Sensor mapping per building (for work orders)
    sensor_map = (
        stops_df.groupby("building_id")["sensor_id"]
        .apply(list)
        .to_dict()
    )

    return {
        "distance_matrix": distance_matrix,
        "num_vehicles":    n_workers,
        "depot":           0,
        "time_windows":    time_windows,
        "service_time":    SERVICE_MINUTES,
        "penalties":       penalties,
        "visit_buildings": visit_buildings,
        "building_index":  building_index,
        "sensor_map":      sensor_map,
        "stops_df":        stops_df,
    }


def solve_vrp(vrp_data: dict, time_limit_seconds: int = 30) -> dict:
    """
    Runs OR-Tools Time-Constrained VRP.

    Strategy:
      First solution  : PATH_CHEAPEST_ARC
      Improvement     : GUIDED_LOCAL_SEARCH
      Time limit      : time_limit_seconds

    Returns
    -------
    dict: worker_id (int) → list of (building_id, arrival_min, departure_min)
    Empty dict if no solution found.
    """
    matrix          = vrp_data["distance_matrix"]
    n_vehicles      = vrp_data["num_vehicles"]
    depot           = vrp_data["depot"]
    time_windows    = vrp_data["time_windows"]
    service_time    = vrp_data["service_time"]
    penalties       = vrp_data["penalties"]
    visit_buildings = vrp_data["visit_buildings"]

    manager = pywrapcp.RoutingIndexManager(len(matrix), n_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # Transit callback (travel time + service time)
    def time_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node   = manager.IndexToNode(to_idx)
        travel    = matrix[from_node][to_node]
        svc       = service_time if from_node != depot else 0
        return travel + svc

    transit_cb_idx = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    # Time dimension
    routing.AddDimension(
        transit_cb_idx,
        slack_max=60,           # allow 60-min waiting at stops
        capacity=SHIFT_MINUTES,
        fix_start_cumul_to_zero=True,
        name="Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # Time windows
    for loc_idx, (start, end) in enumerate(time_windows):
        index = manager.NodeToIndex(loc_idx)
        time_dim.CumulVar(index).SetRange(start, end)

    # Penalties for skippable locations
    for loc_idx, penalty in penalties.items():
        routing.AddDisjunction([manager.NodeToIndex(loc_idx)], penalty)

    # Search parameters
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = time_limit_seconds
    search_params.log_search = False

    solution = routing.SolveWithParameters(search_params)

    if not solution:
        print("WARNING: OR-Tools found no feasible solution.")
        return {}

    result = {}
    for vehicle_id in range(n_vehicles):
        stops = []
        index = routing.Start(vehicle_id)

        while not routing.IsEnd(index):
            node       = manager.IndexToNode(index)
            arrival    = solution.Min(time_dim.CumulVar(index))
            departure  = arrival + (service_time if node != depot else 0)

            if node != depot:
                stops.append((visit_buildings[node], arrival, departure))

            index = solution.Value(routing.NextVar(index))

        if stops:
            result[vehicle_id] = stops

    return result


def format_work_orders(
    solution:      dict,
    buildings_df:  pd.DataFrame,
    sensors_df:    pd.DataFrame,
) -> pd.DataFrame:
    """
    Converts OR-Tools solution to human-readable work orders.

    Returns DataFrame with columns:
        worker_id, stop_number, building_id, building_name,
        arrival_time, departure_time, n_batteries_to_replace, sensor_ids
    """
    building_name = dict(zip(buildings_df["building_id"], buildings_df["building_name"]))
    sensor_map    = (
        sensors_df.groupby("building_id")["sensor_id"]
        .apply(list)
        .to_dict()
    )

    rows = []
    for worker_id, stops in solution.items():
        for stop_num, (building_id, arrival_min, departure_min) in enumerate(stops, start=1):
            arrival_dt   = SHIFT_START + timedelta(minutes=int(arrival_min))
            departure_dt = SHIFT_START + timedelta(minutes=int(departure_min))
            sensor_ids   = sensor_map.get(building_id, [])
            rows.append({
                "worker_id":             worker_id + 1,
                "stop_number":           stop_num,
                "building_id":           building_id,
                "building_name":         building_name.get(building_id, building_id),
                "arrival_time":          arrival_dt.strftime("%H:%M"),
                "departure_time":        departure_dt.strftime("%H:%M"),
                "n_batteries_to_replace": len(sensor_ids),
                "sensor_ids":            ", ".join(sensor_ids),
            })

    return pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    print("Loading data...")
    sensors_df   = pd.read_csv(PRIORITIZED_PATH)
    travel_df    = pd.read_csv(TRAVEL_PATH)
    buildings_df = pd.read_csv(BUILDINGS_PATH)
    print(f"  Sensors to consider : {len(sensors_df)}")
    print(f"  Risk > 60 (required): {(sensors_df['risk_score'] > 60).sum()}")
    print(f"  Risk 40-60 (optional): {((sensors_df['risk_score'] > 40) & (sensors_df['risk_score'] <= 60)).sum()}")

    # ── Build VRP ─────────────────────────────────────────────────────────
    print("\nBuilding VRP problem...")
    vrp_data = create_vrp_data(sensors_df, travel_df, n_workers=3)
    n_stops  = len(vrp_data["visit_buildings"]) - 1   # exclude depot
    print(f"  Locations (incl. depot): {len(vrp_data['visit_buildings'])}")
    print(f"  Worker stops          : {n_stops}")

    # ── Solve ─────────────────────────────────────────────────────────────
    print(f"\nSolving VRP (3 workers, 30-second limit)...")
    solution = solve_vrp(vrp_data, time_limit_seconds=30)

    if not solution:
        print("No solution found — check that travel times and shift window allow coverage.")
        return

    # ── Format work orders ────────────────────────────────────────────────
    work_orders = format_work_orders(solution, buildings_df, sensors_df)

    # ── Print schedule ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  WORK ORDERS FOR TODAY")
    print("=" * 65)
    for worker_id in sorted(work_orders["worker_id"].unique()):
        wo = work_orders[work_orders["worker_id"] == worker_id]
        print(f"\nWorker {worker_id}:")
        for _, row in wo.iterrows():
            print(f"  {row['arrival_time']} → {row['building_name']} "
                  f"({row['building_id']}) — {row['n_batteries_to_replace']} "
                  f"batter{'y' if row['n_batteries_to_replace']==1 else 'ies'}  "
                  f"[sensors: {row['sensor_ids']}]")
    print("=" * 65)

    # ── Summary ───────────────────────────────────────────────────────────
    total_stops = len(work_orders)
    total_travel = sum(
        vrp_data["distance_matrix"]
        [vrp_data["building_index"].get(row["building_id"], 0)]
        [vrp_data["building_index"].get(
            work_orders[work_orders["worker_id"] == row["worker_id"]]
            .iloc[0]["building_id"], 0
        )]
        for _, row in work_orders.iterrows()
    )

    critical_buildings = set(
        sensors_df[sensors_df["risk_score"] > 60]["building_id"].tolist()
    )
    visited_buildings  = set(work_orders["building_id"].tolist())
    covered = len(critical_buildings & visited_buildings)

    # Compute total travel time from work order arrival times
    total_travel_min = 0
    for worker_id in sorted(work_orders["worker_id"].unique()):
        wo = work_orders[work_orders["worker_id"] == worker_id].reset_index(drop=True)
        for i, row in wo.iterrows():
            if i == 0:
                arr = datetime.strptime(row["arrival_time"], "%H:%M")
                total_travel_min += (arr - SHIFT_START.replace(hour=8, minute=0)).seconds // 60
            else:
                prev_dep = datetime.strptime(wo.iloc[i-1]["departure_time"], "%H:%M")
                arr      = datetime.strptime(row["arrival_time"],            "%H:%M")
                total_travel_min += max(0, (arr - prev_dep).seconds // 60)

    th, tm = divmod(total_travel_min, 60)

    print(f"\n  Total stops          : {total_stops}")
    print(f"  Total travel time    : {th}h {tm}min")
    print(f"  Critical sensors covered: {covered} / {len(critical_buildings)} "
          f"({'100%' if len(critical_buildings)==0 else f'{covered/len(critical_buildings)*100:.0f}%'})")

    # ── Save ──────────────────────────────────────────────────────────────
    out = RESULTS_DIR / "work_orders.csv"
    work_orders.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(work_orders)} work orders)")


if __name__ == "__main__":
    main()
