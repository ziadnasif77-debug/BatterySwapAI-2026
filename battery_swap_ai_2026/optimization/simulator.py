"""
optimization/simulator.py
Simulates real-world outcomes of the scheduling decisions and compares cost scenarios.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.priority  import compute_risk_score, BUILDING_WEIGHT
from optimization.scheduler import (
    build_distance_matrix, create_vrp_data,
    solve_vrp, format_work_orders, SHIFT_START,
)

FULL_PREDS_PATH  = Path(__file__).parent.parent / "results"           / "full_predictions.csv"
TRAVEL_PATH      = Path(__file__).parent.parent / "data" / "raw"       / "travel_times.csv"
BUILDINGS_PATH   = Path(__file__).parent.parent / "data" / "raw"       / "buildings.csv"
DATA_PATH        = Path(__file__).parent.parent / "data" / "raw"       / "sensor_readings.csv"
RESULTS_DIR      = Path(__file__).parent.parent / "results"

LABOR_COST_PER_HOUR    =  500   # NOK
DOWNTIME_COST_PER_HOUR = 2000   # NOK
DEFAULT_MISSED_DOWNTIME =  48   # hours assumed if not scheduled


def simulate_outcomes(
    predictions_df:   pd.DataFrame,
    work_orders_df:   pd.DataFrame,
    actual_eol_dates: dict,
) -> pd.DataFrame:
    """
    For each sensor (latest prediction):
      - scheduled + visit before death → SUCCESS (0 downtime)
      - scheduled + already dead at visit → FAILURE (hours since death)
      - not scheduled + rul ≤ 30 days   → MISSED (DEFAULT_MISSED_DOWNTIME hours)
      - not scheduled + rul > 30 days   → LOW_PRIORITY (0 downtime)

    Depot sensor (B001) is auto-serviced at shift start.
    """
    # Latest prediction per sensor
    latest = (predictions_df
              .sort_values("timestamp")
              .groupby("sensor_id").last()
              .reset_index())

    # Build sensor → work-order lookup
    scheduled_map: dict[str, pd.Series] = {}
    if not work_orders_df.empty:
        for _, wo in work_orders_df.iterrows():
            for sid in str(wo["sensor_ids"]).split(","):
                sid = sid.strip()
                if sid:
                    scheduled_map[sid] = wo

    # Depot sensors auto-serviced
    depot_sensors = set(latest.loc[latest["building_id"] == "B001", "sensor_id"])

    rows = []
    for _, row in latest.iterrows():
        sid          = row["sensor_id"]
        rul          = float(row.get("rul_predicted", 999))
        building_id  = str(row.get("building_id", ""))
        actual_eol   = actual_eol_dates.get(sid)
        is_depot     = building_id == "B001"

        if sid in scheduled_map or is_depot:
            if is_depot and sid not in scheduled_map:
                planned_visit = SHIFT_START
            else:
                wo = scheduled_map[sid]
                arr_time      = datetime.strptime(wo["arrival_time"], "%H:%M")
                sched_date    = pd.to_datetime(row["timestamp"]).date()
                planned_visit = datetime.combine(sched_date, arr_time.time())

            if rul <= 0:
                outcome        = "FAILURE"
                downtime_hours = abs(rul) * 24.0
            elif actual_eol is not None and planned_visit.date() > actual_eol.date():
                # Compare at day-level: visiting after death day = failure
                downtime_days  = (planned_visit.date() - actual_eol.date()).days
                outcome        = "FAILURE"
                downtime_hours = downtime_days * 24.0
            else:
                outcome        = "SUCCESS"
                downtime_hours = 0.0
        else:
            planned_visit = None
            if rul <= 30:
                outcome        = "MISSED"
                downtime_hours = float(DEFAULT_MISSED_DOWNTIME)
            else:
                outcome        = "LOW_PRIORITY"
                downtime_hours = 0.0

        rows.append({
            "sensor_id":      sid,
            "building_id":    building_id,
            "scheduled":      sid in scheduled_map or is_depot,
            "planned_visit":  planned_visit,
            "actual_eol":     actual_eol,
            "outcome":        outcome,
            "downtime_hours": downtime_hours,
            "rul_predicted":  rul,
        })

    return pd.DataFrame(rows)


def compute_costs(outcomes_df: pd.DataFrame, work_orders_df: pd.DataFrame) -> dict:
    """
    Cost model (Norwegian field service):
      labor    = total_travel_hours × 500 NOK
      downtime = total_downtime_hours × 2000 NOK
    Travel time derived from work order arrival/departure times.
    """
    total_downtime_hours = float(outcomes_df["downtime_hours"].sum())

    # Travel time: sum of gaps between departure of prev stop and arrival of next,
    # plus time from 08:00 to first stop, per worker
    total_travel_min = 0.0
    if not work_orders_df.empty:
        for wid in work_orders_df["worker_id"].unique():
            wo = work_orders_df[work_orders_df["worker_id"] == wid].sort_values("stop_number")
            for i, r in wo.reset_index().iterrows():
                arr = datetime.strptime(r["arrival_time"], "%H:%M")
                if i == 0:
                    dep_prev = SHIFT_START
                else:
                    dep_prev = datetime.strptime(wo.iloc[i - 1]["departure_time"], "%H:%M")
                gap = (arr - dep_prev).total_seconds() / 60
                total_travel_min += max(0, gap)

    total_travel_hours = total_travel_min / 60.0
    labor_cost         = total_travel_hours * LABOR_COST_PER_HOUR
    downtime_cost      = total_downtime_hours * DOWNTIME_COST_PER_HOUR
    total_cost         = labor_cost + downtime_cost

    return {
        "total_travel_hours":   round(total_travel_hours, 2),
        "total_downtime_hours": round(total_downtime_hours, 2),
        "labor_cost":           round(labor_cost),
        "downtime_cost":        round(downtime_cost),
        "total_cost":           round(total_cost),
    }


def run_scenario(
    threshold:       float,
    label:           str,
    predictions_df:  pd.DataFrame,
    travel_df:       pd.DataFrame,
    buildings_df:    pd.DataFrame,
    actual_eol:      dict,
    n_workers:       int = 3,
) -> dict:
    """
    Run full scheduling + simulation pipeline for a given p_fail_7d threshold.
    Only sensors with p_fail_7d > threshold are scheduled.
    """
    # Latest prediction per sensor with risk score
    latest = (predictions_df
              .sort_values("timestamp")
              .groupby("sensor_id").last()
              .reset_index())
    latest["risk_score"] = latest.apply(compute_risk_score, axis=1)

    # Filter by threshold
    filtered = latest[latest["p_fail_7d"] > threshold].copy()
    n_candidates = len(filtered)

    if filtered.empty:
        work_orders = pd.DataFrame()
        solution    = {}
    else:
        try:
            vrp_data    = create_vrp_data(filtered, travel_df, n_workers)
            solution    = solve_vrp(vrp_data, time_limit_seconds=15)
            work_orders = (format_work_orders(solution, buildings_df, filtered)
                           if solution else pd.DataFrame())
        except Exception as e:
            print(f"  [{label}] Scheduler error: {e}")
            work_orders = pd.DataFrame()
            solution    = {}

    # Simulate outcomes for ALL sensors (not just filtered)
    outcomes  = simulate_outcomes(latest, work_orders, actual_eol)
    costs     = compute_costs(outcomes, work_orders)

    n_success  = int((outcomes["outcome"] == "SUCCESS").sum())
    n_missed   = int((outcomes["outcome"] == "MISSED").sum())
    n_failure  = int((outcomes["outcome"] == "FAILURE").sum())
    n_total    = len(outcomes)
    pct_saved  = round(n_success / n_total * 100, 1) if n_total else 0

    return {
        "label":          label,
        "threshold":      threshold,
        "n_candidates":   n_candidates,
        "n_scheduled":    len(work_orders),
        "n_success":      n_success,
        "n_missed":       n_missed,
        "n_failure":      n_failure,
        "pct_saved":      pct_saved,
        "travel_hours":   costs["total_travel_hours"],
        "downtime_hours": costs["total_downtime_hours"],
        "labor_cost":     costs["labor_cost"],
        "downtime_cost":  costs["downtime_cost"],
        "total_cost":     costs["total_cost"],
    }


def compare_scenarios(
    predictions_df: pd.DataFrame,
    travel_df:      pd.DataFrame,
    buildings_df:   pd.DataFrame,
    actual_eol:     dict,
) -> list:
    """
    Run three scheduling strategies and compare costs.

    Scenario 1 — AGGRESSIVE:    p_fail_7d > 0.30
    Scenario 2 — NORMAL:        p_fail_7d > 1.00
    Scenario 3 — CONSERVATIVE:  p_fail_7d > 5.00

    Prints comparison table and recommends lowest-cost strategy.
    """
    scenarios = [
        (0.30, "AGGRESSIVE"),
        (1.00, "NORMAL"),
        (5.00, "CONSERVATIVE"),
    ]

    results = []
    for threshold, label in scenarios:
        print(f"\n  Running scenario: {label} (threshold = {threshold}%)...")
        r = run_scenario(threshold, label, predictions_df, travel_df,
                         buildings_df, actual_eol)
        results.append(r)

    # Find recommended (lowest total cost)
    best = min(results, key=lambda x: x["total_cost"])
    for r in results:
        r["recommended"] = (r["label"] == best["label"])

    # Print comparison table
    W = 15
    print("\n" + "=" * 68)
    print(f"  {'Metric':<26} {'AGGRESSIVE':>{W}} {'NORMAL':>{W}} {'CONSERVATIVE':>{W}}")
    print("=" * 68)
    metrics = [
        ("Sensors qualified",  "n_candidates",  ""),
        ("Stops scheduled",    "n_scheduled",   ""),
        ("Sensors saved (%)",  "pct_saved",     "%"),
        ("Travel (hrs)",       "travel_hours",  ""),
        ("Downtime (hrs)",     "downtime_hours",""),
        ("Labor cost (NOK)",   "labor_cost",    ""),
        ("Downtime cost (NOK)","downtime_cost", ""),
        ("TOTAL COST (NOK)",   "total_cost",    ""),
    ]
    for display, key, unit in metrics:
        vals = []
        for r in results:
            v = r[key]
            if isinstance(v, float):
                vals.append(f"{v:.1f}{unit}")
            else:
                vals.append(f"{v:,}{unit}")
        print(f"  {display:<26} {vals[0]:>{W}} {vals[1]:>{W}} {vals[2]:>{W}}")

    rec_col = [("  ✅ RECOMMENDED" if r["recommended"] else "")
               for r in results]
    print(f"  {'RECOMMENDED?':<26} {rec_col[0]:>{W}} {rec_col[1]:>{W}} {rec_col[2]:>{W}}")
    print("=" * 68)

    # Save JSON
    out_json = RESULTS_DIR / "scenario_comparison.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {out_json}")

    print(f"\n  RECOMMENDED STRATEGY: {best['label']} "
          f"with total cost {best['total_cost']:,} NOK")

    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    print("Loading data...")
    preds_df    = pd.read_csv(FULL_PREDS_PATH, parse_dates=["timestamp"])
    travel_df   = pd.read_csv(TRAVEL_PATH)
    buildings_df= pd.read_csv(BUILDINGS_PATH)
    raw_df      = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    # Actual EOL dates from raw data
    actual_eol = {
        sid: pd.to_datetime(eol)
        for sid, eol in (
            raw_df[raw_df["end_of_life_date"].notna()]
            .groupby("sensor_id")["end_of_life_date"]
            .first()
            .items()
        )
    }
    print(f"  Predictions: {len(preds_df):,} rows, {preds_df['sensor_id'].nunique()} sensors")
    print(f"  Actual EOL dates: {len(actual_eol)} sensors")
    print(f"  p_fail_7d distribution (latest per sensor):")
    latest_pfail = (preds_df.sort_values("timestamp")
                   .groupby("sensor_id")["p_fail_7d"].last()
                   .sort_values(ascending=False))
    for sid, v in latest_pfail.items():
        print(f"    {sid}: {v:.1f}%")

    # ── Run scenarios ─────────────────────────────────────────────────────
    results = compare_scenarios(preds_df, travel_df, buildings_df, actual_eol)

    # ── Final print ───────────────────────────────────────────────────────
    best = min(results, key=lambda x: x["total_cost"])
    print(f"\nRECOMMENDED STRATEGY: {best['label']} "
          f"with total cost {best['total_cost']:,} NOK")


if __name__ == "__main__":
    main()
