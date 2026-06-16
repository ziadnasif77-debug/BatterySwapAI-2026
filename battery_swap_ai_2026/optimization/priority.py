"""
optimization/priority.py
Risk scoring and sensor visit prioritization.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FULL_PREDS_PATH = Path(__file__).parent.parent / "results"          / "full_predictions.csv"
FEATURES_CSV    = Path(__file__).parent.parent / "data" / "processed" / "features_full.csv"
BUILDINGS_PATH  = Path(__file__).parent.parent / "data" / "raw"      / "buildings.csv"
RESULTS_DIR     = Path(__file__).parent.parent / "results"

BUILDING_WEIGHT = {"hospital": 1.5, "office": 1.0, "warehouse": 0.8}


def compute_risk_score(row: pd.Series) -> float:
    """
    Single urgency score 0–100 for each sensor.

    base_score = (p_fail_7d × 60) + (p_fail_3d × 30) + ((100 - voltage_pct) × 0.10)
    Multiplied by building type weight (hospital ×1.5, office ×1.0, warehouse ×0.8).
    Special cases: voltage_pct ≤ 0 → 100 (dead), rul_predicted ≤ 1 → 95 (dying tomorrow).
    """
    voltage_pct   = float(row.get("voltage_pct",   50.0))
    rul_predicted = float(row.get("rul_predicted", 999.0))
    p_fail_3d     = float(row.get("p_fail_3d",       0.0))
    p_fail_7d     = float(row.get("p_fail_7d",       0.0))
    building_type = str(row.get("building_type",  "office"))

    if voltage_pct <= 0:
        return 100.0
    if rul_predicted <= 1:
        return 95.0

    base_score  = (p_fail_7d * 60) + (p_fail_3d * 30) + ((100 - voltage_pct) * 0.10)
    multiplier  = BUILDING_WEIGHT.get(building_type, 1.0)
    return float(np.clip(base_score * multiplier, 0.0, 100.0))


def assign_risk_color(risk_score: float) -> str:
    """
    Color code for visualization:
      100      → '#808080' (gray   = dead)
      > 70     → '#FF0000' (red    = critical)
      > 40     → '#FF8C00' (orange = warning)
      else     → '#00CC44' (green  = safe)
    """
    if risk_score >= 100:
        return "#808080"
    if risk_score > 70:
        return "#FF0000"
    if risk_score > 40:
        return "#FF8C00"
    return "#00CC44"


def compute_priority_score(sensor_row: pd.Series, nearby_sensors_df: pd.DataFrame) -> float:
    """
    Final ranking score (higher = visit sooner).

    priority = risk_score × building_weight
             + cluster_bonus   (×5 per nearby sensor with risk > 60)
             + recency_bonus   (+10 if dying tomorrow, +5 if dying this week)
    """
    risk_score    = float(sensor_row.get("risk_score",    0.0))
    rul_predicted = float(sensor_row.get("rul_predicted", 999.0))
    building_type = str(sensor_row.get("building_type",  "office"))

    building_weight = BUILDING_WEIGHT.get(building_type, 1.0)

    # Cluster bonus: nearby sensors in same building already flagged as risky
    if not nearby_sensors_df.empty and "risk_score" in nearby_sensors_df.columns:
        cluster_bonus = int((nearby_sensors_df["risk_score"] > 60).sum()) * 5
    else:
        cluster_bonus = 0

    # Recency bonus: imminent failures get a boost
    if rul_predicted <= 1:
        recency_bonus = 10
    elif rul_predicted <= 3:
        recency_bonus = 5
    else:
        recency_bonus = 0

    return float(risk_score * building_weight + cluster_bonus + recency_bonus)


def rank_all_sensors(predictions_df: pd.DataFrame, buildings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply risk and priority scoring to all sensors.
    Returns DataFrame sorted by priority_score descending with columns:
        risk_score, risk_color, risk_category, priority_score, priority_rank
    """
    df = predictions_df.copy()

    # Attach building_type if missing
    if "building_type" not in df.columns:
        df = df.merge(buildings_df[["building_id", "building_type"]],
                      on="building_id", how="left")

    # Step 1: risk scores (vectorised apply)
    df["risk_score"] = df.apply(compute_risk_score, axis=1).round(2)
    df["risk_color"] = df["risk_score"].apply(assign_risk_color)

    # Step 2: priority scores (needs inter-sensor cluster info)
    priority_scores = []
    for idx, row in df.iterrows():
        nearby = df[(df["building_id"] == row["building_id"]) & (df.index != idx)]
        priority_scores.append(compute_priority_score(row, nearby))
    df["priority_score"] = np.round(priority_scores, 2)

    # Step 3: risk categories
    def _category(s: float) -> str:
        if s >= 100:
            return "DEAD"
        if s > 70:
            return "CRITICAL"
        if s > 40:
            return "WARNING"
        return "SAFE"

    df["risk_category"] = df["risk_score"].apply(_category)

    # Sort and rank
    df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)
    df["priority_rank"] = range(1, len(df) + 1)

    # Print category summary
    counts = df["risk_category"].value_counts()
    print("\nRisk Category Distribution:")
    for cat in ["DEAD", "CRITICAL", "WARNING", "SAFE"]:
        n = counts.get(cat, 0)
        print(f"  {cat:<10}: {n:>3} sensors")

    return df


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading predictions and feature data...")
    preds_df    = pd.read_csv(FULL_PREDS_PATH, parse_dates=["timestamp"])
    features_df = pd.read_csv(FEATURES_CSV,    parse_dates=["timestamp"])
    buildings_df= pd.read_csv(BUILDINGS_PATH)

    print(f"  Predictions : {len(preds_df):,} rows, {preds_df['sensor_id'].nunique()} sensors")

    # ── Attach voltage_pct from feature matrix ─────────────────────────────
    volt_pct = (features_df[["sensor_id", "timestamp", "voltage_pct"]]
                .dropna(subset=["voltage_pct"]))
    preds_df = preds_df.merge(volt_pct, on=["sensor_id", "timestamp"], how="left")

    # ── Use LATEST reading per sensor (current state) ──────────────────────
    latest = (preds_df.sort_values("timestamp")
              .groupby("sensor_id").last()
              .reset_index())
    print(f"  Latest reading per sensor: {len(latest)} sensors")

    # ── Rank all sensors ───────────────────────────────────────────────────
    print("\nRanking sensors by priority...")
    ranked = rank_all_sensors(latest, buildings_df)

    # ── Top 10 ────────────────────────────────────────────────────────────
    print("\nTop 10 Most Urgent Sensors:")
    print("=" * 90)
    show_cols = ["priority_rank", "sensor_id", "building_id", "building_type",
                 "rul_predicted", "risk_score", "risk_category", "p_fail_7d",
                 "voltage_pct", "priority_score"]
    available = [c for c in show_cols if c in ranked.columns]
    print(ranked[available].head(10).to_string(index=False))
    print("=" * 90)

    # ── Save ──────────────────────────────────────────────────────────────
    out = RESULTS_DIR / "prioritized_sensors.csv"
    ranked.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(ranked)} sensors)")

    # ── Final summary ──────────────────────────────────────────────────────
    counts = ranked["risk_category"].value_counts()
    print("\n" + "=" * 45)
    print("  PRIORITY RANKING COMPLETE")
    print("=" * 45)
    for cat in ["DEAD", "CRITICAL", "WARNING", "SAFE"]:
        n = counts.get(cat, 0)
        print(f"  {cat:<10}: {n} sensors")
    print(f"  Top priority  : {ranked.iloc[0]['sensor_id']}  "
          f"(risk={ranked.iloc[0]['risk_score']:.1f}, "
          f"RUL={ranked.iloc[0]['rul_predicted']:.1f}d, "
          f"type={ranked.iloc[0]['building_type']})")
    print("=" * 45)


if __name__ == "__main__":
    main()
