"""
baseline.py

Simplest possible RUL (Remaining Useful Life) predictor for battery sensors.
Uses linear extrapolation of the last 14 days of voltage readings to estimate
when voltage will cross the dead threshold (2.5V).

THIS IS THE ENEMY SCORE — every future model must beat its MAE.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta


DEAD_THRESHOLD = 2.5
WINDOW_DAYS    = 14
RUL_CAP        = 365
DATA_PATH      = Path(__file__).parent.parent / "data" / "raw" / "sensor_readings.csv"
RESULTS_DIR    = Path(__file__).parent.parent / "results"


def load_sensor_data(data_path: str) -> pd.DataFrame:
    """
    Load sensor_readings.csv, parse timestamps, sort by sensor_id + timestamp.
    """
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    return df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)


def compute_baseline_rul(
    sensor_df: pd.DataFrame,
    dead_threshold: float = DEAD_THRESHOLD,
) -> float:
    """
    Predict RUL for a single sensor using linear voltage extrapolation.

    Steps:
      - Take last 14 days of readings
      - If fewer than 3 readings: return -1 (insufficient data)
      - Fit linear regression: voltage ~ days_elapsed
      - If slope >= 0 (voltage not declining): return 999
      - days_to_dead = (current_voltage - threshold) / |slope|
      - Return capped to [0, 365]
    """
    if sensor_df.empty:
        return -1.0

    last_ts = sensor_df["timestamp"].iloc[-1]
    window  = sensor_df[sensor_df["timestamp"] >= last_ts - timedelta(days=WINDOW_DAYS)].copy()

    if len(window) < 3:
        return -1.0

    t0 = window["timestamp"].iloc[0]
    window["days_elapsed"] = (window["timestamp"] - t0).dt.total_seconds() / 86400.0

    slope, intercept = np.polyfit(window["days_elapsed"].values, window["voltage"].values, 1)

    if slope >= 0:
        return 999.0

    current_voltage = window["voltage"].iloc[-1]
    if current_voltage <= dead_threshold:
        return 0.0

    days_to_dead = (current_voltage - dead_threshold) / abs(slope)
    return float(np.clip(days_to_dead, 0.0, RUL_CAP))


def evaluate_baseline(df: pd.DataFrame) -> dict:
    """
    Evaluate predictions on sensors with known end_of_life_date.

    prediction_date = last alive reading for each dead sensor
    actual_rul      = (end_of_life_date - prediction_date).days

    Returns dict: mae, rmse, bias, n_sensors_evaluated
    """
    dead_ids = df[df["end_of_life_date"].notna()]["sensor_id"].unique()
    actuals, predicted = [], []

    for sid in dead_ids:
        sdf       = df[df["sensor_id"] == sid].sort_values("timestamp")
        eol_date  = pd.to_datetime(
            sdf[sdf["end_of_life_date"].notna()]["end_of_life_date"].iloc[-1]
        )
        alive     = sdf[sdf["end_of_life_date"].isna()]
        if alive.empty:
            continue

        pred_date  = alive["timestamp"].iloc[-1]
        actual_rul = (eol_date - pred_date).days

        pred_rul = compute_baseline_rul(alive)
        if pred_rul in (-1.0, 999.0):
            continue

        actuals.append(actual_rul)
        predicted.append(pred_rul)

    if not actuals:
        print("  No sensors with sufficient data.")
        return {"mae": None, "rmse": None, "bias": None, "n_sensors_evaluated": 0}

    actuals   = np.array(actuals,   dtype=float)
    predicted = np.array(predicted, dtype=float)
    errors    = predicted - actuals

    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    print(f"  Sensors evaluated    : {len(actuals)}")
    print(f"  Actual RUL range     : {int(actuals.min())}–{int(actuals.max())} days")
    print(f"  Predicted RUL range  : {predicted.min():.1f}–{predicted.max():.1f} days")

    return {
        "mae":                 round(mae,  2),
        "rmse":                round(rmse, 2),
        "bias":                round(bias, 2),
        "n_sensors_evaluated": int(len(actuals)),
    }


def run_baseline_on_all_sensors(data_path: str) -> pd.DataFrame:
    """
    Apply compute_baseline_rul to every sensor and return a results DataFrame.

    Columns: sensor_id, building_id, current_voltage,
             predicted_rul_days, predicted_eol_date, method
    """
    df   = load_sensor_data(data_path)
    rows = []

    for sid, sdf in df.groupby("sensor_id"):
        sdf             = sdf.sort_values("timestamp")
        building_id     = sdf["building_id"].iloc[-1]
        current_voltage = round(float(sdf["voltage"].iloc[-1]), 4)
        last_ts         = sdf["timestamp"].iloc[-1]

        rul = compute_baseline_rul(sdf)

        if rul in (-1.0, 999.0):
            predicted_eol = None
        else:
            predicted_eol = (last_ts + timedelta(days=rul)).date().isoformat()

        rows.append({
            "sensor_id":          sid,
            "building_id":        building_id,
            "current_voltage":    current_voltage,
            "predicted_rul_days": round(rul, 1),
            "predicted_eol_date": predicted_eol,
            "method":             "baseline_linear",
        })

    return pd.DataFrame(rows)


def main():
    print("Loading data...")
    df = load_sensor_data(str(DATA_PATH))
    print(f"  {len(df):,} readings | {df['sensor_id'].nunique()} sensors")

    print("\nEvaluating baseline on dead sensors...")
    metrics = evaluate_baseline(df)

    print("\n" + "=" * 42)
    print("        === BASELINE RESULTS ===")
    print("=" * 42)
    print(f"  MAE  : {metrics['mae']:.1f} days  <- ENEMY SCORE TO BEAT")
    print(f"  RMSE : {metrics['rmse']:.1f} days")
    bias_dir = "predicting too late" if (metrics["bias"] or 0) > 0 else "predicting too early"
    print(f"  Bias : {metrics['bias']:.1f} days ({bias_dir})")
    print(f"  N    : {metrics['n_sensors_evaluated']} sensors evaluated")
    print("=" * 42)

    print("\nRunning on all sensors...")
    results = run_baseline_on_all_sensors(str(DATA_PATH))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "baseline_predictions.csv"
    results.to_csv(out_path, index=False)
    print(f"  Saved {len(results)} predictions → {out_path}\n")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
