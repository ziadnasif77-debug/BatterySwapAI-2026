"""
model/features.py
F01 — Voltage Slope Features
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress
from pathlib import Path


DEAD_THRESHOLD = 2.5
DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "sensor_readings.csv"


def compute_voltage_slopes(sensor_df: pd.DataFrame) -> dict:
    """
    Compute voltage slope features from a single sensor's historical readings.

    Parameters
    ----------
    sensor_df : pd.DataFrame
        Rows for one sensor, sorted by timestamp, containing 'timestamp' and 'voltage'.
        All rows must be <= the prediction timestamp (no leakage).

    Returns
    -------
    dict with keys:
        slope_7d, slope_14d, slope_30d, slope_all  (V/day, negative = declining)
        acceleration  (slope_7d - slope_30d; more negative = getting worse faster)
    """
    if sensor_df.empty:
        return {k: np.nan for k in ("slope_7d", "slope_14d", "slope_30d", "slope_all", "acceleration")}

    df = sensor_df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    last_ts = df["timestamp"].iloc[-1]

    def _slope(window_df: pd.DataFrame) -> float:
        if len(window_df) < 3:
            return np.nan
        t0 = window_df["timestamp"].iloc[0]
        days = (window_df["timestamp"] - t0).dt.total_seconds() / 86400.0
        result = linregress(days.values, window_df["voltage"].values)
        return float(result.slope)

    def _window(days_back: int) -> pd.DataFrame:
        cutoff = last_ts - pd.Timedelta(days=days_back)
        return df[df["timestamp"] >= cutoff]

    slope_7d  = _slope(_window(7))
    slope_14d = _slope(_window(14))
    slope_30d = _slope(_window(30))
    slope_all = _slope(df)

    if np.isnan(slope_7d) or np.isnan(slope_30d):
        acceleration = np.nan
    else:
        acceleration = slope_7d - slope_30d

    return {
        "slope_7d":     slope_7d,
        "slope_14d":    slope_14d,
        "slope_30d":    slope_30d,
        "slope_all":    slope_all,
        "acceleration": acceleration,
    }


def add_slope_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add F01 voltage slope columns to the full sensor DataFrame.

    Each row gets slopes computed from all readings for that sensor UP TO
    AND INCLUDING that row's timestamp (strictly backward-looking, no leakage).

    New columns added:
        slope_7d, slope_14d, slope_30d, slope_all, acceleration

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with columns: sensor_id, timestamp, voltage (at minimum).

    Returns
    -------
    pd.DataFrame with five new float columns appended.
    """
    df = df.copy().sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

    result_rows = []

    for sensor_id, sensor_df in df.groupby("sensor_id", sort=False):
        sensor_df = sensor_df.sort_values("timestamp").reset_index(drop=True)

        slopes_list = []
        for i in range(len(sensor_df)):
            past = sensor_df.iloc[: i + 1]
            slopes_list.append(compute_voltage_slopes(past))

        slopes_df = pd.DataFrame(slopes_list, index=sensor_df.index)
        combined  = pd.concat([sensor_df, slopes_df], axis=1)
        result_rows.append(combined)

    out = pd.concat(result_rows, ignore_index=True)
    out = out.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
    return out


# ── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading sensor data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    print(f"  Loaded {len(df):,} rows for {df['sensor_id'].nunique()} sensors")

    print("Computing F01 slope features (this may take ~30s)...")
    df_feat = add_slope_features(df)

    new_cols = ["slope_7d", "slope_14d", "slope_30d", "slope_all", "acceleration"]

    print("\nFirst 5 rows (selected columns):")
    print(df_feat[["sensor_id", "timestamp", "voltage"] + new_cols].head(5).to_string(index=False))

    print(f"\nF01 complete. New columns: {new_cols}")

    # Verify no NaN in last 80% of readings per sensor
    violations = 0
    for sid, grp in df_feat.groupby("sensor_id"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        cutoff = int(len(grp) * 0.20)
        last_80 = grp.iloc[cutoff:]
        nan_counts = last_80[new_cols].isna().sum()
        bad = nan_counts[nan_counts > 0]
        if not bad.empty:
            print(f"  WARNING {sid}: NaNs in last 80% — {bad.to_dict()}")
            violations += 1

    if violations == 0:
        print("NaN check PASSED — no NaNs in last 80% of any sensor's readings.")
    else:
        print(f"NaN check FAILED — {violations} sensor(s) have NaNs in last 80%.")

    print("\nColumn dtypes:")
    print(df_feat[new_cols].dtypes.to_string())
    print(f"\nTotal rows in feature DataFrame: {len(df_feat):,}")
