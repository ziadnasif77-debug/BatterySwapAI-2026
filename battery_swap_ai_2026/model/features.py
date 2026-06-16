"""
model/features.py
F01 — Voltage Slope Features
F02 — Rolling Voltage Statistics
F03 — Temperature Impact Features
F04 — Lifecycle Position Features
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


def compute_rolling_voltage_stats(sensor_df: pd.DataFrame) -> dict:
    """
    Computes statistical summary of voltage in rolling windows.

    Physics:
    - mean = average health level
    - std  = instability (high std = erratic battery = near death)
    - min  = worst moment (lowest voltage reached recently)
    - drop = total deterioration in window

    Windows: 7d, 14d, 30d  →  voltage_mean_{w}d, voltage_std_{w}d,
                               voltage_min_{w}d, voltage_drop_{w}d

    Overall (no window):
    - voltage_current   : latest voltage reading
    - voltage_max_ever  : highest voltage ever recorded
    - voltage_min_ever  : lowest voltage ever recorded
    - voltage_range_all : max_ever − min_ever
    - voltage_pct       : (current − 2.5) / (max_ever − 2.5) × 100
                          (100% = full, 0% = dead)
    """
    nan_keys = (
        [f"voltage_{stat}_{w}d" for w in (7, 14, 30) for stat in ("mean", "std", "min", "drop")]
        + ["voltage_current", "voltage_max_ever", "voltage_min_ever",
           "voltage_range_all", "voltage_pct"]
    )
    if sensor_df.empty:
        return {k: np.nan for k in nan_keys}

    df = sensor_df.copy().sort_values("timestamp").reset_index(drop=True)
    last_ts = df["timestamp"].iloc[-1]

    result = {}

    for w in (7, 14, 30):
        cutoff = last_ts - pd.Timedelta(days=w)
        win = df[df["timestamp"] >= cutoff]["voltage"]

        if len(win) < 2:
            result[f"voltage_mean_{w}d"] = np.nan
            result[f"voltage_std_{w}d"]  = np.nan
            result[f"voltage_min_{w}d"]  = np.nan
            result[f"voltage_drop_{w}d"] = np.nan
        else:
            result[f"voltage_mean_{w}d"] = float(win.mean())
            result[f"voltage_std_{w}d"]  = float(win.std(ddof=1))
            result[f"voltage_min_{w}d"]  = float(win.min())
            result[f"voltage_drop_{w}d"] = float(win.iloc[0] - win.iloc[-1])

    current   = float(df["voltage"].iloc[-1])
    max_ever  = float(df["voltage"].max())
    min_ever  = float(df["voltage"].min())
    denom     = max_ever - DEAD_THRESHOLD
    pct       = (current - DEAD_THRESHOLD) / denom * 100.0 if denom > 0 else np.nan

    result["voltage_current"]   = current
    result["voltage_max_ever"]  = max_ever
    result["voltage_min_ever"]  = min_ever
    result["voltage_range_all"] = max_ever - min_ever
    result["voltage_pct"]       = float(np.clip(pct, 0.0, 100.0)) if not np.isnan(pct) else np.nan

    return result


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add F02 rolling voltage stat columns to the full sensor DataFrame.

    Same backward-looking pattern as add_slope_features: each row receives
    stats computed only from readings up to and including its own timestamp.

    New columns (13 total):
        voltage_mean_7d,  voltage_std_7d,  voltage_min_7d,  voltage_drop_7d
        voltage_mean_14d, voltage_std_14d, voltage_min_14d, voltage_drop_14d
        voltage_mean_30d, voltage_std_30d, voltage_min_30d, voltage_drop_30d
        voltage_current, voltage_max_ever, voltage_min_ever,
        voltage_range_all, voltage_pct
    """
    df = df.copy().sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

    result_rows = []

    for sensor_id, sensor_df in df.groupby("sensor_id", sort=False):
        sensor_df = sensor_df.sort_values("timestamp").reset_index(drop=True)

        stats_list = []
        for i in range(len(sensor_df)):
            past = sensor_df.iloc[: i + 1]
            stats_list.append(compute_rolling_voltage_stats(past))

        stats_df = pd.DataFrame(stats_list, index=sensor_df.index)
        combined  = pd.concat([sensor_df, stats_df], axis=1)
        result_rows.append(combined)

    out = pd.concat(result_rows, ignore_index=True)
    out = out.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
    return out


# ── Test ─────────────────────────────────────────────────────────────────────

WINTER_MONTHS = {12, 1, 2, 3}


def compute_temperature_features(sensor_df: pd.DataFrame) -> dict:
    """
    Captures how Norwegian cold affects this specific battery.

    Physics:
    - Cold temperatures reduce battery chemical activity
    - Below 0°C: lithium batteries lose 20-30% capacity
    - Below -10°C: severe degradation, accelerated aging
    - Buildings in Bergen get more cold-wet, Oslo gets cold-dry
    - The correlation between voltage drop and temperature
      tells us how cold-sensitive THIS specific battery is
    """
    nan_keys = [
        "temp_mean_all", "temp_mean_14d", "temp_min_ever", "temp_std_all",
        "temp_current", "days_below_0", "days_below_minus10",
        "cold_exposure_pct", "is_winter_now", "volt_temp_corr",
    ]
    if sensor_df.empty:
        return {k: np.nan for k in nan_keys}

    df = sensor_df.copy().sort_values("timestamp").reset_index(drop=True)
    temps   = df["temperature"]
    last_ts = df["timestamp"].iloc[-1]

    temp_mean_all = float(temps.mean())
    temp_std_all  = float(temps.std(ddof=1)) if len(temps) > 1 else np.nan
    temp_min_ever = float(temps.min())
    temp_current  = float(temps.iloc[-1])

    cutoff_14d    = last_ts - pd.Timedelta(days=14)
    win_14        = df[df["timestamp"] >= cutoff_14d]["temperature"]
    temp_mean_14d = float(win_14.mean()) if len(win_14) >= 1 else np.nan

    days_below_0       = int((temps < 0).sum())
    days_below_minus10 = int((temps < -10).sum())
    total_days         = len(df)
    cold_exposure_pct  = days_below_0 / total_days * 100.0

    is_winter_now = 1 if last_ts.month in WINTER_MONTHS else 0

    if len(df) >= 5:
        v_std = float(df["voltage"].std())
        t_std = float(df["temperature"].std())
        if v_std == 0 or t_std == 0:
            volt_temp_corr = 0.0  # no variance → assume no detectable correlation
        else:
            corr_val = df[["voltage", "temperature"]].corr().iloc[0, 1]
            volt_temp_corr = float(corr_val) if not np.isnan(corr_val) else 0.0
    else:
        volt_temp_corr = np.nan

    return {
        "temp_mean_all":       temp_mean_all,
        "temp_mean_14d":       temp_mean_14d,
        "temp_min_ever":       temp_min_ever,
        "temp_std_all":        temp_std_all,
        "temp_current":        temp_current,
        "days_below_0":        days_below_0,
        "days_below_minus10":  days_below_minus10,
        "cold_exposure_pct":   cold_exposure_pct,
        "is_winter_now":       float(is_winter_now),
        "volt_temp_corr":      volt_temp_corr,
    }


def add_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add F03 temperature impact columns to the full sensor DataFrame.

    Strictly backward-looking: each row receives features computed only
    from readings up to and including its own timestamp.

    New columns (10 total):
        temp_mean_all, temp_mean_14d, temp_min_ever, temp_std_all,
        temp_current, days_below_0, days_below_minus10,
        cold_exposure_pct, is_winter_now, volt_temp_corr
    """
    df = df.copy().sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

    result_rows = []

    for sensor_id, sensor_df in df.groupby("sensor_id", sort=False):
        sensor_df = sensor_df.sort_values("timestamp").reset_index(drop=True)

        feat_list = []
        for i in range(len(sensor_df)):
            past = sensor_df.iloc[: i + 1]
            feat_list.append(compute_temperature_features(past))

        feat_df  = pd.DataFrame(feat_list, index=sensor_df.index)
        combined = pd.concat([sensor_df, feat_df], axis=1)
        result_rows.append(combined)

    out = pd.concat(result_rows, ignore_index=True)
    out = out.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
    return out


def compute_lifecycle_features(sensor_df: pd.DataFrame) -> dict:
    """
    Captures WHERE in the battery's total lifespan we currently are.

    Physics:
    - A battery at 3.0V that is 2 months old is very different
      from a battery at 3.0V that is 18 months old
    - Age normalizes voltage to give true health picture
    - lifecycle_position: 0.0 = brand new, 1.0 = at death threshold
    """
    nan_keys = [
        "battery_age_days", "lifecycle_position", "readings_per_day",
        "days_since_last_reading", "voltage_at_30d", "voltage_at_60d",
        "voltage_at_90d", "month_of_year",
    ]
    if sensor_df.empty:
        return {k: np.nan for k in nan_keys}

    df = sensor_df.copy().sort_values("timestamp").reset_index(drop=True)

    first_ts = df["timestamp"].iloc[0]
    last_ts  = df["timestamp"].iloc[-1]

    battery_age_days = (last_ts - first_ts).days
    readings_per_day = len(df) / battery_age_days if battery_age_days > 0 else np.nan

    # days_since_last_reading: use last_ts as "now" (strictly backward-looking)
    days_since_last_reading = 0.0  # the last row IS the current reading

    v_current = float(df["voltage"].iloc[-1])
    v_max     = float(df["voltage"].max())
    denom     = v_max - DEAD_THRESHOLD
    if denom > 0:
        lc = 1.0 - (v_current - DEAD_THRESHOLD) / denom
        lifecycle_position = float(np.clip(lc, 0.0, 1.0))
    else:
        lifecycle_position = np.nan

    def _voltage_at(days_back: int) -> float:
        target = last_ts - pd.Timedelta(days=days_back)
        candidates = df[df["timestamp"] <= target]
        if candidates.empty:
            return np.nan
        # closest reading on or before the target date
        closest = candidates.iloc[-1]
        # only return if within ±3 days of target (data is daily)
        gap = abs((closest["timestamp"] - target).days)
        return float(closest["voltage"]) if gap <= 3 else np.nan

    voltage_at_30d = _voltage_at(30)
    voltage_at_60d = _voltage_at(60)
    voltage_at_90d = _voltage_at(90)

    month_of_year = float(last_ts.month)

    return {
        "battery_age_days":        float(battery_age_days),
        "lifecycle_position":      lifecycle_position,
        "readings_per_day":        readings_per_day,
        "days_since_last_reading": days_since_last_reading,
        "voltage_at_30d":          voltage_at_30d,
        "voltage_at_60d":          voltage_at_60d,
        "voltage_at_90d":          voltage_at_90d,
        "month_of_year":           month_of_year,
    }


def add_lifecycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add F04 lifecycle position columns to the full sensor DataFrame.

    Strictly backward-looking: each row receives features computed only
    from readings up to and including its own timestamp.

    New columns (8 total):
        battery_age_days, lifecycle_position, readings_per_day,
        days_since_last_reading, voltage_at_30d, voltage_at_60d,
        voltage_at_90d, month_of_year
    """
    df = df.copy().sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

    result_rows = []

    for sensor_id, sensor_df in df.groupby("sensor_id", sort=False):
        sensor_df = sensor_df.sort_values("timestamp").reset_index(drop=True)

        feat_list = []
        for i in range(len(sensor_df)):
            past = sensor_df.iloc[: i + 1]
            feat_list.append(compute_lifecycle_features(past))

        feat_df  = pd.DataFrame(feat_list, index=sensor_df.index)
        combined = pd.concat([sensor_df, feat_df], axis=1)
        result_rows.append(combined)

    out = pd.concat(result_rows, ignore_index=True)
    out = out.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
    return out


def _nan_check(df_feat: pd.DataFrame, cols: list, label: str) -> int:
    violations = 0
    for sid, grp in df_feat.groupby("sensor_id"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        cutoff = int(len(grp) * 0.20)
        last_80 = grp.iloc[cutoff:]
        nan_counts = last_80[cols].isna().sum()
        bad = nan_counts[nan_counts > 0]
        if not bad.empty:
            print(f"  WARNING {sid} [{label}]: NaNs in last 80% — {bad.to_dict()}")
            violations += 1
    return violations


if __name__ == "__main__":
    print("Loading sensor data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    print(f"  Loaded {len(df):,} rows for {df['sensor_id'].nunique()} sensors")

    # ── F01 ──────────────────────────────────────────────────────────────────
    print("\nComputing F01 slope features...")
    df_feat = add_slope_features(df)

    f01_cols = ["slope_7d", "slope_14d", "slope_30d", "slope_all", "acceleration"]
    print("\nF01 — first 5 rows:")
    print(df_feat[["sensor_id", "timestamp", "voltage"] + f01_cols].head(5).to_string(index=False))

    v = _nan_check(df_feat, f01_cols, "F01")
    if v == 0:
        print("F01 NaN check PASSED.")
    else:
        print(f"F01 NaN check FAILED — {v} sensor(s).")
    print(f"F01 complete. New columns: {f01_cols}")

    # ── F02 ──────────────────────────────────────────────────────────────────
    print("\nComputing F02 rolling voltage statistics...")
    df_feat = add_rolling_features(df_feat)

    f02_cols = (
        [f"voltage_{stat}_{w}d" for w in (7, 14, 30) for stat in ("mean", "std", "min", "drop")]
        + ["voltage_current", "voltage_max_ever", "voltage_min_ever",
           "voltage_range_all", "voltage_pct"]
    )
    print("\nF02 — first 5 rows (rolling cols):")
    print(df_feat[["sensor_id", "timestamp", "voltage"] + f02_cols].head(5).to_string(index=False))

    v = _nan_check(df_feat, f02_cols, "F02")
    if v == 0:
        print("F02 NaN check PASSED.")
    else:
        print(f"F02 NaN check FAILED — {v} sensor(s).")
    print(f"F02 complete. New columns: {f02_cols}")

    # ── F03 ──────────────────────────────────────────────────────────────────
    print("\nComputing F03 temperature impact features...")
    df_feat = add_temperature_features(df_feat)

    f03_cols = [
        "temp_mean_all", "temp_mean_14d", "temp_min_ever", "temp_std_all",
        "temp_current", "days_below_0", "days_below_minus10",
        "cold_exposure_pct", "is_winter_now", "volt_temp_corr",
    ]
    print("\nF03 — first 5 rows (temp cols):")
    print(df_feat[["sensor_id", "timestamp", "temperature"] + f03_cols].head(5).to_string(index=False))

    v = _nan_check(df_feat, f03_cols, "F03")
    if v == 0:
        print("F03 NaN check PASSED.")
    else:
        print(f"F03 NaN check FAILED — {v} sensor(s).")
    print(f"F03 complete. New columns: {f03_cols}")

    # ── F04 ──────────────────────────────────────────────────────────────────
    print("\nComputing F04 lifecycle position features...")
    df_feat = add_lifecycle_features(df_feat)

    f04_cols = [
        "battery_age_days", "lifecycle_position", "readings_per_day",
        "days_since_last_reading", "voltage_at_30d", "voltage_at_60d",
        "voltage_at_90d", "month_of_year",
    ]
    print("\nF04 — first 5 rows (lifecycle cols):")
    print(df_feat[["sensor_id", "timestamp", "voltage"] + f04_cols].head(5).to_string(index=False))

    # voltage_at_30d/60d/90d are intentionally sparse (NaN until sensor is
    # old enough); only check the non-sparse F04 columns.
    f04_dense_cols = [
        "battery_age_days", "lifecycle_position", "readings_per_day",
        "days_since_last_reading", "month_of_year",
    ]
    v = _nan_check(df_feat, f04_dense_cols, "F04")
    if v == 0:
        print("F04 NaN check PASSED (sparse lookback cols excluded).")
    else:
        print(f"F04 NaN check FAILED — {v} sensor(s).")
    print(f"F04 complete. New columns: {f04_cols}")

    # ── Summary ───────────────────────────────────────────────────────────────
    all_feat_cols = f01_cols + f02_cols + f03_cols + f04_cols
    print(f"\nF04 complete. Total features so far: {len(all_feat_cols)}")
    print(f"Total rows in feature DataFrame: {len(df_feat):,}")
    print("\nAll feature column dtypes:")
    print(df_feat[all_feat_cols].dtypes.to_string())
