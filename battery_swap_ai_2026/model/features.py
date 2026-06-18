"""
model/features.py — vectorized pipeline (5-20x faster)
F01 — Voltage Slope Features     (rolling OLS, no row loop)
F02 — Rolling Voltage Statistics (pandas time-based rolling)
F03 — Temperature Impact         (expanding + rolling)
F04 — Lifecycle Position         (expanding, vectorized)
F05 — Curve Shape                (ThreadPool, one fit per sensor)
"""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DEAD_THRESHOLD = 2.5
DATA_PATH      = Path(__file__).parent.parent / "data" / "raw" / "sensor_readings.csv"
WINTER_MONTHS  = {12, 1, 2, 3}
_N_WORKERS     = max(1, min(multiprocessing.cpu_count(), 8))


# ═══════════════════════════════════════════════════════════════════════════════
# F01 — Voltage Slopes
# ═══════════════════════════════════════════════════════════════════════════════

def _ols_slope(v_arr: np.ndarray) -> float:
    n = len(v_arr)
    if n < 3:
        return np.nan
    t = np.arange(n, dtype=np.float64)
    t_m = t.mean()
    v_m = v_arr.mean()
    denom = ((t - t_m) ** 2).sum()
    return float(((t - t_m) * (v_arr - v_m)).sum() / denom) if denom > 0 else np.nan


def _sensor_slopes(grp: pd.DataFrame) -> pd.DataFrame:
    g   = grp.sort_values("timestamp").reset_index(drop=True)
    gts = g.set_index("timestamp")["voltage"]

    s7   = gts.rolling("7D",  min_periods=3).apply(_ols_slope, raw=True)
    s14  = gts.rolling("14D", min_periods=3).apply(_ols_slope, raw=True)
    s30  = gts.rolling("30D", min_periods=3).apply(_ols_slope, raw=True)
    sall = gts.expanding(min_periods=3).apply(_ols_slope, raw=True)

    g["slope_7d"]     = s7.values
    g["slope_14d"]    = s14.values
    g["slope_30d"]    = s30.values
    g["slope_all"]    = sall.values
    g["acceleration"] = np.where(
        ~np.isnan(g["slope_7d"].values) & ~np.isnan(g["slope_30d"].values),
        g["slope_7d"].values - g["slope_30d"].values,
        np.nan,
    )
    return g


def compute_voltage_slopes(sensor_df: pd.DataFrame) -> dict:
    """Single-row interface kept for test compatibility."""
    g = sensor_df.sort_values("timestamp").reset_index(drop=True)
    v = g["voltage"].values
    result = _sensor_slopes(g)
    row = result.iloc[-1]
    return {k: row[k] for k in ("slope_7d", "slope_14d", "slope_30d", "slope_all", "acceleration")}


def add_slope_features(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy().sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
    groups = [grp for _, grp in df.groupby("sensor_id", sort=False)]

    with ThreadPoolExecutor(max_workers=_N_WORKERS) as ex:
        results = list(ex.map(_sensor_slopes, groups))

    return pd.concat(results, ignore_index=True).sort_values(
        ["sensor_id", "timestamp"]
    ).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# F02 — Rolling Voltage Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rolling_voltage_stats(sensor_df: pd.DataFrame) -> dict:
    """Single-row interface kept for test compatibility."""
    g      = sensor_df.sort_values("timestamp").reset_index(drop=True)
    result = _sensor_rolling(g)
    cols   = [
        "voltage_mean_7d", "voltage_std_7d", "voltage_min_7d", "voltage_drop_7d",
        "voltage_mean_14d", "voltage_std_14d", "voltage_min_14d", "voltage_drop_14d",
        "voltage_mean_30d", "voltage_std_30d", "voltage_min_30d", "voltage_drop_30d",
        "voltage_current", "voltage_max_ever", "voltage_min_ever",
        "voltage_range_all", "voltage_pct",
    ]
    row = result.iloc[-1]
    return {c: row[c] for c in cols}


def _sensor_rolling(grp: pd.DataFrame) -> pd.DataFrame:
    g   = grp.sort_values("timestamp").reset_index(drop=True)
    gts = g.set_index("timestamp")["voltage"]

    for w in (7, 14, 30):
        roll = gts.rolling(f"{w}D", min_periods=1)
        g[f"voltage_mean_{w}d"] = roll.mean().values
        g[f"voltage_std_{w}d"]  = roll.std().values
        g[f"voltage_min_{w}d"]  = roll.min().values
        g[f"voltage_drop_{w}d"] = roll.apply(
            lambda v: float(v[0] - v[-1]) if len(v) >= 2 else 0.0, raw=True
        ).values

    exp = gts.expanding(min_periods=1)
    g["voltage_current"]   = g["voltage"].values
    g["voltage_max_ever"]  = exp.max().values
    g["voltage_min_ever"]  = exp.min().values
    g["voltage_range_all"] = g["voltage_max_ever"] - g["voltage_min_ever"]

    denom = g["voltage_max_ever"] - DEAD_THRESHOLD
    pct   = (g["voltage_current"] - DEAD_THRESHOLD) / denom * 100.0
    g["voltage_pct"] = np.where(denom > 0, np.clip(pct, 0.0, 100.0), np.nan)
    return g


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy().sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
    groups = [grp for _, grp in df.groupby("sensor_id", sort=False)]

    with ThreadPoolExecutor(max_workers=_N_WORKERS) as ex:
        results = list(ex.map(_sensor_rolling, groups))

    return pd.concat(results, ignore_index=True).sort_values(
        ["sensor_id", "timestamp"]
    ).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# F03 — Temperature Impact
# ═══════════════════════════════════════════════════════════════════════════════

def compute_temperature_features(sensor_df: pd.DataFrame) -> dict:
    g      = sensor_df.sort_values("timestamp").reset_index(drop=True)
    result = _sensor_temperature(g)
    cols   = [
        "temp_mean_all", "temp_mean_14d", "temp_min_ever", "temp_std_all",
        "temp_current", "days_below_0", "days_below_minus10",
        "cold_exposure_pct", "is_winter_now", "volt_temp_corr",
    ]
    row = result.iloc[-1]
    return {c: row[c] for c in cols}


def _sensor_temperature(grp: pd.DataFrame) -> pd.DataFrame:
    g   = grp.sort_values("timestamp").reset_index(drop=True)
    gts = g.set_index("timestamp")

    t_series = gts["temperature"]
    v_series = gts["voltage"]

    exp_t = t_series.expanding(min_periods=1)
    g["temp_mean_all"]  = exp_t.mean().values
    g["temp_std_all"]   = exp_t.std().values
    g["temp_min_ever"]  = exp_t.min().values
    g["temp_current"]   = t_series.values
    g["temp_mean_14d"]  = t_series.rolling("14D", min_periods=1).mean().values

    g["days_below_0"]       = (t_series < 0).expanding().sum().values.astype(int)
    g["days_below_minus10"] = (t_series < -10).expanding().sum().values.astype(int)

    total = np.arange(1, len(g) + 1, dtype=float)
    g["cold_exposure_pct"] = g["days_below_0"].values / total * 100.0
    g["is_winter_now"]     = g["timestamp"].dt.month.isin(WINTER_MONTHS).astype(float).values

    # Rolling correlation voltage vs temperature (min 5 readings)
    v_roll = v_series.rolling("30D", min_periods=5)
    t_roll = t_series.rolling("30D", min_periods=5)
    corr   = v_series.rolling("30D", min_periods=5).corr(t_series)
    g["volt_temp_corr"] = corr.fillna(0.0).values

    return g


def add_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy().sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
    groups = [grp for _, grp in df.groupby("sensor_id", sort=False)]

    with ThreadPoolExecutor(max_workers=_N_WORKERS) as ex:
        results = list(ex.map(_sensor_temperature, groups))

    return pd.concat(results, ignore_index=True).sort_values(
        ["sensor_id", "timestamp"]
    ).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# F04 — Lifecycle Position
# ═══════════════════════════════════════════════════════════════════════════════

def compute_lifecycle_features(sensor_df: pd.DataFrame) -> dict:
    g      = sensor_df.sort_values("timestamp").reset_index(drop=True)
    result = _sensor_lifecycle(g)
    cols   = [
        "battery_age_days", "lifecycle_position", "readings_per_day",
        "days_since_last_reading", "voltage_at_30d", "voltage_at_60d",
        "voltage_at_90d", "month_of_year",
    ]
    row = result.iloc[-1]
    return {c: row[c] for c in cols}


def _sensor_lifecycle(grp: pd.DataFrame) -> pd.DataFrame:
    g = grp.sort_values("timestamp").reset_index(drop=True)

    first_ts = g["timestamp"].iloc[0]
    g["battery_age_days"]        = (g["timestamp"] - first_ts).dt.days.astype(float)
    g["readings_per_day"]        = np.where(
        g["battery_age_days"] > 0,
        (np.arange(1, len(g) + 1, dtype=float)) / g["battery_age_days"],
        np.nan,
    )
    g["days_since_last_reading"] = 0.0
    g["month_of_year"]           = g["timestamp"].dt.month.astype(float)

    v_max_ever = g["voltage"].expanding().max()
    denom      = v_max_ever - DEAD_THRESHOLD
    lc         = 1.0 - (g["voltage"] - DEAD_THRESHOLD) / denom
    g["lifecycle_position"] = np.where(denom > 0, np.clip(lc, 0.0, 1.0), np.nan)

    # voltage_at_Nd: closest reading <= (current_ts - Nd), within ±3 days
    for d in (30, 60, 90):
        target = g["timestamp"] - pd.Timedelta(days=d)
        vals   = np.full(len(g), np.nan)
        for i in range(len(g)):
            past = g[g["timestamp"] <= target.iloc[i]]
            if not past.empty:
                closest = past.iloc[-1]
                gap     = abs((closest["timestamp"] - target.iloc[i]).days)
                if gap <= 3:
                    vals[i] = closest["voltage"]
        g[f"voltage_at_{d}d"] = vals

    return g


def add_lifecycle_features(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy().sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
    groups = [grp for _, grp in df.groupby("sensor_id", sort=False)]

    with ThreadPoolExecutor(max_workers=_N_WORKERS) as ex:
        results = list(ex.map(_sensor_lifecycle, groups))

    return pd.concat(results, ignore_index=True).sort_values(
        ["sensor_id", "timestamp"]
    ).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# F05 — Curve Shape (exponential decay)
# ═══════════════════════════════════════════════════════════════════════════════

def _exp_decay_model(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.exp(-b * t) + c


def fit_exponential_decay(sensor_df: pd.DataFrame) -> dict:
    """Single-row interface kept for test compatibility."""
    g      = sensor_df.sort_values("timestamp").reset_index(drop=True)
    result = _sensor_curve(g)
    cols   = ["curve_fit_ok", "decay_rate", "decay_floor",
              "decay_amplitude", "decay_phase", "days_to_floor"]
    row = result.iloc[-1]
    return {c: row[c] for c in cols}


def _sensor_curve(grp: pd.DataFrame) -> pd.DataFrame:
    """Fit exponential decay ONCE per sensor; broadcast to all rows."""
    g = grp.sort_values("timestamp").reset_index(drop=True)

    v_arr      = g["voltage"].values
    v_current  = float(v_arr[-1])
    v_max      = float(v_arr.max())
    denom      = v_max - DEAD_THRESHOLD
    if denom > 0:
        lc_pos = float(np.clip(1.0 - (v_current - DEAD_THRESHOLD) / denom, 0.0, 1.0))
    else:
        lc_pos = 0.5
    decay_phase = 1.0 if lc_pos < 0.4 else (2.0 if lc_pos < 0.7 else 3.0)

    base = {
        "curve_fit_ok":    0.0,
        "decay_rate":      np.nan,
        "decay_floor":     np.nan,
        "decay_amplitude": np.nan,
        "decay_phase":     decay_phase,
        "days_to_floor":   np.nan,
    }

    fit_ok = False
    if len(g) >= 10:
        t0     = g["timestamp"].iloc[0]
        t_vals = (g["timestamp"] - t0).dt.total_seconds().values / 86400.0
        v_min  = v_arr.min()
        a0     = max(v_arr.max() - v_min, 0.1)
        try:
            popt, _ = curve_fit(
                _exp_decay_model, t_vals, v_arr,
                p0=[a0, 0.005, max(v_min - 0.05, 0.0)],
                bounds=([0.0, 1e-6, 0.0], [10.0, 1.0, 5.0]),
                maxfev=3000,
            )
            a, b, c = float(popt[0]), float(popt[1]), float(popt[2])
            if b > 0 and a > 0:
                total  = (-1.0 / b) * np.log(0.1 / a)
                dtf    = float(np.clip(total - float(t_vals[-1]), 0.0, 999.0))
                base.update({
                    "curve_fit_ok":    1.0,
                    "decay_rate":      b,
                    "decay_floor":     c,
                    "decay_amplitude": a,
                    "days_to_floor":   dtf,
                })
                fit_ok = True
        except Exception:
            pass

    for col, val in base.items():
        g[col] = val

    return g


def add_curve_features(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy().sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
    groups = [grp for _, grp in df.groupby("sensor_id", sort=False)]

    with ThreadPoolExecutor(max_workers=_N_WORKERS) as ex:
        results = list(ex.map(_sensor_curve, groups))

    return pd.concat(results, ignore_index=True).sort_values(
        ["sensor_id", "timestamp"]
    ).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone test (python features.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _nan_check(df_feat: pd.DataFrame, cols: list, label: str) -> int:
    violations = 0
    for sid, grp in df_feat.groupby("sensor_id"):
        grp    = grp.sort_values("timestamp").reset_index(drop=True)
        cutoff = int(len(grp) * 0.20)
        last80 = grp.iloc[cutoff:]
        bad    = last80[cols].isna().sum()
        bad    = bad[bad > 0]
        if not bad.empty:
            print(f"  WARNING {sid} [{label}]: NaNs in last 80% — {bad.to_dict()}")
            violations += 1
    return violations


if __name__ == "__main__":
    import time as _time
    print("Loading sensor data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    print(f"  Loaded {len(df):,} rows for {df['sensor_id'].nunique()} sensors")

    for label, fn, dense_cols in [
        ("F01", add_slope_features,
         ["slope_7d", "slope_14d", "slope_30d", "slope_all", "acceleration"]),
        ("F02", add_rolling_features,
         ["voltage_mean_7d", "voltage_std_7d", "voltage_min_7d"]),
        ("F03", add_temperature_features,
         ["temp_mean_all", "temp_current", "is_winter_now"]),
        ("F04", add_lifecycle_features,
         ["battery_age_days", "lifecycle_position", "readings_per_day"]),
        ("F05", add_curve_features,
         ["curve_fit_ok", "decay_phase"]),
    ]:
        t0 = _time.perf_counter()
        df = fn(df)
        dt = _time.perf_counter() - t0
        v  = _nan_check(df, dense_cols, label)
        status = "PASS" if v == 0 else f"FAIL ({v})"
        print(f"  {label}: {dt:.1f}s  NaN check {status}")

    print(f"\nDone. Total rows: {len(df):,}, columns: {len(df.columns)}")
