"""
Feature engineering pipeline for BatterySwapAI 2026.

Transforms raw station telemetry, weather data, and calendar features
into a flat feature matrix suitable for tree-based and linear models.
Reads from data/raw/ and writes processed Parquet files to data/processed/.
"""

from pathlib import Path
import pandas as pd
import numpy as np


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_raw(filename: str) -> pd.DataFrame:
    """Load a CSV from data/raw/."""
    return pd.read_csv(RAW_DIR / filename, parse_dates=["timestamp"])


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append hour-of-day, day-of-week, month, and is_weekend columns."""
    ts = pd.to_datetime(df["timestamp"])
    df = df.copy()
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def add_lag_features(df: pd.DataFrame, col: str, lags: list) -> pd.DataFrame:
    """Add lag columns for the given target column within each station."""
    df = df.sort_values(["station_id", "timestamp"]).copy()
    for lag in lags:
        df[f"{col}_lag_{lag}"] = df.groupby("station_id")[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str, windows: list) -> pd.DataFrame:
    """Add rolling mean and std features within each station group."""
    df = df.sort_values(["station_id", "timestamp"]).copy()
    for w in windows:
        grp = df.groupby("station_id")[col]
        df[f"{col}_roll_mean_{w}"] = grp.transform(lambda x: x.shift(1).rolling(w).mean())
        df[f"{col}_roll_std_{w}"] = grp.transform(lambda x: x.shift(1).rolling(w).std())
    return df


def build_feature_matrix(df: pd.DataFrame, target_col: str = "swap_count"):
    """
    Run the full feature engineering pipeline.

    Returns:
        X: feature DataFrame
        y: target Series
    """
    df = add_calendar_features(df)
    df = add_lag_features(df, target_col, lags=[1, 2, 3, 6, 12, 24])
    df = add_rolling_features(df, target_col, windows=[3, 6, 12, 24])
    df = df.dropna()
    feature_cols = [c for c in df.columns if c not in ["timestamp", "station_id", target_col]]
    return df[feature_cols], df[target_col]


def save_processed(df: pd.DataFrame, filename: str) -> None:
    """Save DataFrame as Parquet in data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / filename, index=False)
