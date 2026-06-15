"""
Baseline prediction model for battery swap demand forecasting.

Implements a simple historical-average baseline that predicts demand
at each station based on the mean swap count for the same hour-of-day
and day-of-week from the training window. Used as a reference point
for evaluating more complex models.
"""

import numpy as np
import pandas as pd


class BaselineModel:
    """Predicts swap demand using hour-of-week historical averages."""

    def __init__(self):
        self._lookup = {}

    def fit(self, df: pd.DataFrame, target_col: str = "swap_count") -> "BaselineModel":
        """
        Compute mean demand per (station_id, day_of_week, hour_of_day).

        Args:
            df: DataFrame with columns [station_id, timestamp, swap_count]
            target_col: Column to average
        """
        df = df.copy()
        df["dow"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
        df["hod"] = pd.to_datetime(df["timestamp"]).dt.hour
        self._lookup = (
            df.groupby(["station_id", "dow", "hod"])[target_col]
            .mean()
            .to_dict()
        )
        self._global_mean = df[target_col].mean()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return predicted swap counts for each row in df.

        Args:
            df: DataFrame with columns [station_id, timestamp]

        Returns:
            numpy array of predictions
        """
        df = df.copy()
        df["dow"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
        df["hod"] = pd.to_datetime(df["timestamp"]).dt.hour

        preds = []
        for _, row in df.iterrows():
            key = (row["station_id"], row["dow"], row["hod"])
            preds.append(self._lookup.get(key, self._global_mean))
        return np.array(preds)
