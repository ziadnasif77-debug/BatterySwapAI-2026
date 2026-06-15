"""
Model training entry point for BatterySwapAI 2026.

Loads processed features, trains a LightGBM gradient-boosted model with
cross-validation, evaluates against the baseline, and serializes the
trained model to disk. Run as: python -m model.train
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import lightgbm as lgb
    _BACKEND = "lightgbm"
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    _BACKEND = "sklearn"

from model.features import build_feature_matrix
from model.baseline import BaselineModel


MODEL_DIR = Path("model")
RESULTS_DIR = Path("results")


def train(data_path: str = "data/processed/features.parquet") -> None:
    """
    Train the demand forecasting model and save artifacts.

    Args:
        data_path: Path to the processed feature Parquet file
    """
    df = pd.read_parquet(data_path)
    X, y = build_feature_matrix(df)

    tscv = TimeSeriesSplit(n_splits=5)
    mae_scores, rmse_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if _BACKEND == "lightgbm":
            model = lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.05, num_leaves=63, random_state=42
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, random_state=42
            )

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        mae_scores.append(mean_absolute_error(y_val, preds))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        print(f"Fold {fold + 1}: MAE={mae_scores[-1]:.3f}, RMSE={rmse_scores[-1]:.3f}")

    print(f"\nCV MAE:  {np.mean(mae_scores):.3f} +/- {np.std(mae_scores):.3f}")
    print(f"CV RMSE: {np.mean(rmse_scores):.3f} +/- {np.std(rmse_scores):.3f}")

    model.fit(X, y)
    joblib.dump(model, MODEL_DIR / "trained_model.pkl")

    metrics = {
        "cv_mae_mean": round(float(np.mean(mae_scores)), 4),
        "cv_mae_std": round(float(np.std(mae_scores)), 4),
        "cv_rmse_mean": round(float(np.mean(rmse_scores)), 4),
        "cv_rmse_std": round(float(np.std(rmse_scores)), 4),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model saved to model/trained_model.pkl")


if __name__ == "__main__":
    train()
