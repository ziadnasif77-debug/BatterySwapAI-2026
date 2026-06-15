"""
Inference module for BatterySwapAI 2026.

Loads a trained model and generates swap demand predictions for new
station observations, including point estimates and confidence intervals
derived from quantile regression or conformal prediction wrappers.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from model.features import build_feature_matrix


MODEL_PATH = Path("model/trained_model.pkl")
RESULTS_DIR = Path("results")


def load_model():
    """Load the serialized trained model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run model.train first."
        )
    return joblib.load(MODEL_PATH)


def predict(df: pd.DataFrame, confidence: float = 0.8) -> pd.DataFrame:
    """
    Generate demand predictions with confidence intervals.

    Args:
        df: Raw input DataFrame with station observations
        confidence: Desired confidence level (0.8 or 0.95)

    Returns:
        DataFrame with columns [station_id, timestamp, prediction, lower, upper]
    """
    model = load_model()
    X, _ = build_feature_matrix(df)
    point_preds = model.predict(X)

    # Approximate interval using +/- z * residual_std (replace with conformal for production)
    z = 1.28 if confidence == 0.8 else 1.96
    residual_std = getattr(model, "_residual_std", point_preds.std() * 0.3)

    result = df.loc[X.index, ["station_id", "timestamp"]].copy()
    result["prediction"] = np.maximum(point_preds, 0)
    result["lower"] = np.maximum(result["prediction"] - z * residual_std, 0)
    result["upper"] = result["prediction"] + z * residual_std
    return result


def save_predictions(df: pd.DataFrame, filename: str = "predictions.csv") -> None:
    """Write prediction DataFrame to results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / filename, index=False)
    print(f"Predictions saved to {RESULTS_DIR / filename}")


if __name__ == "__main__":
    sample = pd.read_csv("data/processed/features.csv", parse_dates=["timestamp"])
    preds = predict(sample)
    save_predictions(preds)
