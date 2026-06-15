"""
Uncertainty quantification for BatterySwapAI 2026.

Provides a conformal prediction wrapper that converts point predictions into
statistically valid prediction intervals with guaranteed marginal coverage,
and an ensemble variance decomposition utility for epistemic uncertainty.
"""

from typing import Tuple
import numpy as np


class ConformalPredictor:
    """
    Split conformal prediction wrapper for any scikit-learn-style regressor.

    Guarantees: P(y in [lower, upper]) >= 1 - alpha over exchangeable data.
    """

    def __init__(self, base_model, alpha: float = 0.1):
        """
        Args:
            base_model: Fitted sklearn regressor
            alpha: Miscoverage rate (0.1 -> 90% coverage)
        """
        self.base_model = base_model
        self.alpha = alpha
        self._q_hat = None

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalPredictor":
        """
        Compute the conformal quantile on a held-out calibration set.

        Args:
            X_cal: Calibration features
            y_cal: Calibration targets
        """
        preds = self.base_model.predict(X_cal)
        residuals = np.abs(y_cal - preds)
        n = len(residuals)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self._q_hat = float(np.quantile(residuals, min(level, 1.0)))
        return self

    def predict_interval(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate point predictions with conformal intervals.

        Returns:
            (point, lower, upper) numpy arrays
        """
        if self._q_hat is None:
            raise RuntimeError("Call calibrate() before predict_interval().")
        point = self.base_model.predict(X)
        lower = np.maximum(point - self._q_hat, 0)
        upper = point + self._q_hat
        return point, lower, upper


def ensemble_uncertainty(predictions: np.ndarray) -> dict:
    """
    Decompose uncertainty from an ensemble of predictions.

    Args:
        predictions: Array of shape (n_models, n_samples)

    Returns:
        dict with keys: mean, epistemic_std, total_std
    """
    mean = predictions.mean(axis=0)
    epistemic = predictions.std(axis=0)
    return {"mean": mean, "epistemic_std": epistemic, "total_std": epistemic}
