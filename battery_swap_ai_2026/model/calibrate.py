"""
Probability calibration for BatterySwapAI 2026.

Wraps a trained point-prediction model with isotonic regression to calibrate
predicted swap counts, reducing systematic over/under-prediction. Calibration
quality is measured using Expected Calibration Error (ECE).
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression


class DemandCalibrator:
    """Calibrates raw model predictions to reduce systematic bias."""

    def __init__(self, method: str = "isotonic"):
        if method not in ("isotonic",):
            raise ValueError(f"Unsupported calibration method: {method}")
        self.method = method
        self._calibrator = IsotonicRegression(out_of_bounds="clip")

    def fit(self, raw_predictions: np.ndarray, actuals: np.ndarray) -> "DemandCalibrator":
        """
        Fit the calibration mapping from raw predictions to actual values.

        Args:
            raw_predictions: Uncalibrated model outputs
            actuals: Ground-truth observed swap counts
        """
        self._calibrator.fit(raw_predictions, actuals)
        return self

    def transform(self, raw_predictions: np.ndarray) -> np.ndarray:
        """Apply calibration to new predictions."""
        return self._calibrator.transform(raw_predictions)

    def fit_transform(self, raw_predictions: np.ndarray, actuals: np.ndarray) -> np.ndarray:
        """Fit and immediately apply calibration."""
        return self.fit(raw_predictions, actuals).transform(raw_predictions)

    def expected_calibration_error(
        self, predictions: np.ndarray, actuals: np.ndarray, n_bins: int = 10
    ) -> float:
        """
        Compute ECE over n_bins equal-width buckets.

        Lower ECE means better-calibrated predictions.
        """
        bins = np.linspace(predictions.min(), predictions.max(), n_bins + 1)
        ece = 0.0
        n = len(predictions)
        for i in range(n_bins):
            mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = actuals[mask].mean()
            bin_conf = predictions[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
        return float(ece)
