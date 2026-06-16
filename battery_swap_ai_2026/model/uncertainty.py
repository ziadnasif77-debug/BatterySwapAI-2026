"""
model/uncertainty.py
Quantile regression intervals and failure probability estimates for RUL predictions.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.feature_pipeline import FEATURE_COLS

DATA_PATH      = Path(__file__).parent.parent / "data" / "raw"       / "sensor_readings.csv"
BUILDINGS_PATH = Path(__file__).parent.parent / "data" / "raw"       / "buildings.csv"
FEATURES_CSV   = Path(__file__).parent.parent / "data" / "processed" / "features_full.csv"
MODEL_PKL      = Path(__file__).parent.parent / "results" / "lightgbm_model.pkl"
RESULTS_DIR    = Path(__file__).parent.parent / "results"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_labeled(feat_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    death_dates = (
        raw_df[raw_df["end_of_life_date"].notna()]
        .groupby("sensor_id")["end_of_life_date"].first()
        .reset_index()
        .rename(columns={"end_of_life_date": "death_date"})
    )
    death_dates["death_date"] = pd.to_datetime(death_dates["death_date"])
    labeled = feat_df.merge(death_dates, on="sensor_id", how="inner")
    labeled["actual_rul"] = (labeled["death_date"] - labeled["timestamp"]).dt.days
    labeled = labeled[(labeled["actual_rul"] >= 0) & (labeled["actual_rul"] <= 365)].copy()
    return labeled.sort_values("timestamp").reset_index(drop=True)


# ── Public functions ──────────────────────────────────────────────────────────

def train_quantile_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple:
    """
    Train two HistGradientBoostingRegressor quantile models (handles NaN natively):
      lower_model  →  10th percentile (optimistic: battery lives longer)
      upper_model  →  90th percentile (pessimistic: battery dies sooner)

    Config: loss='quantile', max_iter=200, max_depth=4, random_state=42
    """
    print("  Training lower quantile model (α=0.10)...")
    lower_model = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.10,
        max_iter=200, max_depth=4, random_state=42,
    )
    lower_model.fit(X_train, y_train)

    print("  Training upper quantile model (α=0.90)...")
    upper_model = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.90,
        max_iter=200, max_depth=4, random_state=42,
    )
    upper_model.fit(X_train, y_train)

    return lower_model, upper_model


def predict_with_intervals(
    point_model,
    lower_model: HistGradientBoostingRegressor,
    upper_model: HistGradientBoostingRegressor,
    X: np.ndarray,
) -> pd.DataFrame:
    """
    Generate point predictions plus 90% prediction intervals.

    Returns DataFrame with:
        rul_predicted   point estimate (days)
        rul_lower_90    10th percentile (dies sooner scenario)
        rul_upper_90    90th percentile (lives longer scenario)
        interval_width  upper − lower (measure of uncertainty)
        confidence      HIGH (<3 days width) / MEDIUM (<7) / LOW (≥7)
    """
    point  = np.clip(point_model.predict(X), 0, None)
    lower  = np.clip(lower_model.predict(X), 0, None)
    upper  = np.clip(upper_model.predict(X), 0, None)

    # Ensure lower ≤ point ≤ upper after clipping
    lower  = np.minimum(lower, point)
    upper  = np.maximum(upper, point)
    width  = upper - lower

    confidence = np.where(width < 3, "HIGH",
                 np.where(width < 7, "MEDIUM", "LOW"))

    return pd.DataFrame({
        "rul_predicted":  np.round(point, 1),
        "rul_lower_90":   np.round(lower, 1),
        "rul_upper_90":   np.round(upper, 1),
        "interval_width": np.round(width, 1),
        "confidence":     confidence,
    })


def compute_failure_probabilities(
    rul_predicted: np.ndarray | float,
    interval_width: np.ndarray | float,
) -> dict:
    """
    Model RUL as a normal distribution:
        mean = rul_predicted
        std  = interval_width / 3.29
               (90% CI spans ±1.645 σ → total width = 3.29 σ)

    P(fail in X days) = CDF(X, mean=rul_predicted, std=std)
    """
    rul_predicted  = np.asarray(rul_predicted,  dtype=float)
    interval_width = np.asarray(interval_width, dtype=float)

    # Avoid division by zero; floor std at 0.5 day
    std = np.maximum(interval_width / 3.29, 0.5)

    return {
        "p_fail_3d":  np.round(norm.cdf(3,  loc=rul_predicted, scale=std) * 100, 1),
        "p_fail_7d":  np.round(norm.cdf(7,  loc=rul_predicted, scale=std) * 100, 1),
        "p_fail_14d": np.round(norm.cdf(14, loc=rul_predicted, scale=std) * 100, 1),
    }


def validate_coverage(predictions_df: pd.DataFrame, actuals: np.ndarray) -> float:
    """
    What % of actual RUL values fall within [rul_lower_90, rul_upper_90]?
    Target: 90%. <80% = intervals too narrow (dangerous). >97% = too wide.
    """
    within = (
        (actuals >= predictions_df["rul_lower_90"].values) &
        (actuals <= predictions_df["rul_upper_90"].values)
    )
    coverage = float(within.mean() * 100)

    if coverage < 80:
        verdict = "⚠ TOO NARROW — intervals are dangerously optimistic"
    elif coverage > 97:
        verdict = "⚠ TOO WIDE  — intervals are not useful"
    else:
        verdict = "✓ ACCEPTABLE"

    print(f"  Interval coverage: {coverage:.1f}%  (target: 90%)  {verdict}")
    return coverage


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    print("Loading model and data...")
    with open(MODEL_PKL, "rb") as f:
        saved = pickle.load(f)
    point_model = saved["model"]
    feat_names  = saved["feature_names"]

    feat_df   = pd.read_csv(FEATURES_CSV,   parse_dates=["timestamp"])
    raw_df    = pd.read_csv(DATA_PATH,       parse_dates=["timestamp"])
    buildings = pd.read_csv(BUILDINGS_PATH)[["building_id", "building_type"]]

    labeled = _load_labeled(feat_df, raw_df)
    labeled = labeled.merge(buildings, on="building_id", how="left")
    print(f"  Labeled rows: {len(labeled):,}")

    # ── Time split ────────────────────────────────────────────────────────
    split_idx = int(len(labeled) * 0.80)
    split_ts  = labeled["timestamp"].iloc[split_idx]
    gap_ts    = split_ts + pd.Timedelta(days=7)

    train_df = labeled[labeled["timestamp"] <= split_ts].copy()
    test_df  = labeled[labeled["timestamp"] > gap_ts].copy().reset_index(drop=True)

    X_train = train_df[feat_names].values
    y_train = train_df["actual_rul"].values
    X_test  = test_df[feat_names].values
    y_test  = test_df["actual_rul"].values

    # ── Train quantile models ─────────────────────────────────────────────
    print("\nTraining quantile models...")
    lower_model, upper_model = train_quantile_models(X_train, y_train)

    # ── Predictions with intervals ────────────────────────────────────────
    print("\nGenerating predictions with 90% intervals...")
    preds_df = predict_with_intervals(point_model, lower_model, upper_model, X_test)

    # ── Validate coverage ─────────────────────────────────────────────────
    print("\nValidating interval coverage:")
    coverage = validate_coverage(preds_df, y_test)

    # ── Failure probabilities ─────────────────────────────────────────────
    print("Computing failure probabilities...")
    fail_probs = compute_failure_probabilities(
        preds_df["rul_predicted"].values,
        preds_df["interval_width"].values,
    )

    # ── Assemble full predictions DataFrame ───────────────────────────────
    full_df = test_df[["sensor_id", "building_id", "building_type",
                        "timestamp", "actual_rul"]].copy()
    full_df = pd.concat([full_df.reset_index(drop=True), preds_df], axis=1)
    full_df["p_fail_3d"]  = fail_probs["p_fail_3d"]
    full_df["p_fail_7d"]  = fail_probs["p_fail_7d"]
    full_df["p_fail_14d"] = fail_probs["p_fail_14d"]

    # ── Summary ───────────────────────────────────────────────────────────
    avg_width  = preds_df["interval_width"].mean()
    conf_counts = preds_df["confidence"].value_counts()

    print("\n" + "=" * 52)
    print("  UNCERTAINTY QUANTIFICATION SUMMARY")
    print("=" * 52)
    print(f"  Test rows                : {len(full_df):,}")
    print(f"  Interval coverage        : {coverage:.1f}%  (target 90%)")
    print(f"  Avg interval width       : {avg_width:.1f} days")
    print(f"  Confidence distribution:")
    for level in ["HIGH", "MEDIUM", "LOW"]:
        n = conf_counts.get(level, 0)
        print(f"    {level:<8}: {n:>4}  ({n/len(full_df)*100:.1f}%)")
    print(f"  Avg P(fail in  3 days)   : {fail_probs['p_fail_3d'].mean():.1f}%")
    print(f"  Avg P(fail in  7 days)   : {fail_probs['p_fail_7d'].mean():.1f}%")
    print(f"  Avg P(fail in 14 days)   : {fail_probs['p_fail_14d'].mean():.1f}%")
    print("=" * 52)

    print("\nSample predictions (first 8 rows):")
    cols_show = ["sensor_id", "actual_rul", "rul_predicted",
                 "rul_lower_90", "rul_upper_90", "interval_width",
                 "confidence", "p_fail_7d"]
    print(full_df[cols_show].head(8).to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────
    out = RESULTS_DIR / "full_predictions.csv"
    full_df.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(full_df)} rows, {out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
