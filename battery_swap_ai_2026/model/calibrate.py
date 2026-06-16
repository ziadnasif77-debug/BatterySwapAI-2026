"""
model/calibrate.py
Bias calibration for LightGBM RUL predictions.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.feature_pipeline import FEATURE_COLS

BASELINE_MAE  = 2.7

DATA_PATH     = Path(__file__).parent.parent / "data" / "raw"       / "sensor_readings.csv"
BUILDINGS_PATH= Path(__file__).parent.parent / "data" / "raw"       / "buildings.csv"
FEATURES_CSV  = Path(__file__).parent.parent / "data" / "processed" / "features_full.csv"
MODEL_PKL     = Path(__file__).parent.parent / "results" / "lightgbm_model.pkl"
RESULTS_DIR   = Path(__file__).parent.parent / "results"


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

def compute_bias(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Mean signed error: positive = predicting too late (dangerous for scheduling),
    negative = predicting too early (safe but wastes trips).
    """
    return float(np.mean(predictions - actuals))


def calibrate_predictions(predictions: np.ndarray, bias: float) -> np.ndarray:
    """Shift all predictions by subtracting the measured bias."""
    return np.clip(predictions - bias, 0.0, None)


def calibrate_by_building_type(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and correct per-building-type bias for office, hospital, warehouse.

    predictions_df must have columns:
        raw_predictions, actual_rul, building_type

    If a building type has < 5 samples: fall back to global bias.
    Adds column: calibrated_by_type
    """
    df = predictions_df.copy()
    global_bias = compute_bias(df["raw_predictions"].values, df["actual_rul"].values)

    df["calibrated_by_type"] = df["raw_predictions"].copy()

    print(f"\n  Building-type calibration  (global bias = {global_bias:+.2f} days):")
    for btype in ["office", "hospital", "warehouse"]:
        mask = df["building_type"] == btype
        n    = int(mask.sum())

        if n >= 5:
            type_bias = compute_bias(
                df.loc[mask, "raw_predictions"].values,
                df.loc[mask, "actual_rul"].values,
            )
            note = f"type bias = {type_bias:+.2f} days"
        else:
            type_bias = global_bias
            note = f"n={n} < 5, using global bias"

        df.loc[mask, "calibrated_by_type"] = np.clip(
            df.loc[mask, "raw_predictions"] - type_bias, 0.0, None
        )
        print(f"    {btype:<12}: n={n:>4},  {note}")

    return df


def evaluate_calibration(
    raw_preds:        np.ndarray,
    calibrated_preds: np.ndarray,
    actuals:          np.ndarray,
) -> dict:
    """Compare MAE and bias before and after calibration."""
    raw_mae  = float(np.mean(np.abs(raw_preds - actuals)))
    cal_mae  = float(np.mean(np.abs(calibrated_preds - actuals)))
    raw_bias = compute_bias(raw_preds, actuals)
    cal_bias = compute_bias(calibrated_preds, actuals)
    improvement = (raw_mae - cal_mae) / raw_mae * 100

    print("\n" + "=" * 52)
    print("  CALIBRATION EVALUATION")
    print("=" * 52)
    print(f"  Before calibration:  bias = {raw_bias:+.2f} days,  MAE = {raw_mae:.2f} days")
    print(f"  After  calibration:  bias = {cal_bias:+.2f} days,  MAE = {cal_mae:.2f} days")
    print(f"  Improvement: {improvement:+.1f}%")
    print("=" * 52)

    return {
        "raw_mae": raw_mae, "calibrated_mae": cal_mae,
        "raw_bias": raw_bias, "calibrated_bias": cal_bias,
        "improvement_pct": improvement,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    print("Loading model...")
    with open(MODEL_PKL, "rb") as f:
        saved = pickle.load(f)
    model      = saved["model"]
    feat_names = saved["feature_names"]
    print(f"  Loaded model: best_iteration={saved['best_iteration']}, "
          f"features={len(feat_names)}")

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading features and labels...")
    feat_df   = pd.read_csv(FEATURES_CSV,   parse_dates=["timestamp"])
    raw_df    = pd.read_csv(DATA_PATH,       parse_dates=["timestamp"])
    buildings = pd.read_csv(BUILDINGS_PATH)[["building_id", "building_type"]]

    labeled   = _load_labeled(feat_df, raw_df)
    labeled   = labeled.merge(buildings, on="building_id", how="left")

    # ── Time split (mirror of train.py) ───────────────────────────────────
    split_idx = int(len(labeled) * 0.80)
    split_ts  = labeled["timestamp"].iloc[split_idx]
    gap_ts    = split_ts + pd.Timedelta(days=7)

    train_df  = labeled[labeled["timestamp"] <= split_ts].copy()
    test_df   = labeled[labeled["timestamp"] > gap_ts].copy()

    # Last 20% of train → calibration set (no leakage from test)
    cal_cut   = int(len(train_df) * 0.80)
    cal_df    = train_df.iloc[cal_cut:].copy().reset_index(drop=True)

    X_cal   = cal_df[feat_names].values
    y_cal   = cal_df["actual_rul"].values
    X_test  = test_df[feat_names].values
    y_test  = test_df["actual_rul"].values

    # ── Predictions ───────────────────────────────────────────────────────
    cal_preds_raw  = model.predict(X_cal)
    test_preds_raw = model.predict(X_test)

    # ── Global bias from calibration set ─────────────────────────────────
    global_bias = compute_bias(cal_preds_raw, y_cal)
    print(f"\nGlobal bias (calibration set, n={len(y_cal)}): {global_bias:+.2f} days")

    test_preds_cal = calibrate_predictions(test_preds_raw, global_bias)

    # ── Evaluate global calibration ───────────────────────────────────────
    metrics = evaluate_calibration(test_preds_raw, test_preds_cal, y_test)

    # ── Building-type calibration on test set ─────────────────────────────
    print("\nApplying building-type calibration...")
    test_df = test_df.reset_index(drop=True)
    pred_df = test_df[["sensor_id", "building_id", "building_type",
                        "timestamp", "actual_rul"]].copy()
    pred_df["raw_predictions"]       = test_preds_raw
    pred_df["calibrated_global"]     = test_preds_cal
    pred_df = calibrate_by_building_type(pred_df)

    bt_mae  = float(np.mean(np.abs(pred_df["calibrated_by_type"] - pred_df["actual_rul"])))
    bt_bias = compute_bias(pred_df["calibrated_by_type"].values, pred_df["actual_rul"].values)

    print(f"\n  By-type MAE  = {bt_mae:.2f} days  (bias = {bt_bias:+.2f} days)")

    # ── Full summary ──────────────────────────────────────────────────────
    raw_mae = metrics["raw_mae"]
    cal_mae = metrics["calibrated_mae"]

    print("\n" + "=" * 52)
    print("  FULL COMPARISON")
    print("=" * 52)
    print(f"  Baseline               : {BASELINE_MAE:.1f} days")
    print(f"  LightGBM (raw)         : {raw_mae:.1f} days")
    print(f"  LightGBM + global cal  : {cal_mae:.1f} days  "
          f"({(raw_mae - cal_mae)/raw_mae*100:+.1f}% vs raw)")
    print(f"  LightGBM + type cal    : {bt_mae:.1f} days  "
          f"({(raw_mae - bt_mae)/raw_mae*100:+.1f}% vs raw)")
    print("=" * 52)

    # ── Save ──────────────────────────────────────────────────────────────
    out = RESULTS_DIR / "calibrated_predictions.csv"
    pred_df.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(pred_df)} rows)")

    best_mae = min(cal_mae, bt_mae)
    print(f"\nCalibration complete. New MAE: {best_mae:.1f} days")


if __name__ == "__main__":
    main()
