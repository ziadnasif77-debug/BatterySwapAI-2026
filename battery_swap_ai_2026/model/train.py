"""
model/train.py
Trains LightGBM RUL prediction model with early stopping and time-series CV.
"""

import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.feature_pipeline import FEATURE_COLS

BASELINE_MAE = 2.7   # days — from model/baseline.py

DATA_PATH    = Path(__file__).parent.parent / "data" / "raw"       / "sensor_readings.csv"
FEATURES_CSV = Path(__file__).parent.parent / "data" / "processed" / "features_full.csv"
RESULTS_DIR  = Path(__file__).parent.parent / "results"
MODEL_OUT    = RESULTS_DIR / "lightgbm_model.pkl"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_labeled(feat_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Attach actual_rul labels to feature rows (dead sensors only)."""
    death_dates = (
        raw_df[raw_df["end_of_life_date"].notna()]
        .groupby("sensor_id")["end_of_life_date"]
        .first()
        .reset_index()
        .rename(columns={"end_of_life_date": "death_date"})
    )
    death_dates["death_date"] = pd.to_datetime(death_dates["death_date"])

    labeled = feat_df.merge(death_dates, on="sensor_id", how="inner")
    labeled["actual_rul"] = (labeled["death_date"] - labeled["timestamp"]).dt.days
    labeled = labeled[(labeled["actual_rul"] >= 0) & (labeled["actual_rul"] <= 365)].copy()
    return labeled.sort_values("timestamp").reset_index(drop=True)


def _time_split(labeled: pd.DataFrame, feat_names: list):
    """80/20 time-based split with 7-day gap. Returns train/test arrays + DataFrames."""
    split_idx = int(len(labeled) * 0.80)
    split_ts  = labeled["timestamp"].iloc[split_idx]
    gap_ts    = split_ts + pd.Timedelta(days=7)

    train_df = labeled[labeled["timestamp"] <= split_ts].copy()
    test_df  = labeled[labeled["timestamp"] > gap_ts].copy()

    X_train = train_df[feat_names].values
    X_test  = test_df[feat_names].values
    y_train = train_df["actual_rul"].values
    y_test  = test_df["actual_rul"].values

    return X_train, X_test, y_train, y_test, train_df, test_df


# ── Public functions ──────────────────────────────────────────────────────────

def train_lightgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
) -> lgb.Booster:
    """
    Trains LightGBM with early stopping on a validation set.

    Config: regression / MAE / 1000 trees max / lr=0.05 / early_stop=50
    Returns the best booster.
    """
    feat_names = [f"f{i}" for i in range(X_train.shape[1])]

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feat_names)
    val_data   = lgb.Dataset(X_val,   label=y_val,   feature_name=feat_names,
                             reference=train_data)

    params = {
        "objective":        "regression",
        "metric":           "mae",
        "num_leaves":       31,
        "learning_rate":    0.05,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "random_state":     42,
        "verbose":          -1,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=-1),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
    )
    print(f"  Best iteration: {model.best_iteration}  "
          f"(val MAE = {model.best_score['valid_0']['l1']:.2f} days)")
    return model


def evaluate_model(
    model:    lgb.Booster,
    X_test:   np.ndarray,
    y_test:   np.ndarray,
    feat_names: list | None = None,
) -> dict:
    """
    Evaluates the model on the test set and prints a comparison with baseline.
    """
    preds = model.predict(X_test)

    mae  = float(np.mean(np.abs(preds - y_test)))
    rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
    bias = float(np.mean(preds - y_test))
    r2   = float(r2_score(y_test, preds))
    pct3 = float(np.mean(np.abs(preds - y_test) <= 3) * 100)
    pct7 = float(np.mean(np.abs(preds - y_test) <= 7) * 100)

    improvement = (BASELINE_MAE - mae) / BASELINE_MAE * 100

    print("\n" + "=" * 48)
    print("  MODEL EVALUATION")
    print("=" * 48)
    print(f"  MAE  : {mae:.1f} days  "
          f"(Baseline was: {BASELINE_MAE} days — improvement: {improvement:+.1f}%)")
    print(f"  RMSE : {rmse:.1f} days")
    print(f"  Bias : {bias:+.1f} days  "
          f"({'predicting too late' if bias > 0 else 'predicting too early'})")
    print(f"  R²   : {r2:.3f}")
    print(f"  Within 3 days : {pct3:.1f}%")
    print(f"  Within 7 days : {pct7:.1f}%")
    print("=" * 48)

    return {
        "mae": mae, "rmse": rmse, "bias": bias, "r2": r2,
        "pct_within_3d": pct3, "pct_within_7d": pct7,
    }


def cross_validate_time_series(
    df_features:  pd.DataFrame,
    feature_cols: list,
    target_col:   str,
) -> list:
    """
    Rolling expanding-window cross-validation (3 windows).

    Window 1: train months 1-6  → test month 7
    Window 2: train months 1-7  → test month 8
    Window 3: train months 1-8  → test month 9

    Months are counted from the first calendar month in the dataset.
    Returns list of MAE scores per window.
    """
    df = df_features.dropna(subset=feature_cols + [target_col]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Anchor: first day of the first month present in the data
    ref_start = df["timestamp"].min().replace(day=1)

    mae_scores = []
    print("\nTime-Series Cross-Validation:")
    print(f"  Reference start : {ref_start.date()}")

    for k in range(3):
        n_train_months = 6 + k
        train_end  = ref_start + pd.DateOffset(months=n_train_months) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end   = ref_start + pd.DateOffset(months=n_train_months + 1) - pd.Timedelta(days=1)

        train_w = df[df["timestamp"] <= train_end]
        test_w  = df[(df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)]

        if len(train_w) < 20 or len(test_w) < 5:
            print(f"  Window {k+1}: skipped "
                  f"(train={len(train_w)}, test={len(test_w)} — too few samples)")
            continue

        X_tr = train_w[feature_cols].values
        y_tr = train_w[target_col].values
        X_te = test_w[feature_cols].values
        y_te = test_w[target_col].values

        # Use last 15% of window's train as internal validation for early stopping
        val_cut = int(len(X_tr) * 0.85)
        X_tr_inner, X_val_inner = X_tr[:val_cut], X_tr[val_cut:]
        y_tr_inner, y_val_inner = y_tr[:val_cut], y_tr[val_cut:]

        model_w = train_lightgbm_model(X_tr_inner, y_tr_inner,
                                       X_val_inner, y_val_inner)
        preds_w = model_w.predict(X_te)
        mae_w   = float(np.mean(np.abs(preds_w - y_te)))
        mae_scores.append(mae_w)

        print(f"  Window {k+1} "
              f"(train ≤ {train_end.date()}, test {test_start.date()}–{test_end.date()}): "
              f"MAE = {mae_w:.2f} days  "
              f"(n_train={len(train_w)}, n_test={len(test_w)})")

    if mae_scores:
        print(f"  Average CV MAE: {np.mean(mae_scores):.2f} days "
              f"(±{np.std(mae_scores):.2f})")
    return mae_scores


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load pre-built features + raw data for labels ─────────────────────
    print("Loading features_full.csv...")
    feat_df = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"])
    print(f"  {len(feat_df):,} rows, {feat_df['sensor_id'].nunique()} sensors")

    print("Loading raw sensor data for RUL labels...")
    raw_df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    # ── Build labeled dataset ──────────────────────────────────────────────
    print("Attaching actual_rul labels...")
    labeled = _load_labeled(feat_df, raw_df)
    feat_names = [c for c in FEATURE_COLS if c in labeled.columns]
    print(f"  Labeled rows : {len(labeled):,}  |  Features : {len(feat_names)}")

    # ── Time-based train/test split ────────────────────────────────────────
    X_train, X_test, y_train, y_test, train_df, test_df = _time_split(labeled, feat_names)
    print(f"\nTraining samples : {len(y_train):,}")
    print(f"Test samples     : {len(y_test):,}")
    print(f"Train period     : {train_df['timestamp'].min().date()} → "
          f"{train_df['timestamp'].max().date()}")
    print(f"Test period      : {test_df['timestamp'].min().date()} → "
          f"{test_df['timestamp'].max().date()}")

    # ── Hold out last 15% of training for early stopping ──────────────────
    val_cut     = int(len(X_train) * 0.85)
    X_tr_inner  = X_train[:val_cut]
    y_tr_inner  = y_train[:val_cut]
    X_val       = X_train[val_cut:]
    y_val       = y_train[val_cut:]

    # ── Train ──────────────────────────────────────────────────────────────
    print("\nTraining LightGBM model...")
    model = train_lightgbm_model(X_tr_inner, y_tr_inner, X_val, y_val)

    # ── Retrain on full train set with best iteration ──────────────────────
    print(f"\nRetraining on full training set ({len(y_train):,} rows) "
          f"for {model.best_iteration} iterations...")
    full_train_data = lgb.Dataset(X_train, label=y_train,
                                  feature_name=[f"f{i}" for i in range(X_train.shape[1])])
    params_final = {
        "objective": "regression", "metric": "mae",
        "num_leaves": 31, "learning_rate": 0.05,
        "min_child_samples": 20, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5,
        "random_state": 42, "verbose": -1,
    }
    final_model = lgb.train(params_final, full_train_data,
                            num_boost_round=model.best_iteration)

    # ── Evaluate ───────────────────────────────────────────────────────────
    results = evaluate_model(final_model, X_test, y_test, feat_names)

    # ── Cross-validation ───────────────────────────────────────────────────
    cv_scores = cross_validate_time_series(labeled, feat_names, "actual_rul")

    # ── Save model ─────────────────────────────────────────────────────────
    with open(MODEL_OUT, "wb") as f:
        pickle.dump({"model": final_model, "feature_names": feat_names,
                     "best_iteration": model.best_iteration}, f)
    print(f"\nModel saved: {MODEL_OUT}  ({MODEL_OUT.stat().st_size // 1024} KB)")

    # ── Final comparison ───────────────────────────────────────────────────
    lgbm_mae = results["mae"]
    improvement = (BASELINE_MAE - lgbm_mae) / BASELINE_MAE * 100

    print("\n" + "=" * 48)
    print("  FINAL COMPARISON")
    print("=" * 48)
    print(f"  Baseline MAE  : {BASELINE_MAE:.1f} days")
    print(f"  LightGBM MAE  : {lgbm_mae:.1f} days")
    print(f"  Improvement   : {improvement:+.1f}%")
    if cv_scores:
        print(f"  CV MAE (avg)  : {np.mean(cv_scores):.2f} days (±{np.std(cv_scores):.2f})")
    print("=" * 48)


if __name__ == "__main__":
    main()
