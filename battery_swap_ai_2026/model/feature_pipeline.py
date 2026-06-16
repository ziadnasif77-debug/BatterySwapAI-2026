"""
model/feature_pipeline.py
Combines F01–F05 into one pipeline and measures feature group importance.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.features import (
    add_slope_features,
    add_rolling_features,
    add_temperature_features,
    add_lifecycle_features,
    add_curve_features,
)

DATA_PATH   = Path(__file__).parent.parent / "data" / "raw" / "sensor_readings.csv"
PROC_DIR    = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Ordered feature columns (matches F01→F05 build order)
FEATURE_COLS = [
    # F01 — Voltage Slopes
    "slope_7d", "slope_14d", "slope_30d", "slope_all", "acceleration",
    # F02 — Rolling Voltage Stats
    "voltage_mean_7d", "voltage_std_7d", "voltage_min_7d", "voltage_drop_7d",
    "voltage_mean_14d", "voltage_std_14d", "voltage_min_14d", "voltage_drop_14d",
    "voltage_mean_30d", "voltage_std_30d", "voltage_min_30d", "voltage_drop_30d",
    "voltage_current", "voltage_max_ever", "voltage_min_ever",
    "voltage_range_all", "voltage_pct",
    # F03 — Temperature Impact
    "temp_mean_all", "temp_mean_14d", "temp_min_ever", "temp_std_all",
    "temp_current", "days_below_0", "days_below_minus10",
    "cold_exposure_pct", "is_winter_now", "volt_temp_corr",
    # F04 — Lifecycle Position
    "battery_age_days", "lifecycle_position", "readings_per_day",
    "days_since_last_reading", "voltage_at_30d", "voltage_at_60d",
    "voltage_at_90d", "month_of_year",
    # F05 — Curve Shape
    "curve_fit_ok", "decay_rate", "decay_floor", "decay_amplitude",
    "decay_phase", "days_to_floor",
]

FEATURE_GROUP = {
    **{c: "F01: Slopes"     for c in ["slope_7d", "slope_14d", "slope_30d", "slope_all", "acceleration"]},
    **{c: "F02: Rolling"    for c in [
        "voltage_mean_7d", "voltage_std_7d", "voltage_min_7d", "voltage_drop_7d",
        "voltage_mean_14d", "voltage_std_14d", "voltage_min_14d", "voltage_drop_14d",
        "voltage_mean_30d", "voltage_std_30d", "voltage_min_30d", "voltage_drop_30d",
        "voltage_current", "voltage_max_ever", "voltage_min_ever",
        "voltage_range_all", "voltage_pct",
    ]},
    **{c: "F03: Temperature" for c in [
        "temp_mean_all", "temp_mean_14d", "temp_min_ever", "temp_std_all",
        "temp_current", "days_below_0", "days_below_minus10",
        "cold_exposure_pct", "is_winter_now", "volt_temp_corr",
    ]},
    **{c: "F04: Lifecycle"  for c in [
        "battery_age_days", "lifecycle_position", "readings_per_day",
        "days_since_last_reading", "voltage_at_30d", "voltage_at_60d",
        "voltage_at_90d", "month_of_year",
    ]},
    **{c: "F05: Curve"      for c in [
        "curve_fit_ok", "decay_rate", "decay_floor", "decay_amplitude",
        "decay_phase", "days_to_floor",
    ]},
}

GROUP_COLORS = {
    "F01: Slopes":      "#3498db",
    "F02: Rolling":     "#2ecc71",
    "F03: Temperature": "#e67e22",
    "F04: Lifecycle":   "#9b59b6",
    "F05: Curve":       "#e74c3c",
}


def build_all_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full F01–F05 feature pipeline on raw sensor data.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw sensor readings (sensor_id, building_id, timestamp,
        voltage, temperature, end_of_life_date).

    Returns
    -------
    pd.DataFrame with all original columns plus 46 engineered features.
    """
    print("  [F01] Computing voltage slope features...")
    df = add_slope_features(raw_df)

    print("  [F02] Computing rolling voltage statistics...")
    df = add_rolling_features(df)

    print("  [F03] Computing temperature impact features...")
    df = add_temperature_features(df)

    print("  [F04] Computing lifecycle position features...")
    df = add_lifecycle_features(df)

    print("  [F05] Computing curve shape features (exponential fit)...")
    df = add_curve_features(df)

    return df


def build_training_dataset(raw_df: pd.DataFrame) -> tuple:
    """
    Creates labeled training data with STRICT time-based split.

    Labels come only from sensors with known end_of_life_date.
    For each reading of a dead sensor:
        actual_rul = (end_of_life_date - timestamp).days
    Keeps rows where 0 <= actual_rul <= 365.

    TIME-BASED SPLIT (never random):
        Sort by timestamp → find 80% position → gap of 7 days.
        Train = timestamp <= split_ts
        Test  = timestamp >  split_ts + 7 days

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    print("\nBuilding feature matrix...")
    feat_df = build_all_features(raw_df)

    # Propagate death dates to every row of each dead sensor
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
    labeled = labeled.sort_values("timestamp").reset_index(drop=True)

    # Time-based 80/20 split with 7-day gap
    split_idx = int(len(labeled) * 0.80)
    split_ts  = labeled["timestamp"].iloc[split_idx]
    gap_ts    = split_ts + pd.Timedelta(days=7)

    train = labeled[labeled["timestamp"] <= split_ts]
    test  = labeled[labeled["timestamp"] > gap_ts]

    # Feature matrix — only keep features that exist in df
    feat_names = [c for c in FEATURE_COLS if c in labeled.columns]

    X_train = train[feat_names].values
    X_test  = test[feat_names].values
    y_train = train["actual_rul"].values
    y_test  = test["actual_rul"].values

    return X_train, X_test, y_train, y_test, feat_names


def measure_feature_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
) -> pd.DataFrame:
    """
    Trains a quick LightGBM regressor (100 trees) and returns
    feature importances by gain, sorted descending.

    Returns
    -------
    pd.DataFrame: feature_name, importance, cumulative_pct
    """
    print("\nTraining LightGBM (100 trees) for feature importance...")
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)

    params = {
        "objective":       "regression",
        "metric":          "mae",
        "num_leaves":      31,
        "learning_rate":   0.05,
        "n_estimators":    100,
        "verbosity":       -1,
        "random_state":    42,
    }
    model = lgb.train(params, train_data, num_boost_round=100)

    imp = pd.DataFrame({
        "feature_name": feature_names,
        "importance":   model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    total = imp["importance"].sum()
    imp["importance_pct"] = imp["importance"] / total * 100
    imp["cumulative_pct"] = imp["importance_pct"].cumsum()

    print("\n" + "=" * 62)
    print("  TOP 15 FEATURES BY IMPORTANCE (gain)")
    print("=" * 62)
    print(f"  {'Rank':<5} {'Feature':<28} {'Group':<16} {'Imp%':>6}")
    print("-" * 62)
    for rank, row in imp.head(15).iterrows():
        group = FEATURE_GROUP.get(row["feature_name"], "—")
        print(f"  {rank+1:<5} {row['feature_name']:<28} {group:<16} {row['importance_pct']:>5.1f}%")
    print("=" * 62)

    zero_imp = imp[imp["importance"] == 0]["feature_name"].tolist()
    if zero_imp:
        print(f"\n  Zero-importance features ({len(zero_imp)}): {zero_imp}")
    else:
        print("\n  No zero-importance features.")

    return imp


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw sensor data...")
    raw_df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    print(f"  {len(raw_df):,} rows, {raw_df['sensor_id'].nunique()} sensors")

    X_train, X_test, y_train, y_test, feat_names = build_training_dataset(raw_df)

    print(f"\nTraining samples : {len(y_train):,}")
    print(f"Test samples     : {len(y_test):,}")
    print(f"Features built   : {len(feat_names)}")

    imp_df = measure_feature_importance(X_train, y_train, feat_names)

    # ── Group-level summary ────────────────────────────────────────────────
    imp_df["group"] = imp_df["feature_name"].map(FEATURE_GROUP)
    group_summary = (imp_df.groupby("group")["importance_pct"]
                     .sum().sort_values(ascending=False).reset_index())
    print("\nFeature GROUP importance:")
    for _, row in group_summary.iterrows():
        print(f"  {row['group']:<20} {row['importance_pct']:>6.1f}%")

    # ── Feature importance plot ────────────────────────────────────────────
    top_n   = min(20, len(imp_df))
    top_imp = imp_df.head(top_n).iloc[::-1]  # reverse for horizontal bar

    colors = [GROUP_COLORS.get(FEATURE_GROUP.get(f, ""), "#95a5a6")
              for f in top_imp["feature_name"]]

    fig, ax = plt.subplots(figsize=(11, 9))
    bars = ax.barh(top_imp["feature_name"], top_imp["importance_pct"],
                   color=colors, edgecolor="white", linewidth=0.4)

    for bar, pct in zip(bars, top_imp["importance_pct"]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=8)

    # Legend for groups
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.8)

    ax.set_xlabel("Feature Importance (% of total gain)", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances — LightGBM Gain\n"
                 f"(trained on {len(y_train):,} labeled samples from 15 dead sensors)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, top_imp["importance_pct"].max() * 1.18)
    plt.tight_layout()

    out_plot = RESULTS_DIR / "feature_importance.png"
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_plot}")

    # ── Save full feature CSV ──────────────────────────────────────────────
    print("Building full feature CSV (all sensors, all rows)...")
    feat_df = build_all_features(raw_df)
    out_csv = PROC_DIR / "features_full.csv"
    feat_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}  ({out_csv.stat().st_size // 1024} KB, {len(feat_df):,} rows)")

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  PIPELINE COMPLETE")
    print("=" * 50)
    print(f"  Total features built  : {len(feat_names)}")
    print(f"  Training samples      : {len(y_train):,}")
    print(f"  Test samples          : {len(y_test):,}")
    print(f"  Top feature           : {imp_df.iloc[0]['feature_name']}  "
          f"({imp_df.iloc[0]['importance_pct']:.1f}%)")
    top5 = imp_df.head(5)["feature_name"].tolist()
    print(f"  Top 5 features        : {top5}")
    zero_imp = imp_df[imp_df["importance"] == 0]["feature_name"].tolist()
    print(f"  Zero-importance cols  : {len(zero_imp)}")
    if zero_imp:
        print(f"    → {zero_imp}")
    print("=" * 50)


if __name__ == "__main__":
    main()
