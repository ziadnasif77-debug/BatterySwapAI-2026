# BatterySwapAI 2026

## Project Overview

BatterySwapAI 2026 predicts the Remaining Useful Life (RUL) of IoT battery sensors deployed across 20 Norwegian buildings using LightGBM with building-type calibration and 90% confidence intervals. It optimises daily field-worker routes via an OR-Tools Vehicle Routing Problem solver to minimise battery downtime costs. The recommended AGGRESSIVE scheduling strategy reduces total operational cost to 5,328,458 NOK with a 28.6% sensor rescue rate.

## Installation

```bash
pip install -r requirements.txt
```

## How to Run the Model

```bash
python model/feature_pipeline.py   # Build 46 features from raw sensor readings
python model/train.py              # Train LightGBM (saves results/lightgbm_model.pkl)
python model/calibrate.py          # Calibrate predictions by building type
python model/uncertainty.py        # Add 90% confidence intervals + failure probabilities
python optimization/priority.py    # Score and rank all sensors by risk
python optimization/scheduler.py   # Generate VRP-based work orders
python optimization/simulator.py   # Compare AGGRESSIVE / NORMAL / CONSERVATIVE cost scenarios
```

## How to Run the Dashboard

```bash
streamlit run demo/dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. The dashboard shows the interactive Norway sensor map, voltage decline forecasts with confidence bands, today's work orders, and model performance metrics.

To regenerate the map standalone:

```bash
python demo/map_builder.py
# Output: demo/battery_map.html
```

## Results Summary

| Metric | Value |
|---|---|
| MAE after calibration | **6.2 days** (−47.8% vs raw 11.9 days) |
| RMSE after calibration | 7.6 days |
| Interval coverage (90% CI) | 26.5% (limited by 15 dead sensors + temporal shift) |
| Recommended strategy | **AGGRESSIVE** (p_fail_7d > 0.3%) |
| Sensors rescued on time | 2 / 7 (28.6%) |
| Total cost (AGGRESSIVE) | **5,328,458 NOK** |
| Total cost (CONSERVATIVE) | 5,520,000 NOK |

## File Structure

```
battery_swap_ai_2026/
├── model/
│   ├── features.py           # F01–F05 feature engineering (46 features)
│   ├── feature_pipeline.py   # Feature pipeline + LightGBM importance ranking
│   ├── train.py              # LightGBM training with time-series cross-validation
│   ├── calibrate.py          # Per-building-type bias calibration
│   └── uncertainty.py        # HistGradientBoosting quantile models (90% CI)
├── optimization/
│   ├── priority.py           # Risk scoring (0–100) + sensor prioritization
│   ├── scheduler.py          # OR-Tools VRP work-order generator
│   └── simulator.py          # Cost scenario simulator (3 strategies)
├── demo/
│   ├── map_builder.py        # Folium interactive map (dark theme, Norway)
│   └── dashboard.py          # Streamlit 4-tab live monitor
├── data/
│   ├── raw/                  # sensor_readings.csv, buildings.csv, travel_times.csv
│   └── processed/            # features_full.csv (46 features × 9,205 rows)
├── results/
│   ├── predictions.csv       # Final RUL predictions (1 row per sensor)
│   ├── work_orders.csv       # Today's field worker schedule
│   ├── metrics.json          # Model performance summary
│   ├── scenario_comparison.json
│   └── lightgbm_model.pkl
└── test_full_pipeline.py     # End-to-end pipeline validation (5/7 checks pass)
```
