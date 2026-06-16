# BatterySwapAI-2026

AI-powered battery swap scheduling for IoT sensors across 20 Norwegian buildings.

## Overview

Predicts Remaining Useful Life (RUL) of battery sensors using LightGBM, calibrates predictions by building type, quantifies uncertainty with 90% confidence intervals, and optimizes field worker routes using OR-Tools VRP.

## Project Structure

```
battery_swap_ai_2026/
├── model/
│   ├── features.py           # F01–F05 feature engineering (46 features)
│   ├── feature_pipeline.py   # Feature pipeline + importance ranking
│   ├── train.py              # LightGBM training + cross-validation
│   ├── calibrate.py          # Bias calibration by building type
│   └── uncertainty.py        # 90% prediction intervals + failure probabilities
├── optimization/
│   ├── priority.py           # Risk scoring + sensor prioritization
│   ├── scheduler.py          # OR-Tools VRP work order scheduler
│   └── simulator.py          # Cost scenario simulator (AGGRESSIVE/NORMAL/CONSERVATIVE)
├── demo/
│   ├── map_builder.py        # Interactive Folium map (Norway sensor locations)
│   └── dashboard.py          # Streamlit live monitor dashboard
├── data/raw/                 # sensor_readings.csv, buildings.csv, travel_times.csv
├── data/processed/           # features_full.csv
└── results/                  # Predictions, work orders, scenario comparison
```

## Results

| Metric | Value |
|---|---|
| LightGBM MAE (raw) | 11.9 days |
| MAE after calibration | 6.2 days (−47.8%) |
| Interval coverage | 26.5% (90% target; limited by dataset size) |
| Recommended strategy | AGGRESSIVE — 5,328,458 NOK total cost |
| Sensors saved (AGGRESSIVE) | 28.6% |

## Running the Dashboard

```bash
streamlit run demo/dashboard.py
```

## Generating the Map

```bash
python demo/map_builder.py
# Opens: demo/battery_map.html
```

## Running the Full Pipeline

```bash
python model/feature_pipeline.py   # Build features
python model/train.py              # Train LightGBM
python model/calibrate.py          # Calibrate predictions
python model/uncertainty.py        # Add confidence intervals
python optimization/priority.py    # Risk scoring
python optimization/scheduler.py   # Generate work orders
python optimization/simulator.py   # Run cost scenarios
```
