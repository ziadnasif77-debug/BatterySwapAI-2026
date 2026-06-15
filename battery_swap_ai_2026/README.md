# BatterySwapAI 2026

An AI-powered system for optimizing battery swap operations — predicting swap demand, prioritizing work orders, and scheduling field technicians to maximize uptime across EV battery swap stations.

## Project Overview

BatterySwapAI 2026 uses machine learning to:

- **Predict** battery swap demand across stations using time-series and contextual features
- **Prioritize** work orders based on urgency, location, and technician availability
- **Schedule** field operations to minimize downtime and maximize coverage
- **Simulate** station operations to evaluate scheduling strategies before deployment

## Project Structure

```
battery_swap_ai_2026/
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── MASTER_PROMPT.md        # AI assistant master prompt
├── data/
│   ├── raw/               # Raw input data (unmodified)
│   └── processed/         # Cleaned and transformed data
├── model/                 # ML model code
│   ├── baseline.py        # Baseline prediction model
│   ├── features.py        # Feature engineering pipeline
│   ├── train.py           # Model training entry point
│   ├── predict.py         # Inference / prediction
│   ├── calibrate.py       # Probability calibration
│   └── uncertainty.py     # Uncertainty quantification
├── optimization/          # Scheduling and optimization
│   ├── priority.py        # Work order priority scoring
│   ├── scheduler.py       # Technician scheduling engine
│   └── simulator.py       # Station operation simulator
├── demo/                  # Interactive demo and visualization
│   ├── dashboard.py       # Streamlit dashboard
│   └── map_builder.py     # Station map visualization
├── results/               # Output artifacts
│   ├── predictions.csv    # Model predictions
│   ├── work_orders.csv    # Optimized work order list
│   └── metrics.json       # Evaluation metrics
└── notebooks/             # Jupyter exploration notebooks
    ├── 01_exploration.ipynb
    ├── 02_features.ipynb
    ├── 03_model.ipynb
    └── 04_optimization.ipynb
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python -m model.train

# Generate predictions
python -m model.predict

# Run the demo dashboard
streamlit run demo/dashboard.py
```

## Data

Place raw CSV/JSON data files in `data/raw/`. The feature pipeline (`model/features.py`) reads from `data/raw/` and writes processed feature matrices to `data/processed/`.

## Results

Outputs are written to `results/`:

- `predictions.csv` — per-station demand forecasts with confidence intervals
- `work_orders.csv` — prioritized and scheduled work orders
- `metrics.json` — model accuracy, calibration, and scheduling KPIs

## License

MIT
