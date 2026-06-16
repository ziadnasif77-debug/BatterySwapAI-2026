#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup.sh — BatterySwapAI 2026 — Mac / Linux
# Usage:  chmod +x setup.sh && ./setup.sh
# ──────────────────────────────────────────────────────────────────────────────
set -e

PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   BatterySwapAI 2026 — Setup Script     ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. Check Python version ────────────────────────────────────────────────
echo "▸ Checking Python version..."
PYVER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYMAJ=$($PYTHON -c "import sys; print(sys.version_info.major)")
PYMIN=$($PYTHON -c "import sys; print(sys.version_info.minor)")

if [ "$PYMAJ" -lt 3 ] || { [ "$PYMAJ" -eq 3 ] && [ "$PYMIN" -lt 10 ]; }; then
    echo "  ✗ Python $PYVER found — need 3.10 or newer"
    echo "    Download: https://www.python.org/downloads/"
    exit 1
fi
echo "  ✓ Python $PYVER"

# ── 2. Create virtual environment ─────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "▸ Creating virtual environment in $VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"
fi

# ── 3. Activate venv ──────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"
echo "▸ Virtual environment: $(python --version) at $(which python)"

# ── 4. Upgrade pip ────────────────────────────────────────────────────────
echo "▸ Upgrading pip..."
pip install --upgrade pip --quiet

# ── 5. Install dependencies ───────────────────────────────────────────────
echo "▸ Installing packages from requirements.txt..."
pip install -r requirements.txt

# ── 6. Generate synthetic data ────────────────────────────────────────────
echo "▸ Generating synthetic sensor data..."
python data/raw/generate_dummy_data.py

# ── 7. Run feature pipeline ───────────────────────────────────────────────
echo "▸ Building feature matrix..."
python model/feature_pipeline.py

# ── 8. Train baseline + LightGBM models ──────────────────────────────────
echo "▸ Training baseline model..."
python model/baseline.py

echo "▸ Training LightGBM + calibration..."
python model/train.py

# ── 9. Uncertainty quantification ─────────────────────────────────────────
echo "▸ Computing prediction intervals & failure probabilities..."
python model/uncertainty.py

# ── 10. Optimization pipeline ─────────────────────────────────────────────
echo "▸ Scoring sensor priorities..."
python optimization/priority.py

echo "▸ Scheduling field visits (VRP)..."
python optimization/scheduler.py

echo "▸ Running cost simulations..."
python optimization/simulator.py

# ── 11. Build demo map ────────────────────────────────────────────────────
echo "▸ Building interactive Norway map..."
python demo/map_builder.py

# ── 12. Run test suite ────────────────────────────────────────────────────
echo ""
echo "▸ Running end-to-end test suite..."
python test_full_pipeline.py

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Setup complete!                        ║"
echo "║                                          ║"
echo "║   To launch the dashboard:               ║"
echo "║   source .venv/bin/activate              ║"
echo "║   streamlit run demo/dashboard.py        ║"
echo "║                                          ║"
echo "║   To open the map:                       ║"
echo "║   open demo/battery_map.html             ║"
echo "╚══════════════════════════════════════════╝"
echo ""
