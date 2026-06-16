@echo off
REM ──────────────────────────────────────────────────────────────────────────
REM setup.bat — BatterySwapAI 2026 — Windows
REM Usage: double-click setup.bat, or run from Command Prompt / PowerShell
REM ──────────────────────────────────────────────────────────────────────────
setlocal enabledelayedexpansion

echo.
echo ╔══════════════════════════════════════════╗
echo ║   BatterySwapAI 2026 — Setup Script     ║
echo ╚══════════════════════════════════════════╝
echo.

REM ── 1. Check Python ──────────────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Download from https://www.python.org/downloads/
    echo         Make sure to tick "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Found Python %PYVER%

REM Require 3.10+
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJ=%%a
    set PYMIN=%%b
)
if %PYMAJ% LSS 3 (
    echo [ERROR] Need Python 3.10+, found %PYVER%
    pause & exit /b 1
)
if %PYMAJ% EQU 3 if %PYMIN% LSS 10 (
    echo [ERROR] Need Python 3.10+, found %PYVER%
    pause & exit /b 1
)

REM ── 2. Create virtual environment ─────────────────────────────────────────
if not exist ".venv\" (
    echo [..] Creating virtual environment...
    python -m venv .venv
)

REM ── 3. Activate venv ──────────────────────────────────────────────────────
call .venv\Scripts\activate.bat
echo [OK] Virtual environment activated

REM ── 4. Upgrade pip ────────────────────────────────────────────────────────
echo [..] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM ── 5. Install packages ───────────────────────────────────────────────────
echo [..] Installing packages from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Package installation failed. Check your internet connection.
    pause & exit /b 1
)
echo [OK] All packages installed

REM ── 6. Generate data ──────────────────────────────────────────────────────
echo [..] Generating synthetic sensor data...
python data\raw\generate_dummy_data.py

REM ── 7. Feature pipeline ───────────────────────────────────────────────────
echo [..] Building feature matrix...
python model\feature_pipeline.py

REM ── 8. Train models ───────────────────────────────────────────────────────
echo [..] Training baseline model...
python model\baseline.py

echo [..] Training LightGBM + calibration...
python model\train.py

REM ── 9. Uncertainty ────────────────────────────────────────────────────────
echo [..] Computing prediction intervals...
python model\uncertainty.py

REM ── 10. Optimization ──────────────────────────────────────────────────────
echo [..] Scoring sensor priorities...
python optimization\priority.py

echo [..] Scheduling field visits (VRP)...
python optimization\scheduler.py

echo [..] Running cost simulations...
python optimization\simulator.py

REM ── 11. Demo map ──────────────────────────────────────────────────────────
echo [..] Building interactive Norway map...
python demo\map_builder.py

REM ── 12. Test suite ────────────────────────────────────────────────────────
echo.
echo [..] Running end-to-end test suite...
python test_full_pipeline.py

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║  Setup complete!                                        ║
echo ║                                                         ║
echo ║  To launch the dashboard:                               ║
echo ║    .venv\Scripts\activate.bat                           ║
echo ║    streamlit run demo\dashboard.py                      ║
echo ║                                                         ║
echo ║  To open the map:                                       ║
echo ║    start demo\battery_map.html                          ║
echo ╚══════════════════════════════════════════════════════════╝
echo.
pause
