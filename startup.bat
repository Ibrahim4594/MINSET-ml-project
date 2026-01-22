@echo off
setlocal enabledelayedexpansion

echo.
echo ═══════════════════════════════════════════════════
echo     MNIST Digit Recognition - Startup
echo ═══════════════════════════════════════════════════
echo.

REM Check Python
echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Please install Python 3.8+
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% found

REM Create virtual environment
echo [2/5] Setting up environment...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing packages...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo Some packages failed. Trying with --no-cache-dir...
    pip install --no-cache-dir -q -r requirements.txt
)
echo [OK] Dependencies installed

REM Check model
echo [3/5] Checking model...
if not exist "models\mnist_model.h5" (
    echo Model not found. Training...
    python train_model.py
    echo [OK] Model trained
) else (
    echo [OK] Model found
)

REM Create directories
echo [4/5] Creating directories...
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "visualizations" mkdir visualizations
if not exist "prediction_data" mkdir prediction_data
if not exist "api_logs" mkdir api_logs
echo [OK] Directories ready

REM Start application
echo [5/5] Starting application...
echo.
echo ═══════════════════════════════════════════════════
echo  Application running on http://localhost:5000
echo  Press Ctrl+C to stop
echo ═══════════════════════════════════════════════════
echo.

python app.py

pause
