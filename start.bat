@echo off
setlocal
title DAG_Brain - AI Causal Modelling & Research Notebook Studio
cd /d "%~dp0"

echo ============================================================================
echo   DAG_Brain (Causal Studio) - One-Click Self-Hosted Launcher
echo ============================================================================

if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo [INFO] Created default .env configuration file.
    )
)

REM Fast-path: check if current Python already has all required dependencies
python -c "import fastapi, uvicorn, networkx, httpx, pydantic" >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [START] Launching DAG_Brain Studio on http://127.0.0.1:8000 ...
    python -m backend.main --open
    goto :eof
)

if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] Creating isolated Python virtual environment...
    where uv >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        uv venv .venv
        uv pip install -r requirements.txt
    ) else (
        python -m venv .venv
        ".venv\Scripts\python.exe" -m pip install --upgrade pip
        ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    )
)

echo [START] Launching DAG_Brain Studio on http://127.0.0.1:8000 ...
".venv\Scripts\python.exe" -m backend.main --open
pause
