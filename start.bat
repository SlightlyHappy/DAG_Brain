@REM SPDX-License-Identifier: AGPL-3.0-or-later
@REM Copyright (C) 2026 SlightlyHappy. DAG Brain: https://github.com/SlightlyHappy/DAG_Brain
@REM Created and maintained by SlightlyHappy. See NOTICE for the attribution terms.
@echo off
setlocal
title DAG Brain
cd /d "%~dp0"

echo ============================================================================
echo   DAG Brain - self-hosted launcher
echo ============================================================================

if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo [INFO] Created default .env configuration file.
    )
)

REM Use the system Python only if it has every requirement at the right version.
python -m backend.check_env >nul 2>nul
if not errorlevel 1 (
    echo [START] Launching DAG Brain on http://127.0.0.1:8000 ...
    python -m backend.main --open
    goto :eof
)

echo [SETUP] The system Python is missing or has outdated packages:
python -m backend.check_env 2>nul
echo [SETUP] Using an isolated environment in .venv instead.

set "UVCMD="
where uv >nul 2>nul
if not errorlevel 1 set "UVCMD=uv"
if not defined UVCMD if exist "%USERPROFILE%\.local\bin\uv.exe" set "UVCMD=%USERPROFILE%\.local\bin\uv.exe"

if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] Creating the virtual environment...
    if defined UVCMD (
        "%UVCMD%" venv .venv
    ) else (
        python -m venv .venv
        ".venv\Scripts\python.exe" -m pip install --upgrade pip
    )
)

REM Install or update dependencies whenever requirements.txt asks for more than .venv has.
".venv\Scripts\python.exe" -m backend.check_env >nul 2>nul
if errorlevel 1 (
    echo [SETUP] Installing dependencies...
    if defined UVCMD (
        "%UVCMD%" pip install --python ".venv\Scripts\python.exe" -r requirements.txt
    ) else (
        ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    )
)

".venv\Scripts\python.exe" -m backend.check_env
if errorlevel 1 (
    echo [ERROR] Dependencies could not be installed. See the messages above.
    pause
    goto :eof
)

echo [START] Launching DAG Brain on http://127.0.0.1:8000 ...
".venv\Scripts\python.exe" -m backend.main --open
pause
