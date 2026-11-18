$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  DAG_Brain (Causal Studio) - One-Click Self-Hosted Launcher" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
    Write-Host "[INFO] Initialized .env from .env.example" -ForegroundColor Green
}

# Fast-path: check if host Python already has required dependencies installed
$hasDeps = $false
try {
    python -c "import fastapi, uvicorn, networkx, httpx, pydantic" 2>$null
    if ($LASTEXITCODE -eq 0) { $hasDeps = $true }
} catch {}

if ($hasDeps) {
    Write-Host "[START] Starting DAG_Brain Studio at http://127.0.0.1:8000 ..." -ForegroundColor Green
    python -m backend.main --open
    exit 0
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$uvCmd = Get-Command "uv" -ErrorAction SilentlyContinue
if (-not $uvCmd -and (Test-Path "$env:USERPROFILE\.local\bin\uv.exe")) {
    $uvCmd = "$env:USERPROFILE\.local\bin\uv.exe"
}

if (-not (Test-Path $venvPython)) {
    Write-Host "[SETUP] Creating virtual environment and installing dependencies..." -ForegroundColor Yellow
    if ($uvCmd) {
        & $uvCmd venv .venv
        & $uvCmd pip install -r requirements.txt
    } else {
        python -m venv .venv
        & $venvPython -m pip install --upgrade pip
        & $venvPython -m pip install -r requirements.txt
    }
}

Write-Host "[START] Starting DAG_Brain Studio at http://127.0.0.1:8000 ..." -ForegroundColor Green
& $venvPython -m backend.main --open
