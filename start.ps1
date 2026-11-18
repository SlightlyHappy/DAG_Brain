$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  DAG Brain - self-hosted launcher" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
    Write-Host "[INFO] Initialized .env from .env.example" -ForegroundColor Green
}

function Test-Env([string]$py) {
    try {
        & $py -m backend.check_env *> $null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

# Use the system Python only if it has every requirement at the right version.
if (Test-Env "python") {
    Write-Host "[START] Starting DAG Brain at http://127.0.0.1:8000 ..." -ForegroundColor Green
    python -m backend.main --open
    exit 0
}

Write-Host "[SETUP] The system Python is missing or has outdated packages:" -ForegroundColor Yellow
try { python -m backend.check_env } catch {}
Write-Host "[SETUP] Using an isolated environment in .venv instead." -ForegroundColor Yellow

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$uvCmd = (Get-Command "uv" -ErrorAction SilentlyContinue).Source
if (-not $uvCmd -and (Test-Path "$env:USERPROFILE\.local\bin\uv.exe")) { $uvCmd = "$env:USERPROFILE\.local\bin\uv.exe" }

if (-not (Test-Path $venvPython)) {
    Write-Host "[SETUP] Creating the virtual environment..." -ForegroundColor Yellow
    if ($uvCmd) { & $uvCmd venv .venv } else { python -m venv .venv; & $venvPython -m pip install --upgrade pip }
}

# Install or update dependencies whenever requirements.txt asks for more than .venv has.
if (-not (Test-Env $venvPython)) {
    Write-Host "[SETUP] Installing dependencies..." -ForegroundColor Yellow
    if ($uvCmd) { & $uvCmd pip install --python $venvPython -r requirements.txt } else { & $venvPython -m pip install -r requirements.txt }
}

if (-not (Test-Env $venvPython)) {
    & $venvPython -m backend.check_env
    Write-Host "[ERROR] Dependencies could not be installed. See the messages above." -ForegroundColor Red
    exit 1
}

Write-Host "[START] Starting DAG Brain at http://127.0.0.1:8000 ..." -ForegroundColor Green
& $venvPython -m backend.main --open
