<#
.SYNOPSIS
    Creates/activates the local .venv and installs project dependencies.
.DESCRIPTION
    Run dot-sourced (`. .\setup.ps1`) in PowerShell to keep .venv active in the current shell.
#>

$ErrorActionPreference = "Stop"

$VenvDir = ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found on PATH. Install Python 3.10+ and try again."
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment in $VenvDir..." -ForegroundColor Cyan
    python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment in $VenvDir."
    }
} else {
    Write-Host "Virtual environment already exists at $VenvDir." -ForegroundColor Yellow
}

if (Test-Path $ActivateScript) {
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    . $ActivateScript
}

Write-Host "Upgrading pip and installing dependencies from requirements.txt..." -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip."
}

& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install dependencies from requirements.txt."
}

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "To ensure .venv stays active in your terminal, run: .\.venv\Scripts\Activate.ps1"
