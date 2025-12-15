# Bundle Adjustment Setup Script for Windows (PowerShell)
# Usage: .\setup.ps1

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Bundle Adjustment Setup (Windows)" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Check for Python
Write-Host "Checking for Python..." -NoNewline
try {
    $pythonVersion = python --version 2>&1
    Write-Host " OK" -ForegroundColor Green
    Write-Host "  Found: $pythonVersion" -ForegroundColor Gray
} catch {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    Write-Host "Install from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation!" -ForegroundColor Yellow
    exit 1
}

# Check if venv exists
if (Test-Path "venv") {
    Write-Host ""
    Write-Host "Virtual environment already exists at venv\" -ForegroundColor Yellow
    $recreate = Read-Host "Do you want to recreate it? (y/N)"
    if ($recreate -eq "y" -or $recreate -eq "Y") {
        Write-Host "Removing old venv..." -ForegroundColor Gray
        Remove-Item -Recurse -Force venv
    } else {
        Write-Host "Using existing venv..." -ForegroundColor Gray
    }
}

# Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host ""
    Write-Host "Creating virtual environment..." -NoNewline
    python -m venv venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host " OK" -ForegroundColor Green
    } else {
        Write-Host " FAILED" -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -NoNewline
& .\venv\Scripts\Activate.ps1
Write-Host " OK" -ForegroundColor Green

# Upgrade pip
Write-Host "Upgrading pip..." -NoNewline
python -m pip install --upgrade pip --quiet
Write-Host " OK" -ForegroundColor Green

# Install requirements
if (Test-Path "requirements.txt") {
    Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
    pip install -r requirements.txt
    Write-Host "Python dependencies installed" -ForegroundColor Green
} else {
    Write-Host "WARNING: No requirements.txt found" -ForegroundColor Yellow
}

# Check for build dependencies
Write-Host ""
Write-Host "Checking build dependencies..." -ForegroundColor Cyan

# Check for CMake
Write-Host "  CMake: " -NoNewline
try {
    $cmakeVersion = cmake --version 2>&1 | Select-String "version" | Select-Object -First 1
    Write-Host "OK ($cmakeVersion)" -ForegroundColor Green
} catch {
    Write-Host "NOT FOUND" -ForegroundColor Yellow
    Write-Host "    Install from: https://cmake.org/download/" -ForegroundColor Gray
}

# Check for MSVC
Write-Host "  MSVC: " -NoNewline
try {
    $null = cl 2>&1
    if ($LASTEXITCODE -eq 0 -or $LASTEXITCODE -eq 1) {
        Write-Host "OK" -ForegroundColor Green
    } else {
        throw
    }
} catch {
    Write-Host "NOT FOUND" -ForegroundColor Yellow
    Write-Host "    Install Visual Studio with C++ tools" -ForegroundColor Gray
}

# Check for Git Bash (needed for replicate script)
Write-Host "  Git Bash: " -NoNewline
$gitBashPaths = @(
    "C:\Program Files\Git\bin\bash.exe",
    "C:\Program Files (x86)\Git\bin\bash.exe",
    "$env:LOCALAPPDATA\Programs\Git\bin\bash.exe"
)
$gitBashFound = $false
foreach ($path in $gitBashPaths) {
    if (Test-Path $path) {
        Write-Host "OK ($path)" -ForegroundColor Green
        $gitBashFound = $true
        break
    }
}
if (-not $gitBashFound) {
    Write-Host "NOT FOUND" -ForegroundColor Yellow
    Write-Host "    Install from: https://git-scm.com/download/win" -ForegroundColor Gray
    Write-Host "    (Needed to run the bash replicate script)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "NOTE: Ceres Solver required for C++ BA module" -ForegroundColor Yellow
Write-Host "  Install with vcpkg: vcpkg install ceres" -ForegroundColor Gray
Write-Host "  Or follow: http://ceres-solver.org/installation.html" -ForegroundColor Gray

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run the bundle adjustment pipeline:" -ForegroundColor White
Write-Host "  1. Keep this PowerShell window open (venv is active)" -ForegroundColor Gray
Write-Host "  2. Run: " -NoNewline -ForegroundColor Gray
Write-Host ".\replicate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "OR if using Git Bash:" -ForegroundColor White
Write-Host "  bash replicate" -ForegroundColor Yellow
Write-Host ""
Write-Host "Solver options:" -ForegroundColor White
Write-Host "  .\replicate.ps1           # Dogleg (default)" -ForegroundColor Gray
Write-Host "  .\replicate.ps1 -lm       # Levenberg-Marquardt" -ForegroundColor Gray
Write-Host ""
