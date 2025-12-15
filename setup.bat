@echo off
setlocal enabledelayedexpansion

echo =========================================
echo   Bundle Adjustment Setup (Windows)
echo =========================================
echo.
echo NOTE: For better Windows support, use PowerShell instead:
echo   powershell -ExecutionPolicy Bypass -File setup.ps1
echo.
echo Continuing with batch script setup...
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Install from: https://www.python.org/downloads/
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [OK] Found !PYTHON_VERSION!

REM Check if venv exists
if exist venv (
    echo [WARNING] Virtual environment already exists at venv\
    set /p RECREATE="Do you want to recreate it? (y/N): "
    if /i "!RECREATE!"=="y" (
        echo Removing old venv...
        rmdir /s /q venv
    ) else (
        echo Using existing venv...
    )
)

REM Create virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip upgraded

REM Install requirements
if exist requirements.txt (
    echo Installing Python dependencies from requirements.txt...
    pip install -r requirements.txt
    echo [OK] Python dependencies installed
) else (
    echo [WARNING] No requirements.txt found, skipping Python dependencies
)

echo.
echo Checking build dependencies...

REM Check for CMake
cmake --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] cmake not found (required for building C++ BA module)
    echo   Install from: https://cmake.org/download/
) else (
    for /f "tokens=*" %%i in ('cmake --version ^| findstr /r "version"') do echo [OK] Found %%i
)

REM Check for C++ compiler
cl >nul 2>&1
if errorlevel 1 (
    echo [WARNING] MSVC compiler not found (required for building BA module)
    echo   Install Visual Studio with C++ tools
) else (
    echo [OK] Found MSVC compiler
)

echo.
echo [WARNING] NOTE: Ceres Solver must be installed separately
echo   Windows: Use vcpkg: vcpkg install ceres
echo   Or follow: http://ceres-solver.org/installation.html

echo.
echo =========================================
echo   Setup Complete!
echo =========================================
echo.
echo To activate the virtual environment:
echo   venv\Scripts\activate.bat
echo.
echo To run the bundle adjustment pipeline:
echo   bash replicate           # Use Dogleg solver (default)
echo   bash replicate -lm       # Use Levenberg-Marquardt solver
echo.
echo Note: Run 'replicate' in Git Bash or WSL
echo.

pause
