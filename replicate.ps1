# Bundle Adjustment Pipeline Runner for Windows
# Usage: .\replicate.ps1 [-lm|-dogleg]

param(
    [switch]$lm,
    [switch]$dogleg
)

$ErrorActionPreference = "Stop"

# Determine solver
$solver = "dogleg"
if ($lm) {
    $solver = "lm"
} elseif ($dogleg) {
    $solver = "dogleg"
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Check if venv is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "WARNING: Virtual environment not activated!" -ForegroundColor Yellow
    Write-Host "Activating venv..." -ForegroundColor Gray
    if (Test-Path "venv\Scripts\Activate.ps1") {
        & .\venv\Scripts\Activate.ps1
    } else {
        Write-Host "ERROR: venv not found. Run setup.ps1 first!" -ForegroundColor Red
        exit 1
    }
}

# Set paths
$ROOT = $ScriptDir
$SCRIPTS = Join-Path $ROOT "scripts"
$SDK = Join-Path $ROOT "robotcar-dataset-sdk-3.1"
$DATA = Join-Path $ROOT "sample"
$EXTR = Join-Path $SDK "extrinsics"
$MODELS = Join-Path $SDK "models"
$OUTPUT = Join-Path $ROOT "output"

# Create output directory
New-Item -ItemType Directory -Force -Path $OUTPUT | Out-Null

# Config
$DURATION_S = "10.0"
$FRAME_STRIDE = "2"
$MAX_FRAMES = "10"
$LEFT_STREAM = "stereo/left"
$RIGHT_STREAM = "stereo/right"
$INTR = Join-Path $OUTPUT "ba_intrinsics_stereo_left.json"

Write-Host "[0/3] Dumping camera intrinsics..." -ForegroundColor Cyan
python (Join-Path $SCRIPTS "dump_intrinsics.py") `
    --dataset_root $DATA `
    --models_dir $MODELS `
    --camera_stream "stereo/left" `
    --output $INTR

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[1/3] Building pointcloud (duration=${DURATION_S}s)..." -ForegroundColor Cyan
python (Join-Path $SCRIPTS "make_pointcloud.py") `
    --dataset_root $DATA `
    --extrinsics_dir $EXTR `
    --lidar "lms_front" `
    --poses_file "gps/ins.csv" `
    --duration $DURATION_S `
    --output (Join-Path $OUTPUT "pointcloud.npz")

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/3] Building ORB-stereo BA dataset (duration=${DURATION_S}s, stride=${FRAME_STRIDE}, max_frames=${MAX_FRAMES})" -ForegroundColor Cyan
python (Join-Path $SCRIPTS "make_ba_dataset_orb_stereo.py") `
    --dataset_root $DATA `
    --extrinsics_dir $EXTR `
    --models_dir $MODELS `
    --poses_file "gps/ins.csv" `
    --left_stream $LEFT_STREAM `
    --right_stream $RIGHT_STREAM `
    --left_timestamps (Join-Path $DATA "stereo.timestamps") `
    --right_timestamps (Join-Path $DATA "stereo.timestamps") `
    --duration $DURATION_S `
    --frame_stride $FRAME_STRIDE `
    --max_frames $MAX_FRAMES `
    --output_prefix (Join-Path $OUTPUT "ba_robotcar")

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[3/3] Done." -ForegroundColor Green
Write-Host "Created:" -ForegroundColor Gray
Write-Host "  $(Join-Path $OUTPUT 'pointcloud.npz')" -ForegroundColor Gray
Write-Host "  $(Join-Path $OUTPUT 'ba_robotcar_poses.txt')" -ForegroundColor Gray
Write-Host "  $(Join-Path $OUTPUT 'ba_robotcar_points.txt')" -ForegroundColor Gray
Write-Host "  $(Join-Path $OUTPUT 'ba_robotcar_observations.txt')" -ForegroundColor Gray
Write-Host "  $INTR" -ForegroundColor Gray

Write-Host "[4/4] Running Bundle Adjustment with solver: $solver" -ForegroundColor Cyan
$env:BA_ROOT = $ROOT

# Build and run BA module
$buildDir = Join-Path $ROOT "build"
if (Test-Path $buildDir) {
    Remove-Item -Recurse -Force $buildDir
}
New-Item -ItemType Directory -Path $buildDir | Out-Null
Set-Location $buildDir

cmake ..
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build . --config Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Run BA module
$baModule = Join-Path $buildDir "Release\ba_module.exe"
if (-not (Test-Path $baModule)) {
    $baModule = Join-Path $buildDir "ba_module.exe"
}

if (Test-Path $baModule) {
    & $baModule $solver
} else {
    Write-Host "ERROR: ba_module.exe not found!" -ForegroundColor Red
    exit 1
}

Set-Location $ROOT
