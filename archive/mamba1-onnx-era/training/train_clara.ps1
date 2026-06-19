# CLARA Training — PowerShell Launcher
# Sets ROCm environment variables and launches training.
#
# Usage:
#   .\train_clara.ps1                          # identity autoencoder
#   .\train_clara.ps1 -Phase saturation         # saturation
#   .\train_clara.ps1 -Phase identity -Fresh    # fresh best-val baseline
#   .\train_clara.ps1 -NoAmp                    # CPU training

param(
    [string]$Phase = "identity",
    [string]$Device = "cuda",
    [switch]$Fresh,
    [switch]$NoAmp,
    [string]$LogFile = "train_clara.log"
)

$env:HSA_ENABLE_SDMA = "0"
$env:PYTORCH_ALLOC_CONF = "garbage_collection_threshold:0.8,max_split_size_mb:256"
$env:MIOPEN_LOG_LEVEL = "3"
$env:MIOPEN_FIND_MODE = "2"
$env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"

$args = @(
    "train_clara_launcher.py",
    "--phase", $Phase,
    "--device", $Device
)

if ($Fresh) { $args += "--fresh-best-val" }
if ($NoAmp) { $args += "--no-amp" }

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " CLARA Training — Phase: $Phase" -ForegroundColor Cyan
Write-Host " Device: $Device" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python @args 2>&1 | Tee-Object -FilePath $LogFile
