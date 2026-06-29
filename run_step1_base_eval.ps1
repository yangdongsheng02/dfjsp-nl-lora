# Step 1: Base model full-schedule eval (50 test samples)
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:TOKENIZERS_PARALLELISM = "false"

Write-Host "Preflight..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe preflight_schedule.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
.\.venv\Scripts\python.exe validate_pipeline.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`nBase model schedule eval (50 samples, ~3-4 hours)..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe eval_schedule.py --mode base --max-samples 0 --show 3 2>&1 |
    Tee-Object -FilePath eval_schedule_base.log
