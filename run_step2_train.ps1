# Step 2: LoRA fine-tune on full schedule labels
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:TOKENIZERS_PARALLELISM = "false"

Write-Host "Training full-schedule LoRA..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe train_unsloth.py 2>&1 | Tee-Object -FilePath train_schedule.log
