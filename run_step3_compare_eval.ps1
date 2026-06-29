# Step 3: Compare base vs LoRA on schedule task
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:TOKENIZERS_PARALLELISM = "false"

Write-Host "Compare eval (base vs LoRA, 50 samples)..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe eval_schedule.py --mode compare --max-samples 0 --show 3 2>&1 |
    Tee-Object -FilePath eval_schedule_compare.log
