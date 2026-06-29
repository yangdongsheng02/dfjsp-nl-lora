# Full schedule pipeline: validate -> base eval -> train -> compare eval
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:TOKENIZERS_PARALLELISM = "false"
$py = ".\.venv\Scripts\python.exe"

function Run-Py([string]$Args, [string]$Log) {
    & $py @Args 2>&1 | Tee-Object -FilePath $Log
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "=== 1/4 validate_pipeline ===" -ForegroundColor Cyan
Run-Py @("validate_pipeline.py") "validate_schedule.log"

Write-Host "`n=== 2/4 base model schedule eval (50 test) ===" -ForegroundColor Cyan
Run-Py @("eval_schedule.py", "--mode", "base", "--max-samples", "0", "--show", "3") "eval_schedule_base.log"

Write-Host "`n=== 3/4 LoRA training (full schedule) ===" -ForegroundColor Cyan
Run-Py @("train_unsloth.py") "train_schedule.log"

Write-Host "`n=== 4/4 base vs LoRA compare eval ===" -ForegroundColor Cyan
Run-Py @("eval_schedule.py", "--mode", "compare", "--max-samples", "0", "--show", "3") "eval_schedule_compare.log"

Write-Host "`nDone. Logs: eval_schedule_base.log, train_schedule.log, eval_schedule_compare.log" -ForegroundColor Green
