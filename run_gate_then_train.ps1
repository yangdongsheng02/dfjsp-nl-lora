# Overfit gate (HF eval, aligned base) → full LoRA train if gate passes.
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:TOKENIZERS_PARALLELISM = "false"
$env:UNSLOTH_COMPILE_DISABLE = "1"
$env:TORCH_COMPILE_DISABLE = "1"
$py = ".\.venv\Scripts\python.exe"

Write-Host "=== overfit gate (HF eval) ===" -ForegroundColor Cyan
& $py overfit_sanity.py 2>&1 | Tee-Object -FilePath overfit_sanity.log
if ($LASTEXITCODE -ne 0) {
    Write-Host "Gate FAILED — full train not started." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n=== full LoRA train (500 steps, Unsloth) ===" -ForegroundColor Cyan
& $py train_unsloth.py 2>&1 | Tee-Object -FilePath train_nl.log
exit $LASTEXITCODE
