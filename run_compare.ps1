# 微调前 vs 微调后对比评估（无需重新训练）
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:PYTHONUTF8 = "1"
$env:HF_ENDPOINT = "https://hf-mirror.com"

Write-Host "=== 流水线校验 ===" -ForegroundColor Cyan
.\.venv\Scripts\python.exe validate_pipeline.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== 对比评估（base vs LoRA，test 全量）===" -ForegroundColor Cyan
.\.venv\Scripts\python.exe eval_model.py --mode compare --max-samples 0 --show 2
exit $LASTEXITCODE
