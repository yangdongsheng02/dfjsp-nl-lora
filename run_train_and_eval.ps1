# 一键：训练 LoRA（约 25 分钟）→ 微调前/后对比评估
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_ENDPOINT = "https://hf-mirror.com"

Write-Host "=== 1/4 流水线校验（必须通过才训练）===" -ForegroundColor Cyan
.\.venv\Scripts\python.exe validate_pipeline.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== 2/4 训练 LoRA（200 step）===" -ForegroundColor Cyan
.\.venv\Scripts\python.exe train_unsloth.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== 3/4 生成冒烟测试 ===" -ForegroundColor Cyan
.\.venv\Scripts\python.exe debug_gen.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== 4/4 评估：微调前 vs 微调后（test 全量）===" -ForegroundColor Cyan
.\.venv\Scripts\python.exe eval_model.py --mode compare --max-samples 0 --show 1
exit $LASTEXITCODE
