# 评估微调后的 LoRA（默认 test 集前 10 条，约 1–2 分钟）
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_ENDPOINT = "https://hf-mirror.com"

.\.venv\Scripts\python.exe eval_model.py @args
