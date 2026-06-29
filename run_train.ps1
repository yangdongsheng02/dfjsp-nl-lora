# 启动训练（Windows 需 UTF-8 模式）
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# 国内访问 HuggingFace 慢时可取消下一行注释
$env:HF_ENDPOINT = "https://hf-mirror.com"

.\.venv\Scripts\python.exe train_unsloth.py
