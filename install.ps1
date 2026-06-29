# scheduling_research — 一键安装（Windows PowerShell）
# 用法: .\install.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== Step 1/6: PyTorch CUDA 11.8 ===" -ForegroundColor Cyan
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 `
    --index-url https://download.pytorch.org/whl/cu118

Write-Host "=== Step 2/6: ML 依赖 ===" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "=== Step 3/6: xformers（--no-deps）===" -ForegroundColor Cyan
pip install xformers==0.0.27.post2 `
    --index-url https://download.pytorch.org/whl/cu118 --no-deps --force-reinstall

Write-Host "=== Step 4/6: accelerate（--no-deps，避免覆盖 torch）===" -ForegroundColor Cyan
pip install accelerate==1.1.1 --no-deps --force-reinstall

Write-Host "=== Step 5/6: unsloth 2025.6.8（--no-deps）===" -ForegroundColor Cyan
pip install --no-deps -r requirements-unsloth.txt

Write-Host "=== Step 6/6: 验证 ===" -ForegroundColor Cyan
python check_env.py

Write-Host "`n训练请运行: .\run_train.ps1" -ForegroundColor Green
