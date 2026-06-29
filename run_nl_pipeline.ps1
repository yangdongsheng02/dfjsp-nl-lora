param(
    [ValidateSet("1", "2", "3", "all")]
    [string]$Step = "1"
)

Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:TOKENIZERS_PARALLELISM = "false"
$env:UNSLOTH_COMPILE_DISABLE = "1"
$env:TORCH_COMPILE_DISABLE = "1"
$py = ".\.venv\Scripts\python.exe"

function Run-Py([string[]]$PyArgs, [string]$Log) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    if (Test-Path $Log) { Remove-Item $Log -Force }
    $errLog = "$Log.err"
    if (Test-Path $errLog) { Remove-Item $errLog -Force }
    $argStr = ($PyArgs | ForEach-Object {
        if ($_ -match '\s') { "`"$_`"" } else { $_ }
    }) -join " "
    # Redirect to files so logs update live (ReadToEnd buffers until exit).
    $proc = Start-Process -FilePath $py -ArgumentList $argStr `
        -WorkingDirectory $PSScriptRoot `
        -RedirectStandardOutput $Log -RedirectStandardError $errLog `
        -NoNewWindow -PassThru -Wait
    $utf8 = New-Object System.Text.UTF8Encoding $false
    if (Test-Path $Log) {
        Get-Content $Log -Encoding UTF8 | ForEach-Object { Write-Host $_ }
    }
    if (Test-Path $errLog) {
        $err = [System.IO.File]::ReadAllText($errLog, $utf8)
        if ($err) {
            [System.IO.File]::AppendAllText($Log, $err, $utf8)
            Write-Host $err -NoNewline -ForegroundColor Yellow
        }
        Remove-Item $errLog -Force
    }
    $ErrorActionPreference = $prev
    if ($proc.ExitCode -ne 0) { exit $proc.ExitCode }
}

if ($Step -eq "2" -or $Step -eq "all") {
    Write-Host "=== preflight (train gate) ===" -ForegroundColor Cyan
    Run-Py @("preflight_train.py") "preflight_train.log"
} else {
    Write-Host "=== validate (NL) ===" -ForegroundColor Cyan
    Run-Py @("validate_pipeline.py") "validate_nl.log"
}

if ($Step -eq "2" -or $Step -eq "all") {
    Write-Host "`n=== overfit sanity (1-sample gate) ===" -ForegroundColor Cyan
    Run-Py @("overfit_sanity.py") "overfit_sanity.log"
}

if ($Step -eq "1" -or $Step -eq "all") {
    Write-Host "`n=== Step 1: base NL eval ===" -ForegroundColor Cyan
    Run-Py @("eval_schedule.py", "--mode", "base", "--max-samples", "0", "--show", "3") "eval_nl_base.log"
}

if ($Step -eq "2" -or $Step -eq "all") {
    Write-Host "`n=== Step 2: NL LoRA train ===" -ForegroundColor Cyan
    Run-Py @("train_unsloth.py") "train_nl.log"
}

if ($Step -eq "3" -or $Step -eq "all") {
    Write-Host "`n=== Step 3: base vs LoRA compare ===" -ForegroundColor Cyan
    Run-Py @("eval_schedule.py", "--mode", "compare", "--max-samples", "0", "--show", "3") "eval_nl_compare.log"
}

Write-Host "`nDone." -ForegroundColor Green
