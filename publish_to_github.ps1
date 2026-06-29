# Create public GitHub repo and push (requires: gh auth login)
param(
    [string]$RepoName = "dfjsp-nl-lora",
    [ValidateSet("public", "private")]
    [string]$Visibility = "public"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$gh = Get-Command gh -ErrorAction SilentlyContinue
if (-not $gh) {
    Write-Error "GitHub CLI (gh) not found. Install: winget install GitHub.cli"
}

gh auth status 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Not logged in. Run: gh auth login" -ForegroundColor Yellow
    exit 1
}

$desc = "DFJSP natural-language end-to-end scheduling with LoRA on 4GB GPU (negative result, reproducible pipeline)"

if (git remote get-url origin 2>$null) {
    Write-Host "Remote 'origin' already exists. Pushing ..."
    git push -u origin HEAD
    gh repo view --web
    exit $LASTEXITCODE
}

gh repo create $RepoName --$Visibility --source=. --remote=origin --description $desc --push
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nDone. Open repo:" -ForegroundColor Green
    gh repo view --web
}
