param()

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvActivate = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $venvActivate)) {
    Write-Error "Virtual environment not found at $venvActivate. Create it with: python -m venv .venv"
    exit 1
}

. $venvActivate
Write-Host "Activated docs-chatbot virtual environment." -ForegroundColor Green
