<#
.SYNOPSIS
    Install script for the agentY project.
    Works on Windows PowerShell 5.1+ and PowerShell 7+, and macOS/Linux with PowerShell 7+.

    The UI is the "agentY" tab inside ComfyUI (the agentY-comfyuiConnect custom
    node). There is no Chainlit, Docker, Postgres, or MinIO anymore — conversations
    persist to a local SQLite file.
#>

Set-StrictMode -Version 3.0
$ErrorActionPreference = "Stop"

$Script:OnWindows = $true
if (Get-Variable -Name IsWindows -Scope Global -ErrorAction SilentlyContinue) {
    $Script:OnWindows = [bool]$IsWindows
}

function Write-Header  { param([string]$Text) Write-Host ""; Write-Host "---  $Text  ---" -ForegroundColor Cyan }
function Write-Success { param([string]$Text) Write-Host "  [ok] $Text" -ForegroundColor Green }
function Write-Info    { param([string]$Text) Write-Host "  [i]  $Text" -ForegroundColor Yellow }
function Write-Fail    { param([string]$Text) Write-Host "  [!]  $Text" -ForegroundColor Red }
function Exit-WithError { param([string]$Message, [int]$Code = 1) Write-Fail $Message; exit $Code }

$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = (Get-Location).Path }

# ---------------------------------------------------------------------------
# 1. Preflight
# ---------------------------------------------------------------------------
Write-Header "1 / 4  Preflight checks"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Exit-WithError "'uv' is not installed or not on PATH. Install it from https://docs.astral.sh/uv/getting-started/installation/"
}
Write-Success "uv found: $(uv --version)"

# ---------------------------------------------------------------------------
# 2. Virtual environment
# ---------------------------------------------------------------------------
Write-Header "2 / 4  Virtual environment"

$Script:VenvDir = Join-Path $ProjectRoot ".venv"
if ($Script:OnWindows) {
    $Script:VenvPython     = Join-Path $Script:VenvDir "Scripts\python.exe"
    $Script:ActivateScript = Join-Path $Script:VenvDir "Scripts\Activate.ps1"
} else {
    $Script:VenvPython     = Join-Path $Script:VenvDir "bin/python"
    $Script:ActivateScript = Join-Path $Script:VenvDir "bin/Activate.ps1"
}

$venvLooksValid = (Test-Path $Script:VenvDir) -and (Test-Path $Script:VenvPython)
if (-not $venvLooksValid) {
    if (Test-Path $Script:VenvDir) {
        Write-Info ".venv exists but appears incomplete - recreating"
        Remove-Item -Recurse -Force $Script:VenvDir
    }
    Write-Info "Creating .venv with uv ..."
    Push-Location $ProjectRoot
    try {
        uv venv .venv
        if ($LASTEXITCODE -ne 0) { Exit-WithError "uv venv failed." }
    } finally { Pop-Location }
    Write-Success ".venv created"
} else {
    Write-Info ".venv already exists - skipping creation"
}
& $Script:ActivateScript
Write-Success ".venv activated"

# ---------------------------------------------------------------------------
# 3. Python dependencies
# ---------------------------------------------------------------------------
Write-Header "3 / 4  Python dependencies"

Push-Location $ProjectRoot
try {
    $RequirementsFile = Join-Path $ProjectRoot "requirements.txt"
    if (-not (Test-Path $RequirementsFile)) { Exit-WithError "requirements.txt not found at $RequirementsFile." }
    Write-Info "Installing requirements.txt ..."
    uv pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { Exit-WithError "uv pip install -r requirements.txt failed." }
} finally { Pop-Location }
Write-Success "Python dependencies installed"

# ---------------------------------------------------------------------------
# 4. .env setup
# ---------------------------------------------------------------------------
Write-Header "4 / 4  .env setup"

$EnvFile    = Join-Path $ProjectRoot ".env"
$EnvExample = Join-Path $ProjectRoot ".env_example"
if (-not (Test-Path $EnvFile)) {
    if (-not (Test-Path $EnvExample)) { Exit-WithError ".env_example not found. Cannot create .env." }
    Copy-Item $EnvExample $EnvFile
    Write-Info "Copied .env_example -> .env"
} else {
    Write-Info ".env already exists - skipping copy"
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Write-Header "Setup complete"
Write-Host ""
Write-Host "  agentY is ready. Next steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. Fill in your API keys in .env (HF_TOKEN, ANTHROPIC_API_KEY, COMFYUI_API_KEY)." -ForegroundColor Yellow
Write-Host ""
Write-Host "  2. Install the ComfyUI chat UI (once) - separate repo:" -ForegroundColor Yellow
Write-Host "       git clone https://github.com/szprivate/agentY-comfyuiConnect  <ComfyUI>\custom_nodes\agentY-comfyuiConnect" -ForegroundColor White
Write-Host "     then restart ComfyUI." -ForegroundColor White
Write-Host ""
Write-Host "  3. Start the agent chat host:" -ForegroundColor Yellow
Write-Host "       ./run_agent.ps1" -ForegroundColor White
Write-Host ""
Write-Host "  4. Open ComfyUI and click the agentY tab in the left sidebar." -ForegroundColor Yellow
Write-Host ""
