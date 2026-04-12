# run_chainlit.ps1 - Launch agentY with the Chainlit web GUI
#
# Usage:
#   .\run_chainlit.ps1                    # default port 8000
#   .\run_chainlit.ps1 -Port 8080
#   .\run_chainlit.ps1 -Watch             # auto-reload on file changes

param(
    [switch]$Help,
    [int]$Port = 8000,
    [switch]$Watch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Help) {
    Write-Host ""
    Write-Host "Usage: .\run_chainlit.ps1 [OPTIONS]" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Port <number>   Port to listen on (default: 8000)."
    Write-Host "  -Watch           Reload the app automatically when source files change."
    Write-Host "  -Help            Show this help message and exit."
    Write-Host ""
    Write-Host "Configuration (same as console mode):"
    Write-Host "  Models and LLM settings are read from config/settings.json."
    Write-Host "  API keys and secrets are read from .env in the project root."
    Write-Host ""
    Write-Host "Access the GUI at:  http://localhost:<Port>"
    Write-Host ""
    exit 0
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location $ProjectRoot

try {
    # Activate the virtual environment (create it if missing)
    $venvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
    if (-not (Test-Path $venvActivate)) {
        Write-Host "[run_chainlit] .venv not found - creating virtual environment..." -ForegroundColor Yellow
        python -m venv .venv
        & $venvActivate
        Write-Host "[run_chainlit] Installing dependencies..." -ForegroundColor Yellow
        pip install -r requirements.txt
    } else {
        & $venvActivate
    }

    # Ensure chainlit is installed
    python -c "import chainlit" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[run_chainlit] Installing chainlit..." -ForegroundColor Yellow
        pip install "chainlit>=2.0.0"
    }

    # Build the chainlit arguments
    $ChainlitArgs = @("run", "src/chainlit_app.py", "--port", $Port)
    if ($Watch) { $ChainlitArgs += "-w" }

    Write-Host ""
    $guiUrl = "http://localhost:" + $Port
    Write-Host "Starting agentY Chainlit GUI..." -ForegroundColor Cyan
    Write-Host "Open your browser at: $guiUrl" -ForegroundColor Green
    Write-Host ""

    chainlit @ChainlitArgs
}
finally {
    Pop-Location
}
