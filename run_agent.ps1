# run_agent.ps1 - Launch agentY
#
# Pipeline mode (default) — Researcher resolves the spec, Brain executes it:
#   .\run_agent.ps1
#   .\run_agent.ps1 -LlmResearcher "ollama,qwen3-coder:32b"
#   .\run_agent.ps1 -LlmBrain "claude,claude-sonnet-4-5"
#   .\run_agent.ps1 -SkipBrain

param(
    [switch]$Help,

    # Pipeline – Researcher  e.g. -LlmResearcher "ollama,qwen3:9b"  or  -LlmResearcher "claude,claude-haiku-4-5"
    [string]$LlmResearcher = "",

    # Pipeline – Brain  e.g. -LlmBrain "claude,claude-sonnet-4-5"  or  -LlmBrain "ollama,qwen3-vl:30b"
    [string]$LlmBrain = "",

    [switch]$SkipBrain
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Help) {
    Write-Host ""
    Write-Host "Usage: .\run_agent.ps1 [OPTIONS]" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -LlmResearcher `"provider,model`"  LLM for the Researcher stage."
    Write-Host "                                    e.g.  .\run_agent.ps1 -LlmResearcher `"ollama,qwen3-coder:32b`""
    Write-Host "                                          .\run_agent.ps1 -LlmResearcher `"claude,claude-haiku-4-5`""
    Write-Host "  -LlmBrain `"provider,model`"       LLM for the Brain stage."
    Write-Host "                                    e.g.  .\run_agent.ps1 -LlmBrain `"claude,claude-sonnet-4-5`""
    Write-Host "                                          .\run_agent.ps1 -LlmBrain `"ollama,qwen3-vl:30b`""
    Write-Host "  -SkipBrain                       Return Researcher output and skip the Brain stage."
    Write-Host "  -Help                            Show this help message and exit."
    Write-Host ""
    Write-Host "Environment variables (.env file in project root):"
    Write-Host "  ANTHROPIC_API_KEY                Anthropic API key"
    Write-Host "  OLLAMA_HOST                      Ollama server URL (default: http://localhost:11434)"
    Write-Host "  COMFYUI_API_KEY                  ComfyUI API key"
    Write-Host "  HF_TOKEN                         Hugging Face token for gated models"
    Write-Host "  SLACK_BOT_TOKEN                  Slack bot token (enables Slack integration)"
    Write-Host "  SLACK_MEMBER_ID                  Slack member ID"
    Write-Host ""
    Write-Host "Defaults (config/settings.json):"
    Write-Host "  Default LLMs and models are read from settings.json and can be overridden"
    Write-Host "  by -LlmResearcher / -LlmBrain flags or environment variables."
    Write-Host ""
    exit 0
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location $ProjectRoot

try {
    # Activate the virtual environment
    $venvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
    if (-not (Test-Path $venvActivate)) {
        Write-Host "[run_agent] .venv not found - creating virtual environment..." -ForegroundColor Yellow
        python -m venv .venv
        & $venvActivate
        Write-Host "[run_agent] Installing dependencies..." -ForegroundColor Yellow
        pip install -r requirements.txt
    } else {
        & $venvActivate
    }

    # Build argument list
    $PythonArgs = @()

    # Parse -LlmResearcher  "provider,modelname"
    if ($LlmResearcher -ne "") {
        $parts = $LlmResearcher -split ",", 2
        $provider = $parts[0].Trim()
        $model    = if ($parts.Count -gt 1) { $parts[1].Trim() } else { "" }
        $PythonArgs += "--researcher-llm", $provider
        if ($model -ne "") {
            switch ($provider) {
                "ollama"  { $PythonArgs += "--researcher-ollama-model",    $model }
                "claude"  { $PythonArgs += "--researcher-anthropic-model", $model }
                default   { $PythonArgs += "--researcher-ollama-model",    $model }
            }
        }
    }

    # Parse -LlmBrain  "provider,modelname"
    if ($LlmBrain -ne "") {
        $parts = $LlmBrain -split ",", 2
        $provider = $parts[0].Trim()
        $model    = if ($parts.Count -gt 1) { $parts[1].Trim() } else { "" }
        $PythonArgs += "--brain-llm", $provider
        if ($model -ne "") {
            switch ($provider) {
                "claude"  { $PythonArgs += "--brain-anthropic-model", $model }
                "ollama"  { $PythonArgs += "--brain-ollama-model",    $model }
                default   { $PythonArgs += "--brain-ollama-model",    $model }
            }
        }
    }

    if ($SkipBrain) { $PythonArgs += "--skip-brain" }

    Write-Host ""
    Write-Host "Starting agentY..." -ForegroundColor Cyan
    python -m src.main @PythonArgs
}
finally {
    Pop-Location
}

