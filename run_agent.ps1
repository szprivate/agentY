# run_agent.ps1 - Launch agentY
# Usage: .\run_agent.ps1 [-OllamaModel <model>]
#   -OllamaModel  Ollama model name to use (implies --llm ollama).
#                 Example: .\run_agent.ps1 -OllamaModel llama3.2

param(
    [switch]$Help,
    [string]$OllamaModel = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Help) {
    Write-Host ""
    Write-Host "Usage: .\run_agent.ps1 [OPTIONS]" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -OllamaModel <model>   Ollama model name to use (implies --llm ollama)."
    Write-Host "                         Example: .\run_agent.ps1 -OllamaModel llama3.2"
    Write-Host "                         Overrides the OLLAMA_MODEL env var."
    Write-Host "  -Help                  Show this help message and exit."
    Write-Host ""
    Write-Host "Environment variables:"
    Write-Host "  AGENT_LLM              LLM backend: 'claude' (default) or 'ollama'."
    Write-Host "  OLLAMA_MODEL           Default Ollama model (e.g. qwen3-vl:30b)."
    Write-Host "  OLLAMA_HOST            Ollama server URL (default: http://localhost:11434)."
    Write-Host "  ANTHROPIC_MODEL        Claude model ID (default: claude-haiku-4-5)."
    Write-Host "  API_KEY_COMFY_ORG      ComfyUI API key."
    Write-Host "  HF_TOKEN               Hugging Face token for gated models."
    Write-Host "  SLACK_BOT_TOKEN        Slack bot token (enables Slack integration)."
    Write-Host "  SLACK_MEMBER_ID        Slack member ID (enables Slack integration)."
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

    # Launch the agent
    Write-Host ""
    Write-Host "Starting agentY..." -ForegroundColor Cyan
    $PythonArgs = @()
    if ($OllamaModel -ne "") {
        Write-Host "[run_agent] Using Ollama model: $OllamaModel" -ForegroundColor Cyan
        $PythonArgs += "--llm", "ollama", "--ollama-model", $OllamaModel
    }
    python -m src.main @PythonArgs
}
finally {
    Pop-Location
}
