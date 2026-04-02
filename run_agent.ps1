# run_agent.ps1 - Launch agentY
#
# Pipeline mode (default) — Researcher resolves the spec, Brain executes it:
#   .\run_agent.ps1
#   .\run_agent.ps1 -ResearcherOllamaModel qwen3-coder:32b
#   .\run_agent.ps1 -BrainAnthropicModel claude-sonnet-4-5
#   .\run_agent.ps1 -SkipBrain
#
# Single-agent mode (legacy):
#   .\run_agent.ps1 -Mode single
#   .\run_agent.ps1 -Mode single -OllamaModel llama3.2

param(
    [switch]$Help,

    # Mode
    [ValidateSet("pipeline","single","")]
    [string]$Mode = "",

    # Pipeline – Researcher
    [ValidateSet("ollama","claude","")]
    [string]$ResearcherLlm = "",
    [string]$ResearcherOllamaModel = "",
    [string]$ResearcherAnthropicModel = "",

    # Pipeline – Brain
    [ValidateSet("claude","ollama","")]
    [string]$BrainLlm = "",
    [string]$BrainAnthropicModel = "",
    [string]$BrainOllamaModel = "",
    [switch]$SkipBrain,

    # Single-agent (legacy)
    [ValidateSet("claude","ollama","")]
    [string]$Llm = "",
    [string]$OllamaModel = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Help) {
    Write-Host ""
    Write-Host "Usage: .\run_agent.ps1 [OPTIONS]" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Mode:"
    Write-Host "  -Mode <pipeline|single>          Execution mode (default: pipeline)."
    Write-Host ""
    Write-Host "Pipeline mode options (default):"
    Write-Host "  -ResearcherLlm <ollama|claude>   LLM for the Researcher (default: ollama)."
    Write-Host "  -ResearcherOllamaModel <model>   Ollama model for the Researcher."
    Write-Host "  -ResearcherAnthropicModel <id>   Anthropic model for the Researcher."
    Write-Host "  -BrainLlm <claude|ollama>        LLM for the Brain (default: claude)."
    Write-Host "  -BrainAnthropicModel <id>        Anthropic model for the Brain."
    Write-Host "  -BrainOllamaModel <model>        Ollama model for the Brain."
    Write-Host "  -SkipBrain                      Pipeline mode only: return Researcher output and skip Brain stage."
    Write-Host ""
    Write-Host "Single-agent mode options (-Mode single):"
    Write-Host "  -Llm <claude|ollama>             LLM backend."
    Write-Host "  -OllamaModel <model>             Ollama model name (implies -Llm ollama)."
    Write-Host ""
    Write-Host "  -Help                            Show this help message and exit."
    Write-Host ""
    Write-Host "Environment variables (all overridable in .env or shell):"
    Write-Host "  AGENT_MODE                       pipeline | single"
    Write-Host "  RESEARCHER_LLM                   ollama | claude"
    Write-Host "  RESEARCHER_OLLAMA_MODEL          e.g. qwen3-coder:32b"
    Write-Host "  RESEARCHER_ANTHROPIC_MODEL       e.g. claude-haiku-4-5"
    Write-Host "  BRAIN_LLM                        claude | ollama"
    Write-Host "  BRAIN_ANTHROPIC_MODEL            e.g. claude-sonnet-4-5"
    Write-Host "  BRAIN_OLLAMA_MODEL               e.g. qwen3-vl:30b"
    Write-Host "  ANTHROPIC_API_KEY                Anthropic API key"
    Write-Host "  OLLAMA_HOST                      Ollama server URL (default: http://localhost:11434)"
    Write-Host "  API_KEY_COMFY_ORG                ComfyUI API key"
    Write-Host "  HF_TOKEN                         Hugging Face token for gated models"
    Write-Host "  SLACK_BOT_TOKEN                  Slack bot token (enables Slack integration)"
    Write-Host "  SLACK_MEMBER_ID                  Slack member ID"
    Write-Host ""
    Write-Host "Settings (config/settings.json - overridden by env vars):"
    Write-Host "  llm.agent_mode, llm.pipeline.*, llm.ollama.host, llm.anthropic.*"
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

    if ($Mode -ne "") {
        $PythonArgs += "--mode", $Mode
    }

    # Pipeline args
    if ($ResearcherLlm -ne "")          { $PythonArgs += "--researcher-llm", $ResearcherLlm }
    if ($ResearcherOllamaModel -ne "")  { $PythonArgs += "--researcher-ollama-model", $ResearcherOllamaModel }
    if ($ResearcherAnthropicModel -ne ""){ $PythonArgs += "--researcher-anthropic-model", $ResearcherAnthropicModel }
    if ($BrainLlm -ne "")              { $PythonArgs += "--brain-llm", $BrainLlm }
    if ($BrainAnthropicModel -ne "")   { $PythonArgs += "--brain-anthropic-model", $BrainAnthropicModel }
    if ($BrainOllamaModel -ne "")      { $PythonArgs += "--brain-ollama-model", $BrainOllamaModel }
    if ($SkipBrain)                       { $PythonArgs += "--skip-brain" }

    # Single-agent args
    if ($OllamaModel -ne "") {
        $PythonArgs += "--mode", "single", "--ollama-model", $OllamaModel
    } elseif ($Llm -ne "") {
        $PythonArgs += "--mode", "single", "--llm", $Llm
    }

    Write-Host ""
    Write-Host "Starting agentY..." -ForegroundColor Cyan
    python -m src.main @PythonArgs
}
finally {
    Pop-Location
}

