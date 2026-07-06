# run_agent.ps1 - Launch the agentY headless chat host (ComfyUI sidebar backend)
#
# The UI now lives inside ComfyUI (the "agentY" tab in the left sidebar, provided
# by comfyui_extension/agentY-comfyuiConnect). This script starts the backend the
# sidebar talks to over HTTP/SSE on http://127.0.0.1:<Port>. No Chainlit, Docker,
# Postgres, or MinIO.
#
# Usage:
#   .\run_agent.ps1                                          # backend on port 5000
#   .\run_agent.ps1 -Port 5001
#   .\run_agent.ps1 -LlmQueryTemplates "ollama,qwen3-coder:32b"
#   .\run_agent.ps1 -LlmAssembleWorkflow "claude,claude-sonnet-4-5"

param(
    [switch]$Help,

    [int]$Port = 5000,
    [string]$BindHost = "127.0.0.1",

    # Enable verbose hang/stall tracing (sets AGENTY_DEBUG=1 -> .logs/debug.log)
    [switch]$Debug,

    # Pipeline - QueryTemplates  e.g. -LlmQueryTemplates "ollama,qwen3:9b"  or  "claude,claude-haiku-4-5"
    [string]$LlmQueryTemplates = "",

    # Pipeline - AssembleWorkflow  e.g. -LlmAssembleWorkflow "claude,claude-sonnet-4-5"  or  "ollama,qwen3-vl:30b"
    [string]$LlmAssembleWorkflow = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Help) {
    Write-Host ""
    Write-Host "Usage: .\run_agent.ps1 [OPTIONS]" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Port <number>                   Backend port the ComfyUI sidebar connects to (default: 5000)."
    Write-Host "  -BindHost <addr>                 Bind address (default: 127.0.0.1; use 0.0.0.0 for LAN)."
    Write-Host "  -LlmQueryTemplates `"provider,model`"  LLM for the QueryTemplates stage (sets env vars)."
    Write-Host "  -LlmAssembleWorkflow `"provider,model`"  LLM for the AssembleWorkflow stage (sets env vars)."
    Write-Host "  -Debug                           Enable hang/stall tracing to .logs/debug.log."
    Write-Host "  -Help                            Show this help message and exit."
    Write-Host ""
    Write-Host "The chat UI is the agentY tab in ComfyUI's left sidebar. Install once:"
    Write-Host "  copy comfyui_extension\agentY-comfyuiConnect into <ComfyUI>\custom_nodes\ and restart ComfyUI."
    Write-Host ""
    exit 0
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location $ProjectRoot

if ($Debug) {
    $env:AGENTY_DEBUG = "1"
    Write-Host "[run_agent] AGENTY_DEBUG enabled - tracing hangs/stalls to .logs/debug.log" -ForegroundColor Yellow
}

try {
    # Activate the virtual environment (create it if missing)
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

    # Map -LlmQueryTemplates "provider,model" -> env vars consumed by create_pipeline()
    if ($LlmQueryTemplates -ne "") {
        $parts = $LlmQueryTemplates -split ",", 2
        $provider = $parts[0].Trim()
        $model    = if ($parts.Count -gt 1) { $parts[1].Trim() } else { "" }
        $env:QUERYTEMPLATES_LLM = $provider
        if ($model -ne "") {
            switch ($provider) {
                "ollama" { $env:QUERYTEMPLATES_OLLAMA_MODEL    = $model }
                "claude" { $env:QUERYTEMPLATES_ANTHROPIC_MODEL = $model }
                default  { $env:QUERYTEMPLATES_OLLAMA_MODEL    = $model }
            }
        }
    }

    # Map -LlmAssembleWorkflow "provider,model" -> env vars consumed by create_pipeline()
    if ($LlmAssembleWorkflow -ne "") {
        $parts = $LlmAssembleWorkflow -split ",", 2
        $provider = $parts[0].Trim()
        $model    = if ($parts.Count -gt 1) { $parts[1].Trim() } else { "" }
        $env:ASSEMBLEWORKFLOW_LLM = $provider
        if ($model -ne "") {
            switch ($provider) {
                "claude" { $env:ASSEMBLEWORKFLOW_ANTHROPIC_MODEL = $model }
                "ollama" { $env:ASSEMBLEWORKFLOW_OLLAMA_MODEL    = $model }
                default  { $env:ASSEMBLEWORKFLOW_OLLAMA_MODEL    = $model }
            }
        }
    }

    # ── Refresh ComfyUI model caches ────────────────────────────────────────
    Write-Host "[run_agent] Refreshing ComfyUI model cache..." -ForegroundColor Cyan
    $env:COMFYUI_MODELS_REFRESHED = ""   # clear any leftover value from a previous run in this session
    python scripts/refresh_models.py
    $env:COMFYUI_MODELS_REFRESHED = "1"  # prevent re-runs in child processes
    Write-Host ""

    # ── Check and install missing custom nodes ───────────────────────────────
    Write-Host "[run_agent] Checking for missing custom node dependencies..." -ForegroundColor Cyan
    python scripts/check_missing_custom_nodes.py
    Write-Host ""

    Write-Host ""
    Write-Host "Starting agentY chat host on http://${BindHost}:$Port ..." -ForegroundColor Cyan
    Write-Host "Open ComfyUI and click the agentY tab in the left sidebar." -ForegroundColor Green
    Write-Host ""

    python -m src.agenty_ui_server --host $BindHost --port $Port
}
finally {
    Pop-Location
}
