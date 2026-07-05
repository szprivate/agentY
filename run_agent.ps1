# run_agent.ps1 - Launch agentY with the Chainlit web GUI
#
# Usage:
#   .\run_agent.ps1                                          # default port 8000
#   .\run_agent.ps1 -Port 8080
#   .\run_agent.ps1 -Watch                                   # auto-reload on file changes
#   .\run_agent.ps1 -LlmQueryTemplates "ollama,qwen3-coder:32b"
#   .\run_agent.ps1 -LlmAssembleWorkflow "claude,claude-sonnet-4-5"

param(
    [switch]$Help,

    [int]$Port = 8000,
    [switch]$Watch,

    # Enable verbose hang/stall tracing (sets AGENTY_DEBUG=1 → .logs/debug.log)
    [switch]$Debug,

    # Pipeline – QueryTemplates  e.g. -LlmQueryTemplates "ollama,qwen3:9b"  or  -LlmQueryTemplates "claude,claude-haiku-4-5"
    [string]$LlmQueryTemplates = "",

    # Pipeline – AssembleWorkflow  e.g. -LlmAssembleWorkflow "claude,claude-sonnet-4-5"  or  -LlmAssembleWorkflow "ollama,qwen3-vl:30b"
    [string]$LlmAssembleWorkflow = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Help) {
    Write-Host ""
    Write-Host "Usage: .\run_agent.ps1 [OPTIONS]" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Port <number>                   Port to listen on (default: 8000)."
    Write-Host "  -Watch                           Reload the app automatically when source files change."
    Write-Host "  -LlmQueryTemplates `"provider,model`"  LLM for the QueryTemplates stage (sets env vars)."
    Write-Host "                                   e.g.  .\run_agent.ps1 -LlmQueryTemplates `"ollama,qwen3-coder:32b`""
    Write-Host "                                         .\run_agent.ps1 -LlmQueryTemplates `"claude,claude-haiku-4-5`""
    Write-Host "  -LlmAssembleWorkflow `"provider,model`"      LLM for the AssembleWorkflow stage (sets env vars)."
    Write-Host "                                   e.g.  .\run_agent.ps1 -LlmAssembleWorkflow `"claude,claude-sonnet-4-5`""
    Write-Host "                                         .\run_agent.ps1 -LlmAssembleWorkflow `"ollama,qwen3-vl:30b`""
    Write-Host "  -Help                            Show this help message and exit."
    Write-Host ""
    Write-Host "Environment variables (.env file in project root):"
    Write-Host "  ANTHROPIC_API_KEY                Anthropic API key"
    Write-Host "  OLLAMA_HOST                      Ollama server URL (default: http://localhost:11434)"
    Write-Host "  COMFYUI_API_KEY                  ComfyUI API key"
    Write-Host "  HF_TOKEN                         Hugging Face token for gated models"
    Write-Host "  CHAINLIT_USERNAME / CHAINLIT_PASSWORD  Web UI credentials"
    Write-Host ""
    Write-Host "Defaults (config/settings.json):"
    Write-Host "  Default LLMs and models are read from settings.json and can be overridden"
    Write-Host "  by -LlmQueryTemplates / -LlmAssembleWorkflow flags or environment variables."
    Write-Host ""
    Write-Host "Access the GUI at:  http://localhost:<Port>"
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

    # Ensure chainlit is installed
    python -c "import chainlit" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[run_agent] Installing chainlit..." -ForegroundColor Yellow
        pip install "chainlit>=2.0.0"
    }

    # Map -LlmQueryTemplates "provider,model" → env vars consumed by create_pipeline()
    if ($LlmQueryTemplates -ne "") {
        $parts = $LlmQueryTemplates -split ",", 2
        $provider = $parts[0].Trim()
        $model    = if ($parts.Count -gt 1) { $parts[1].Trim() } else { "" }
        $env:QueryTemplates_LLM = $provider
        if ($model -ne "") {
            switch ($provider) {
                "ollama" { $env:QueryTemplates_OLLAMA_MODEL    = $model }
                "claude" { $env:QueryTemplates_ANTHROPIC_MODEL = $model }
                default  { $env:QueryTemplates_OLLAMA_MODEL    = $model }
            }
        }
    }

    # Map -LlmAssembleWorkflow "provider,model" → env vars consumed by create_pipeline()
    if ($LlmAssembleWorkflow -ne "") {
        $parts = $LlmAssembleWorkflow -split ",", 2
        $provider = $parts[0].Trim()
        $model    = if ($parts.Count -gt 1) { $parts[1].Trim() } else { "" }
        $env:AssembleWorkflow_LLM = $provider
        if ($model -ne "") {
            switch ($provider) {
                "claude" { $env:AssembleWorkflow_ANTHROPIC_MODEL = $model }
                "ollama" { $env:AssembleWorkflow_OLLAMA_MODEL    = $model }
                default  { $env:AssembleWorkflow_OLLAMA_MODEL    = $model }
            }
        }
    }

    # ── Ensure Docker Desktop is running ────────────────────────────────────
    $dockerRunning = $false
    try {
        docker info 2>&1 | Out-Null
        $dockerRunning = ($LASTEXITCODE -eq 0)
    } catch {}

    if (-not $dockerRunning) {
        Write-Host "[run_agent] Docker Desktop does not appear to be running. Attempting to start it..." -ForegroundColor Yellow
        $dockerDesktopExe = "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
        if (Test-Path $dockerDesktopExe) {
            Start-Process $dockerDesktopExe
            Write-Host "[run_agent] Waiting for Docker Desktop to become ready (up to 60 s)..." -ForegroundColor Yellow
            $timeout = 60
            $elapsed = 0
            do {
                Start-Sleep -Seconds 3
                $elapsed += 3
                try {
                    docker info 2>&1 | Out-Null
                    $dockerRunning = ($LASTEXITCODE -eq 0)
                } catch {}
            } while (-not $dockerRunning -and $elapsed -lt $timeout)

            if ($dockerRunning) {
                Write-Host "[run_agent] Docker Desktop is ready." -ForegroundColor Green
            } else {
                Write-Host "[run_agent] Docker Desktop did not start within $timeout seconds. Docker-dependent features may not work." -ForegroundColor Red
            }
        } else {
            Write-Host "[run_agent] Docker Desktop executable not found at '$dockerDesktopExe'. Skipping Docker startup." -ForegroundColor Red
        }
    }

    # ── Ensure chainlit-datalayer compose project is running ────────────────
    $dockerAvailable = $null
    try { $dockerAvailable = Get-Command docker -ErrorAction Stop } catch {}

    if ($dockerAvailable) {
        # Read chainlit_datalayer_dir from config/settings.json
        $settingsPath = Join-Path $ProjectRoot "config/settings.json"
        $dlComposeDir = ""
        if (Test-Path $settingsPath) {
            $settingsRaw = Get-Content $settingsPath -Raw
            # Strip whole-line // comments before parsing (matches the Python
            # loader in src/agent.py). A regex-based strip is unsafe here: it
            # corrupts "http://..." URLs and chokes on comments that contain
            # double quotes, which silently breaks the datalayer startup.
            $settingsClean = ($settingsRaw -split "`r?`n" |
                Where-Object { -not $_.TrimStart().StartsWith('//') }) -join "`n"
            try {
                $settings = $settingsClean | ConvertFrom-Json
                $dlComposeDir = $settings.chainlit_datalayer_dir
            } catch {
                Write-Host "[run_agent] WARNING: Could not parse settings.json for chainlit_datalayer_dir." -ForegroundColor Yellow
            }
        }
        if (-not $dlComposeDir) {
            Write-Host "[run_agent] WARNING: chainlit_datalayer_dir not set in config/settings.json - skipping datalayer startup." -ForegroundColor Yellow
        }
        $dlComposeFile = if ($dlComposeDir) { Join-Path $dlComposeDir "compose.yaml" } else { "" }

        if ($dlComposeDir -and -not (Test-Path $dlComposeFile)) {
            Write-Host "[run_agent] WARNING: chainlit-datalayer compose file not found at $dlComposeFile" -ForegroundColor Yellow
        } elseif ($dlComposeDir) {
            # Count running containers that belong to this compose project
            $dlRunning = docker compose -f $dlComposeFile ps --status running --quiet 2>&1
            if ($dlRunning -match '\S') {
                Write-Host "[run_agent] chainlit-datalayer is already running." -ForegroundColor Green
            } else {
                Write-Host "[run_agent] Starting chainlit-datalayer compose project..." -ForegroundColor Cyan
                $prevEAP = $ErrorActionPreference
                $ErrorActionPreference = 'SilentlyContinue'
                docker compose -f $dlComposeFile up -d 2>&1 | ForEach-Object { Write-Host "  $_" }
                $ErrorActionPreference = $prevEAP
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "[run_agent] chainlit-datalayer started successfully." -ForegroundColor Green
                } else {
                    Write-Host "[run_agent] WARNING: Failed to start chainlit-datalayer." -ForegroundColor Yellow
                }
            }
        }
    }

    # ── Start MinIO (docker-compose) ─────────────────────────────────────────
    if ($dockerAvailable) {
        $composeFile = Join-Path $ProjectRoot "docker-compose.yml"
        if (Test-Path $composeFile) {
            Write-Host "[run_agent] Starting MinIO storage service..." -ForegroundColor Cyan
            # Use 'docker compose' (v2 plugin) with fallback to 'docker-compose' (v1 standalone)
            $usePluginCompose = $false
            try {
                docker compose version 2>&1 | Out-Null
                if ($LASTEXITCODE -eq 0) { $usePluginCompose = $true }
            } catch {}

            $prevEAP = $ErrorActionPreference
            $ErrorActionPreference = 'SilentlyContinue'
            if ($usePluginCompose) {
                docker compose -f $composeFile up -d minio createbuckets 2>&1 | ForEach-Object { Write-Host "$_" }
            } else {
                docker-compose -f $composeFile up -d minio createbuckets 2>&1 | ForEach-Object { Write-Host "$_" }
            }
            $ErrorActionPreference = $prevEAP

            if ($LASTEXITCODE -eq 0) {
                Write-Host "[run_agent] MinIO ready  ->  API: http://localhost:9000  Console: http://localhost:9001" -ForegroundColor Green
            } else {
                Write-Host "[run_agent] WARNING: docker-compose returned a non-zero exit code. Continuing anyway..." -ForegroundColor Yellow
            }
        } else {
            Write-Host "[run_agent] docker-compose.yml not found - skipping MinIO startup." -ForegroundColor Yellow
        }
    } else {
        Write-Host "[run_agent] Docker not found - skipping MinIO startup. File uploads will not persist." -ForegroundColor Yellow
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

    # Build chainlit arguments
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

