# run_agent.ps1 - Launch agentY with the Chainlit web GUI
#
# Usage:
#   .\run_agent.ps1                                          # default port 8000
#   .\run_agent.ps1 -Port 8080
#   .\run_agent.ps1 -Watch                                   # auto-reload on file changes
#   .\run_agent.ps1 -LlmResearcher "ollama,qwen3-coder:32b"
#   .\run_agent.ps1 -LlmBrain "claude,claude-sonnet-4-5"

param(
    [switch]$Help,

    [int]$Port = 8000,
    [switch]$Watch,

    # Pipeline – Researcher  e.g. -LlmResearcher "ollama,qwen3:9b"  or  -LlmResearcher "claude,claude-haiku-4-5"
    [string]$LlmResearcher = "",

    # Pipeline – Brain  e.g. -LlmBrain "claude,claude-sonnet-4-5"  or  -LlmBrain "ollama,qwen3-vl:30b"
    [string]$LlmBrain = ""
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
    Write-Host "  -LlmResearcher `"provider,model`"  LLM for the Researcher stage (sets env vars)."
    Write-Host "                                   e.g.  .\run_agent.ps1 -LlmResearcher `"ollama,qwen3-coder:32b`""
    Write-Host "                                         .\run_agent.ps1 -LlmResearcher `"claude,claude-haiku-4-5`""
    Write-Host "  -LlmBrain `"provider,model`"      LLM for the Brain stage (sets env vars)."
    Write-Host "                                   e.g.  .\run_agent.ps1 -LlmBrain `"claude,claude-sonnet-4-5`""
    Write-Host "                                         .\run_agent.ps1 -LlmBrain `"ollama,qwen3-vl:30b`""
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
    Write-Host "  by -LlmResearcher / -LlmBrain flags or environment variables."
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

    # Map -LlmResearcher "provider,model" → env vars consumed by create_pipeline()
    if ($LlmResearcher -ne "") {
        $parts = $LlmResearcher -split ",", 2
        $provider = $parts[0].Trim()
        $model    = if ($parts.Count -gt 1) { $parts[1].Trim() } else { "" }
        $env:RESEARCHER_LLM = $provider
        if ($model -ne "") {
            switch ($provider) {
                "ollama" { $env:RESEARCHER_OLLAMA_MODEL    = $model }
                "claude" { $env:RESEARCHER_ANTHROPIC_MODEL = $model }
                default  { $env:RESEARCHER_OLLAMA_MODEL    = $model }
            }
        }
    }

    # Map -LlmBrain "provider,model" → env vars consumed by create_pipeline()
    if ($LlmBrain -ne "") {
        $parts = $LlmBrain -split ",", 2
        $provider = $parts[0].Trim()
        $model    = if ($parts.Count -gt 1) { $parts[1].Trim() } else { "" }
        $env:BRAIN_LLM = $provider
        if ($model -ne "") {
            switch ($provider) {
                "claude" { $env:BRAIN_ANTHROPIC_MODEL = $model }
                "ollama" { $env:BRAIN_OLLAMA_MODEL    = $model }
                default  { $env:BRAIN_OLLAMA_MODEL    = $model }
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
            # Strip single-line comments (// ...) before parsing
            $settingsClean = $settingsRaw -replace '(?m)//[^"\r\n]*', ''
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

