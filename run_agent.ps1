# run_agent.ps1 - Launch the agentY headless chat host (ComfyUI sidebar backend)
#
# The UI now lives inside ComfyUI (the "agentY" tab in the left sidebar, provided
# by the separate agentY-comfyuiConnect repo). This script starts the backend the
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

    # Skip the startup update check (also: set AGENTY_NO_UPDATE=1, or
    # auto_update = false in config/settings.local.json)
    [switch]$NoUpdate,

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
    Write-Host "  -NoUpdate                        Skip the startup check for updates on the remote."
    Write-Host "  -Help                            Show this help message and exit."
    Write-Host ""
    Write-Host "The chat UI is the agentY tab in ComfyUI's left sidebar (separate repo). Install once:"
    Write-Host "  git clone https://github.com/szprivate/agentY-comfyuiConnect into <ComfyUI>\custom_nodes\ and restart ComfyUI."
    Write-Host ""
    exit 0
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location $ProjectRoot

if ($Debug) {
    $env:AGENTY_DEBUG = "1"
    Write-Host "[run_agent] AGENTY_DEBUG enabled - tracing hangs/stalls to .logs/debug.log" -ForegroundColor Yellow
}

# ── Startup update check ────────────────────────────────────────────────────
# Fast-forwards the repos that make up the RUNNING agent (this one, plus the
# agenty_core tool layer it installs editable) to whatever the remote has.
#
# Deliberately conservative, because this runs unattended on every start and the
# working copy is the user's:
#   * local commits that aren't pushed are left alone — never rebase, never reset;
#   * --ff-only, so a diverged branch reports and stops rather than merging;
#   * a fetch failure (offline, VPN, remote down) is a shrug, not a failed start.
# Anything it declines to do it says out loud, so a stale checkout is never silent.
#
# A dirty working tree no longer blocks the update. It used to, and that meant the
# check never ran in practice: the agent writes generated artifacts into its own
# checkouts (saved templates, a regenerated recipe DB), so a repo is nearly always
# dirty. Instead we compare what the incoming commits TOUCH against what is locally
# modified — only that intersection is a real conflict, and git happily fast-forwards
# around everything else. See Get-DirtyPaths / Update-Repo below.

# Locally modified tracked files + untracked files, as plain repo-relative paths.
# core.quotepath=false stops git escaping non-ASCII names into "\303\244" forms,
# which would never match the incoming list.
function Get-DirtyPaths {
    param([string]$Dir)
    $out = @()
    $tracked = & git -c core.quotepath=false -C $Dir diff --name-only HEAD 2>$null
    if ($tracked) { $out += $tracked }
    $untracked = & git -c core.quotepath=false -C $Dir ls-files --others --exclude-standard 2>$null
    if ($untracked) { $out += $untracked }
    return @($out | Where-Object { $_ } | Select-Object -Unique)
}

function Update-Repo {
    param([string]$Name, [string]$Dir)

    if (-not (Test-Path (Join-Path $Dir ".git"))) { return $null }

    $upstream = & git -C $Dir rev-parse --abbrev-ref --symbolic-full-name "@{u}" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $upstream) {
        Write-Host "[update] $Name has no upstream branch - skipping." -ForegroundColor DarkYellow
        return $null
    }

    & git -C $Dir fetch --quiet 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[update] $Name - could not reach the remote; continuing with the local copy." -ForegroundColor DarkYellow
        return $null
    }

    $behind = (& git -C $Dir rev-list --count "HEAD..@{u}" 2>$null)
    $ahead  = (& git -C $Dir rev-list --count "@{u}..HEAD" 2>$null)
    if ($LASTEXITCODE -ne 0) { return $null }
    $behind = [int]$behind; $ahead = [int]$ahead

    if ($behind -eq 0) {
        Write-Host "[update] $Name is up to date." -ForegroundColor DarkGray
        return $null
    }
    if ($ahead -gt 0) {
        Write-Host "[update] $Name has diverged ($ahead local, $behind remote) - skipping. Merge it yourself." -ForegroundColor Yellow
        return $null
    }

    $before = (& git -C $Dir rev-parse HEAD 2>$null)

    # Which local changes actually stand in the way? Only files the incoming
    # commits modify. Everything else survives the fast-forward untouched, so a
    # repo full of generated artifacts still updates.
    $incoming = & git -c core.quotepath=false -C $Dir diff --name-only "HEAD..@{u}" 2>$null
    # @(...) around the call, not just inside the function: PowerShell unrolls a
    # returned array into the pipeline, so one dirty file arrives as a bare string
    # and none arrives as $null — and under Set-StrictMode neither has a .Count.
    $dirty = @(Get-DirtyPaths -Dir $Dir)
    $collisions = @()
    if ($dirty.Count -gt 0 -and $incoming) {
        $inc = @{}
        foreach ($f in $incoming) { $inc[$f] = $true }
        $collisions = @($dirty | Where-Object { $inc.ContainsKey($_) })
    }
    if ($dirty.Count -gt 0 -and $collisions.Count -eq 0) {
        Write-Host "[update] $Name has $($dirty.Count) local change(s), none of them touched by this update - keeping them." -ForegroundColor DarkGray
    }

    $stashed = $false
    if ($collisions.Count -gt 0) {
        # Park ONLY the colliding paths. Nothing is discarded: a stash entry is a
        # commit, recoverable with `git stash pop` long after this run.
        Write-Host "[update] $Name - $($collisions.Count) local file(s) also changed upstream; parking them in a stash:" -ForegroundColor Yellow
        foreach ($c in $collisions) { Write-Host "           $c" -ForegroundColor DarkYellow }
        $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        & git -C $Dir stash push --include-untracked -m "agentY auto-update $stamp" -- $collisions | Out-Host
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[update] $Name - could not stash; leaving the repo untouched." -ForegroundColor Yellow
            return $null
        }
        $stashed = $true
    }

    Write-Host "[update] $Name is $behind commit(s) behind $upstream - fast-forwarding..." -ForegroundColor Cyan
    & git -C $Dir merge --ff-only "@{u}" | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[update] $Name could not fast-forward - leaving it as it was." -ForegroundColor Yellow
        if ($stashed) { & git -C $Dir stash pop | Out-Host }
        return $null
    }
    $after = (& git -C $Dir rev-parse HEAD 2>$null)
    Write-Host "[update] $Name updated -> $($after.Substring(0,7))" -ForegroundColor Green

    if ($stashed) {
        # No 2>&1 here: redirecting a native command's stderr in PowerShell wraps
        # each line in a NativeCommandError, which renders as a red failure block
        # even though this path is handled. Let git's own message through as text.
        & git -C $Dir stash pop | Out-Host
        if ($LASTEXITCODE -ne 0) {
            # The restore conflicts with what just arrived. Leaving conflict markers
            # in the tree would be worse than useless — these are mostly generated
            # JSON the app parses on startup — so put those paths back to the new
            # HEAD and leave the stash entry alone. Nothing is lost; it just needs
            # a human. Only the colliding paths are reset, so any OTHER local work
            # still sitting in the tree is untouched.
            & git -C $Dir reset -q -- $collisions 2>$null | Out-Null
            & git -C $Dir checkout -q --force -- $collisions 2>$null | Out-Null
            Write-Host "[update] $Name - your version of those file(s) conflicts with the update." -ForegroundColor Yellow
            Write-Host "         They are SAVED in the stash and the working tree now matches the remote." -ForegroundColor Yellow
            Write-Host "         Recover with:  git -C `"$Dir`" stash list   /   git -C `"$Dir`" stash pop" -ForegroundColor Yellow
        } else {
            Write-Host "[update] $Name - local changes restored on top of the update." -ForegroundColor DarkGray
        }
    }

    # Only reinstall dependencies when the pulled range actually touched them.
    $touched = & git -C $Dir diff --name-only "$before..$after" -- requirements.txt pyproject.toml 2>$null
    if ($touched) { return $Dir }
    return $null
}

$Script:DepsChanged = @()
$Script:ComfyUIDir = ""
$skipUpdate = $NoUpdate -or ($env:AGENTY_NO_UPDATE -and $env:AGENTY_NO_UPDATE -notin @("0", "false", "no", ""))
if (-not $skipUpdate) {
    # An explicit auto_update = false in the settings file opts out permanently.
    $settingsLocal = Join-Path $ProjectRoot "config\settings.local.json"
    if (Test-Path $settingsLocal) {
        try {
            $sj = Get-Content $settingsLocal -Raw | ConvertFrom-Json
            if ($null -ne $sj.auto_update -and -not $sj.auto_update) { $skipUpdate = $true }
            if ($sj.PSObject.Properties.Name -contains "comfyui_dir" -and $sj.comfyui_dir) {
                $Script:ComfyUIDir = [string]$sj.comfyui_dir
            }
        } catch { }
    }
}
if ($skipUpdate) {
    Write-Host "[update] Update check skipped." -ForegroundColor DarkGray
} else {
    $parent = Split-Path -Parent $ProjectRoot
    $targets = @(
        @{ n = "agentY";      d = $ProjectRoot }
        @{ n = "agenty_core"; d = (Join-Path $parent "agenty_core") }
    )

    # The sidebar extension is a THIRD checkout, and often two: the clone ComfyUI
    # actually loads (<ComfyUI>\custom_nodes\agentY-comfyuiConnect) and, for anyone
    # who works on it, a dev clone beside agentY. Update every one we can find —
    # a stale installed clone is the usual reason the panel is missing a feature the
    # host already has. Candidates are cheap path guesses; nothing is searched.
    $extCandidates = @(
        (Join-Path $parent "agentY-comfyuiConnect")
    )
    foreach ($root in @($env:AGENTY_COMFYUI_DIR, $Script:ComfyUIDir)) {
        if ($root) { $extCandidates += (Join-Path $root "custom_nodes\agentY-comfyuiConnect") }
    }
    foreach ($name in @("comfyui", "ComfyUI", "ComfyUI_windows_portable\ComfyUI")) {
        $extCandidates += (Join-Path (Join-Path $parent $name) "custom_nodes\agentY-comfyuiConnect")
    }
    $seen = @{}
    $i = 0
    foreach ($cand in $extCandidates) {
        if (-not $cand -or -not (Test-Path (Join-Path $cand ".git"))) { continue }
        $full = (Resolve-Path $cand).Path
        if ($seen.ContainsKey($full)) { continue }
        $seen[$full] = $true
        $i++
        $label = if ($i -eq 1) { "agentY-comfyuiConnect" } else { "agentY-comfyuiConnect #$i" }
        $targets += @{ n = $label; d = $full; ext = $true }
    }

    $extUpdated = $false
    foreach ($r in $targets) {
        if (-not (Test-Path $r.d)) { continue }
        $before = (& git -C $r.d rev-parse HEAD 2>$null)
        $changed = Update-Repo -Name $r.n -Dir $r.d
        if ($changed) { $Script:DepsChanged += $r.n }
        if ($r.ContainsKey("ext")) {
            $after = (& git -C $r.d rev-parse HEAD 2>$null)
            if ($before -and $after -and $before -ne $after) { $extUpdated = $true }
        }
    }
    if ($extUpdated) {
        Write-Host "[update] The ComfyUI sidebar extension changed - restart ComfyUI (or reload the browser for JS-only changes) to pick it up." -ForegroundColor Yellow
    }
    Write-Host ""
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

    # A pull that changed requirements.txt / pyproject.toml needs the environment
    # brought back in line BEFORE the app imports anything.
    if ($Script:DepsChanged.Count -gt 0) {
        Write-Host "[update] Dependencies changed in: $($Script:DepsChanged -join ', ') - reinstalling..." -ForegroundColor Cyan
        $uv = Get-Command uv -ErrorAction SilentlyContinue
        if ($uv) { uv pip install -r requirements.txt | Out-Host }
        else { pip install -r requirements.txt | Out-Host }
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[update] Dependency install returned $LASTEXITCODE - check the output above." -ForegroundColor Yellow
        }
        Write-Host ""
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

    # ── Free the target port ────────────────────────────────────────────────
    # A previous host whose Ctrl+C didn't fully stop it can linger; on Windows
    # SO_REUSEADDR then lets that stale instance keep answering with OLD code, so
    # a plain restart isn't enough (new routes 404). Stop any leftover agentY host
    # still bound to $Port before launching a fresh one. A non-agentY process on
    # the port is reported but never killed.
    try {
        $bound = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
                 Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($procId in $bound) {
            if (-not $procId) { continue }
            $cim = Get-CimInstance Win32_Process -Filter "ProcessId = $procId" -ErrorAction SilentlyContinue
            if ($cim -and $cim.CommandLine -like "*agenty_ui_server*") {
                Write-Host "[run_agent] Port $Port held by a leftover agentY host (PID $procId) - stopping it." -ForegroundColor Yellow
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            } elseif ($cim) {
                $pname = (Get-Process -Id $procId -ErrorAction SilentlyContinue).Name
                Write-Host "[run_agent] WARNING: port $Port is held by PID $procId ($pname), which is not an agentY host - leaving it alone. Use -Port to pick another port." -ForegroundColor Red
            }
        }
        if ($bound) { Start-Sleep -Milliseconds 500 }
    } catch {
        Write-Host "[run_agent] Port check skipped: $($_.Exception.Message)" -ForegroundColor DarkYellow
    }

    # ── Refresh ComfyUI model caches ────────────────────────────────────────
    Write-Host "[run_agent] Refreshing ComfyUI model cache..." -ForegroundColor Cyan
    $env:COMFYUI_MODELS_REFRESHED = ""   # clear any leftover value from a previous run in this session
    python scripts/refresh_models.py
    $env:COMFYUI_MODELS_REFRESHED = "1"  # prevent re-runs in child processes
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
