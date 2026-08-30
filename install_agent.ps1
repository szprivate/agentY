<#
.SYNOPSIS
    One-shot installer / bootstrapper for the full agentY stack.

    Sets up the four repos that make up agentY, prompts for the secrets it needs,
    and drops the chat UI into your ComfyUI:

      * agentY                (this repo)  - the Strands chat host / pipeline
      * agenty_core           (sibling)    - shared ComfyUI/HF/web/file tool layer
                                             (installed editable; required)
      * agentY-mcp            (sibling)    - the MCP-server / Claude-Desktop variant
                                             (optional; skip with -SkipMcp)
      * agentY-comfyuiConnect (into ComfyUI\custom_nodes) - the sidebar chat UI

    The UI is the "agentY" tab inside ComfyUI. There is no Chainlit, Docker,
    Postgres, or MinIO - conversations persist to a local SQLite file.

    Runs on Windows PowerShell 5.1+ and PowerShell 7+.

    On macOS use install_agent.sh instead: same seven stages and the same
    switches, but it checks for the Command Line Tools that insightface and
    sam3 need to compile there, and it does not offer a CUDA build of torch
    that macOS has no wheel for.

.PARAMETER ComfyUIPath
    Path to your ComfyUI install (the folder containing custom_nodes\). If omitted
    the script auto-detects common locations and otherwise asks.

.PARAMETER ParentDir
    Where the sibling repos (agenty_core, agentY-mcp) live / will be cloned.
    Defaults to this repo's parent directory.

.PARAMETER SkipMcp
    Do not clone / set up the agentY-mcp sibling repo.

.PARAMETER SkipComfyNode
    Do not touch ComfyUI (skip locating it and installing the sidebar node).

.PARAMETER NonInteractive
    Never prompt. Use existing values / defaults only (for CI or re-runs).

.PARAMETER SkipTorch
    Do not offer the CUDA build of torch. requirements.txt then pulls the CPU one
    in transitively, which costs SAM3 grounding about a minute per call.

.PARAMETER TorchIndexUrl
    Wheel index the CUDA build of torch comes from. Defaults to cu128; pick the
    one matching your driver from https://pytorch.org/get-started/locally/.

.EXAMPLE
    .\install_agent.ps1
.EXAMPLE
    .\install_agent.ps1 -ComfyUIPath "D:\ai\ComfyUI" -SkipMcp
#>

[CmdletBinding()]
param(
    [switch]$Help,
    [string]$ComfyUIPath   = "",
    [string]$ParentDir     = "",
    [switch]$SkipMcp,
    [switch]$SkipComfyNode,
    [switch]$NonInteractive,
    [switch]$SkipTorch,
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128"
)

Set-StrictMode -Version 3.0
$ErrorActionPreference = "Stop"

# -- UI helpers ---------------------------------------------------------------
function Write-Header  { param([string]$Text) Write-Host ""; Write-Host "===  $Text  ===" -ForegroundColor Cyan }
function Write-Success { param([string]$Text) Write-Host "  [ok] $Text" -ForegroundColor Green }
function Write-Info    { param([string]$Text) Write-Host "  [i]  $Text" -ForegroundColor Yellow }
function Write-Fail    { param([string]$Text) Write-Host "  [!]  $Text" -ForegroundColor Red }
function Exit-WithError { param([string]$Message, [int]$Code = 1) Write-Host ""; Write-Fail $Message; exit $Code }

if ($Help) {
    Get-Help $MyInvocation.MyCommand.Definition -Detailed
    exit 0
}

$Script:OnWindows = $true
if (Get-Variable -Name IsWindows -Scope Global -ErrorAction SilentlyContinue) {
    $Script:OnWindows = [bool]$IsWindows
}

$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = (Get-Location).Path }
if (-not $ParentDir)   { $ParentDir = Split-Path -Parent $ProjectRoot }

# -- Low-level helpers --------------------------------------------------------
function Write-TextNoBom {
    # Write UTF-8 without a BOM - python-dotenv / json readers dislike a leading BOM.
    param([string]$Path, [string]$Text)
    $enc = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Text, $enc)
}

function Get-EnvValue {
    param([string]$File, [string]$Key)
    if (-not (Test-Path $File)) { return $null }
    foreach ($line in (Get-Content -LiteralPath $File)) {
        if ($line -match "^\s*$([regex]::Escape($Key))\s*=") {
            return ($line -replace "^\s*$([regex]::Escape($Key))\s*=\s*", '').Trim()
        }
    }
    return $null
}

function Set-EnvValue {
    param([string]$File, [string]$Key, [string]$Value)
    $lines = @()
    if (Test-Path $File) { $lines = @(Get-Content -LiteralPath $File) }
    $done = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "^\s*$([regex]::Escape($Key))\s*=") {
            $lines[$i] = "$Key=$Value"; $done = $true; break
        }
    }
    if (-not $done) { $lines += "$Key=$Value" }
    Write-TextNoBom $File (($lines -join "`r`n") + "`r`n")
}

function Test-Placeholder {
    # Treat the .env_example stubs and blanks as "not yet set".
    param([string]$Value)
    if (-not $Value) { return $true }
    return ($Value -match '^(hf_\.\.\.|sk-ant-\.\.\.|comfyui-\.\.\.|sk-\.\.\.)$')
}

function Format-Masked {
    param([string]$Value)
    if (Test-Placeholder $Value) { return "(not set)" }
    if ($Value.Length -le 8) { return ('*' * $Value.Length) }
    return $Value.Substring(0, 4) + ('*' * 4) + '...'
}

function Read-Secret {
    # Prompt for a secret, keeping the existing value on <Enter>. Returns the
    # resolved value and writes it into $File when the user supplies a new one.
    param([string]$File, [string]$Key, [string]$Label, [string]$HelpText)
    $cur   = Get-EnvValue $File $Key
    $isSet = -not (Test-Placeholder $cur)
    if ($NonInteractive) {
        if ($isSet) { return $cur } else { return "" }
    }
    Write-Host ""
    Write-Host "  $Label" -ForegroundColor White
    if ($HelpText) { Write-Host "    $HelpText" -ForegroundColor DarkGray }
    $suffix = if ($isSet) { " [Enter = keep $(Format-Masked $cur)]" } else { " [Enter = skip]" }
    $entered = Read-Host "    $Key$suffix"
    if ($entered.Trim() -ne "") {
        Set-EnvValue $File $Key $entered.Trim()
        Write-Success "$Key set"
        return $entered.Trim()
    }
    if ($isSet) { return $cur }
    return ""
}

function Invoke-Native {
    # Run a native command, streaming its output straight to the host (so it does
    # not pollute the return value), then report success from $LASTEXITCODE. Note:
    # we do NOT redirect stderr (2>&1) - under $ErrorActionPreference='Stop' that
    # would turn git/uv's normal stderr progress into a terminating error.
    param([string]$What, [scriptblock]$Block, [switch]$AllowFail)
    & $Block | Out-Host
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        if ($AllowFail) { Write-Info "$What returned $code (continuing)."; return $false }
        Exit-WithError "$What failed (exit $code)."
    }
    return $true
}

function Ensure-Repo {
    # Clone $Url into $Dir if missing; otherwise best-effort fast-forward pull.
    param([string]$Name, [string]$Url, [string]$Dir, [switch]$Required)
    if (Test-Path (Join-Path $Dir ".git")) {
        Write-Info "$Name present at $Dir - updating (git pull --ff-only)"
        Invoke-Native "git pull ($Name)" { git -C $Dir pull --ff-only } -AllowFail | Out-Null
        Write-Success "$Name up to date"
        return
    }
    if ((Test-Path $Dir) -and (Get-ChildItem -LiteralPath $Dir -Force -ErrorAction SilentlyContinue)) {
        Write-Info "$Name exists at $Dir but is not a git checkout - leaving it untouched"
        return
    }
    Write-Info "Cloning $Name -> $Dir"
    $ok = Invoke-Native "git clone ($Name)" { git clone $Url $Dir } -AllowFail:(-not $Required)
    if ($ok) { Write-Success "$Name cloned" }
    elseif ($Required) { Exit-WithError "Could not clone required repo $Name from $Url." }
}

function Get-VenvPython {
    param([string]$Dir)
    $pyRel = if ($Script:OnWindows) { "Scripts\python.exe" } else { "bin/python" }
    return (Join-Path (Join-Path $Dir ".venv") $pyRel)
}

function Install-Torch {
    # requirements.txt deliberately leaves torch unpinned: the right build depends
    # on the machine's CUDA version, and PyPI's Windows wheel is CPU-only - on CPU
    # a single SAM3 grounding call goes from ~0.2s to about a minute. So the GPU
    # build has to be installed HERE, before the sam3 -> timm -> torch chain in
    # requirements.txt resolves and pulls the CPU one in instead.
    # Runs with the cwd already inside $Dir, so uv targets that venv.
    param([string]$Dir)
    if ($SkipTorch) { Write-Info "Skipping the CUDA torch install (-SkipTorch)"; return }
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        Write-Info "No NVIDIA GPU detected (no nvidia-smi) - leaving torch to requirements.txt"
        return
    }
    $py = Get-VenvPython $Dir
    if (Test-Path $py) {
        & $py -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"
        if ($LASTEXITCODE -eq 0) {
            & $py -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
            if ($LASTEXITCODE -eq 0) { Write-Success "torch already installed with CUDA support"; return }
            Write-Info "torch is installed but CPU-only - reinstalling from the CUDA index"
        }
    }
    if ($NonInteractive) {
        Write-Info "NVIDIA GPU detected; non-interactive, so skipping the ~3 GB CUDA download. Run:"
        Write-Host  "       uv pip install torch torchvision --index-url $TorchIndexUrl" -ForegroundColor White
        return
    }
    Write-Host ""
    Write-Host "  An NVIDIA GPU was detected." -ForegroundColor White
    Write-Host "    SAM3 grounding (locating what to circle) wants the CUDA build of torch;" -ForegroundColor DarkGray
    Write-Host "    the wheel on PyPI is CPU-only and makes a call take about a minute." -ForegroundColor DarkGray
    Write-Host "    About 3 GB from $TorchIndexUrl." -ForegroundColor DarkGray
    $ans = Read-Host "    Install the CUDA build now? [Y/n]"
    if ($ans.Trim() -match '^(n|no)$') {
        Write-Info "Skipped - requirements.txt will pull the CPU build"
        return
    }
    Invoke-Native "uv pip install (torch)" { uv pip install --python $py torch torchvision --index-url $TorchIndexUrl } -AllowFail | Out-Null
}

function Setup-Venv {
    # Create (if missing) a uv venv in $Dir and install its requirements.txt.
    param([string]$Name, [string]$Dir, [switch]$WithTorch)
    $venv = Join-Path $Dir ".venv"
    $py = Get-VenvPython $Dir
    Push-Location $Dir
    try {
        if (-not ((Test-Path $venv) -and (Test-Path $py))) {
            if (Test-Path $venv) { Write-Info "$Name .venv incomplete - recreating"; Remove-Item -Recurse -Force $venv }
            Write-Info "Creating $Name .venv (uv venv)"
            Invoke-Native "uv venv ($Name)" { uv venv .venv } | Out-Null
        } else {
            Write-Info "$Name .venv already exists"
        }
        if ($WithTorch) { Install-Torch -Dir $Dir }
        $req = Join-Path $Dir "requirements.txt"
        if (-not (Test-Path $req)) { Exit-WithError "requirements.txt not found in $Dir." }
        Write-Info "Installing $Name dependencies (uv pip install -r requirements.txt)"
        # --python: name the target interpreter. With a conda environment active
        # (miniconda auto-activates `base`), uv installs into THAT rather than the
        # .venv we just made, and the whole dependency set lands somewhere agentY
        # never looks - an install that reports success and imports nothing.
        Invoke-Native "uv pip install ($Name)" { uv pip install --python $py -r requirements.txt } | Out-Null
    } finally { Pop-Location }
    Write-Success "$Name environment ready"
}

function Test-Environment {
    # Import-check every dependency agentY names, in the venv that will run it.
    # Most of them are also somebody else's transitive dep, so a gap in
    # requirements.txt otherwise stays invisible until the machine that resolved
    # differently quietly loses a feature.
    param([string]$Dir)
    $py     = Get-VenvPython $Dir
    $script = Join-Path (Join-Path $Dir "scripts") "check_env.py"
    if (-not ((Test-Path $py) -and (Test-Path $script))) {
        Write-Info "Dependency check skipped (no venv python or scripts/check_env.py)"
        return
    }
    & $py $script --gpu | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Required dependencies are missing - see the list above."
        Write-Info "Re-run after fixing:  .venv\Scripts\python.exe scripts\check_env.py"
    } else {
        Write-Success "Every required dependency imports"
    }
}

function Ensure-EnvFile {
    param([string]$Dir)
    $envFile    = Join-Path $Dir ".env"
    $envExample = Join-Path $Dir ".env_example"
    if (-not (Test-Path $envFile)) {
        if (-not (Test-Path $envExample)) { Exit-WithError ".env_example not found in $Dir." }
        Copy-Item $envExample $envFile
        Write-Info "Created .env from .env_example in $Dir"
    }
    return $envFile
}

function Test-ComfyUIDir {
    param([string]$Path)
    if (-not $Path -or -not (Test-Path $Path)) { return $null }
    # Accept the ComfyUI root, or a common wrapper folder one level up.
    foreach ($cand in @($Path, (Join-Path $Path "ComfyUI"), (Join-Path $Path "ComfyUI_windows_portable\ComfyUI"))) {
        if (Test-Path (Join-Path $cand "custom_nodes")) { return (Resolve-Path $cand).Path }
    }
    return $null
}

function Find-ComfyUI {
    param([string]$Hint)
    $candidates = @()
    if ($Hint) { $candidates += $Hint }
    if ($Script:OnWindows) {
        foreach ($drive in @("D:", "E:", "C:")) {
            $candidates += @(
                "$drive\ai\ComfyUI", "$drive\AI\ComfyUI", "$drive\ComfyUI",
                "$drive\ai\comfyui__", "$drive\ai\ComfyUI_windows_portable\ComfyUI",
                "$drive\ComfyUI_windows_portable\ComfyUI"
            )
        }
        if ($env:USERPROFILE) { $candidates += "$env:USERPROFILE\ComfyUI" }
    } else {
        if ($env:HOME) { $candidates += @("$env:HOME/ComfyUI", "$env:HOME/comfyui") }
    }
    foreach ($c in $candidates) {
        $hit = Test-ComfyUIDir $c
        if ($hit) { return $hit }
    }
    return $null
}

# =============================================================================
Write-Host ""
Write-Host "  agentY stack installer" -ForegroundColor Cyan
Write-Host "  repo root: $ProjectRoot" -ForegroundColor DarkGray
Write-Host "  siblings : $ParentDir" -ForegroundColor DarkGray

# -- 1. Preflight -------------------------------------------------------------
Write-Header "1 / 7  Preflight"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Exit-WithError "'git' is not on PATH. Install Git and re-run."
}
Write-Success "git found"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Exit-WithError "'uv' is not on PATH. Install it: https://docs.astral.sh/uv/getting-started/installation/"
}
Write-Success "uv found: $(uv --version)"

# -- 2. Sibling repos (agenty_core, agentY-mcp) -------------------------------
Write-Header "2 / 7  Sibling repos"
$CoreDir = Join-Path $ParentDir "agenty_core"
Ensure-Repo -Name "agenty_core" -Url "https://github.com/szprivate/agenty_core.git" -Dir $CoreDir -Required
if (-not (Test-Path (Join-Path $CoreDir "pyproject.toml"))) {
    Exit-WithError "agenty_core looks incomplete at $CoreDir (no pyproject.toml). agentY's requirements.txt installs it editable via '-e ../agenty_core'."
}

$McpDir = Join-Path $ParentDir "agentY-mcp"
if (-not $SkipMcp) {
    Ensure-Repo -Name "agentY-mcp" -Url "https://github.com/szprivate/agentY-mcp.git" -Dir $McpDir
} else {
    Write-Info "Skipping agentY-mcp (-SkipMcp)"
}

# -- 3. agentY environment ----------------------------------------------------
Write-Header "3 / 7  agentY environment"
Setup-Venv -Name "agentY" -Dir $ProjectRoot -WithTorch

# -- 4. Secrets (.env) --------------------------------------------------------
Write-Header "4 / 7  Secrets (.env)"
$EnvFile = Ensure-EnvFile $ProjectRoot
if ($NonInteractive) {
    Write-Info "Non-interactive: leaving .env values as-is. Edit $EnvFile to set keys."
} else {
    Write-Host "  Press Enter to keep an existing value or skip an optional one." -ForegroundColor DarkGray
}
$hfToken  = Read-Secret $EnvFile "HF_TOKEN"          "Hugging Face token (gated model downloads)"       "Create at https://huggingface.co/  (account -> Access Tokens)"
$null     = Read-Secret $EnvFile "ANTHROPIC_API_KEY" "Anthropic API key (Claude) - recommended"         "Create at https://platform.claude.com/"
$comfyKey = Read-Secret $EnvFile "COMFYUI_API_KEY"   "ComfyUI API key (optional - auth / API nodes)"    "https://platform.comfy.org/profile/api-keys  - blank for a local ComfyUI"
$null     = Read-Secret $EnvFile "DASHSCOPE_API_KEY" "DashScope / Alibaba Model Studio key (optional)"  "For Qwen models: https://bailian.console.alibabacloud.com/"
Write-Success "agentY .env ready ($EnvFile)"

# -- 5. ComfyUI: locate + install the sidebar node ----------------------------
Write-Header "5 / 7  ComfyUI sidebar node"
$ResolvedComfy = $null
if ($SkipComfyNode) {
    Write-Info "Skipping ComfyUI node install (-SkipComfyNode)"
} else {
    $ResolvedComfy = Test-ComfyUIDir $ComfyUIPath
    if (-not $ResolvedComfy) { $ResolvedComfy = Find-ComfyUI $ComfyUIPath }
    if ($ResolvedComfy) {
        Write-Success "Found ComfyUI: $ResolvedComfy"
        if (-not $NonInteractive) {
            $ans = Read-Host "    Use this ComfyUI? [Y/n] (or type another path)"
            if ($ans.Trim() -and $ans.Trim() -notmatch '^(y|yes)$') {
                if ($ans.Trim() -match '^(n|no)$') { $ResolvedComfy = $null }
                else { $ResolvedComfy = Test-ComfyUIDir $ans.Trim() }
            }
        }
    }
    if (-not $ResolvedComfy -and -not $NonInteractive) {
        $entered = Read-Host "    ComfyUI folder (contains custom_nodes\), or Enter to skip"
        if ($entered.Trim()) {
            $ResolvedComfy = Test-ComfyUIDir $entered.Trim()
            if (-not $ResolvedComfy) { Write-Fail "That folder doesn't look like a ComfyUI install - skipping." }
        }
    }

    if ($ResolvedComfy) {
        $nodeDir = Join-Path (Join-Path $ResolvedComfy "custom_nodes") "agentY-comfyuiConnect"
        Ensure-Repo -Name "agentY-comfyuiConnect" -Url "https://github.com/szprivate/agentY-comfyuiConnect.git" -Dir $nodeDir
        Write-Success "Sidebar node installed under $ResolvedComfy\custom_nodes - restart ComfyUI once."

        # Record where the agentY host lives so the sidebar's "Start server" button
        # can relaunch run_agent.ps1 when the host is down. The agentY host also
        # rewrites this on startup; this bootstrap makes the button work day-1,
        # before the host has ever run. (Gitignored - machine-specific.)
        if (Test-Path $nodeDir) {
            $hostCfg = Join-Path $nodeDir ".agenty_host.json"
            $cfgObj  = [pscustomobject]@{ project_root = $ProjectRoot; run_script = "run_agent.ps1" }
            Write-TextNoBom $hostCfg ($cfgObj | ConvertTo-Json -Depth 5)
            Write-Success "Recorded agentY host location for the 'Start server' button"
        }

        # Offer to set this ComfyUI's URL as a local override (settings.local.json).
        # Committed defaults live in config/settings.default.toml (localhost); the
        # local JSON is deep-merged over them and is gitignored.
        $localSettings = Join-Path $ProjectRoot "config\settings.local.json"
        if (-not $NonInteractive) {
            $curUrl = "http://127.0.0.1:8188"
            $obj = $null
            if (Test-Path $localSettings) {
                try { $obj = Get-Content -LiteralPath $localSettings -Raw | ConvertFrom-Json } catch { $obj = $null }
                if ($obj -and $obj.comfyui_url) { $curUrl = [string]$obj.comfyui_url }
            }
            $newUrl = Read-Host "    ComfyUI URL for settings.local.json [Enter = keep $curUrl]"
            if ($newUrl.Trim() -and $newUrl.Trim() -ne $curUrl) {
                if (-not $obj) { $obj = [pscustomobject]@{} }
                $obj | Add-Member -NotePropertyName comfyui_url -NotePropertyValue $newUrl.Trim() -Force
                Write-TextNoBom $localSettings ($obj | ConvertTo-Json -Depth 10)
                Write-Success "settings.local.json comfyui_url -> $($newUrl.Trim())"
            }
        }
    } else {
        Write-Info "No ComfyUI configured. Install the node later:"
        Write-Host  "       git clone https://github.com/szprivate/agentY-comfyuiConnect  <ComfyUI>\custom_nodes\agentY-comfyuiConnect" -ForegroundColor White
    }
}

# -- 6. agentY-mcp environment (optional) -------------------------------------
Write-Header "6 / 7  agentY-mcp environment"
if ($SkipMcp -or -not (Test-Path (Join-Path $McpDir "requirements.txt"))) {
    Write-Info "Skipping agentY-mcp environment."
} else {
    Setup-Venv -Name "agentY-mcp" -Dir $McpDir
    $mcpEnv = Ensure-EnvFile $McpDir
    # The MCP host (Claude Desktop) supplies the model, so agentY-mcp only needs
    # HF_TOKEN + COMFYUI_API_KEY. Reuse what we just collected for agentY.
    if (-not (Test-Placeholder $hfToken))  { Set-EnvValue $mcpEnv "HF_TOKEN"        $hfToken;  Write-Info "Propagated HF_TOKEN to agentY-mcp .env" }
    if (-not (Test-Placeholder $comfyKey)) { Set-EnvValue $mcpEnv "COMFYUI_API_KEY" $comfyKey; Write-Info "Propagated COMFYUI_API_KEY to agentY-mcp .env" }
    Write-Success "agentY-mcp ready ($McpDir)"
}

# -- 7. Verify ----------------------------------------------------------------
Write-Header "7 / 7  Dependency check"
Test-Environment -Dir $ProjectRoot

# -- Done ---------------------------------------------------------------------
Write-Header "Setup complete"
Write-Host ""
Write-Host "  Installed:" -ForegroundColor Cyan
Write-Host "    - agentY        $ProjectRoot" -ForegroundColor White
Write-Host "    - agenty_core   $CoreDir  (editable dependency)" -ForegroundColor White
if (-not $SkipMcp -and (Test-Path (Join-Path $McpDir 'requirements.txt'))) {
    Write-Host "    - agentY-mcp    $McpDir" -ForegroundColor White
}
if ($ResolvedComfy) {
    Write-Host "    - sidebar node  $ResolvedComfy\custom_nodes\agentY-comfyuiConnect" -ForegroundColor White
}
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Cyan
Write-Host "    1. Start the agent chat host:" -ForegroundColor Yellow
Write-Host "         .\run_agent.ps1" -ForegroundColor White
if ($ResolvedComfy) {
    Write-Host "    2. Restart ComfyUI, then click the 'agentY' tab in its left sidebar." -ForegroundColor Yellow
} else {
    Write-Host "    2. Install agentY-comfyuiConnect into ComfyUI\custom_nodes and restart ComfyUI." -ForegroundColor Yellow
}
if (-not $SkipMcp -and (Test-Path (Join-Path $McpDir 'requirements.txt'))) {
    Write-Host "    3. (optional) Register agentY-mcp with Claude Desktop - see $McpDir\README.md" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "  Review secrets/paths anytime in:  $EnvFile" -ForegroundColor DarkGray
Write-Host "  Defaults: $ProjectRoot\config\settings.default.toml  (overrides: settings.local.json)" -ForegroundColor DarkGray
Write-Host ""
