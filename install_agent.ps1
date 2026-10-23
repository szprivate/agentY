<#
.SYNOPSIS
    One-shot installer / bootstrapper for the full agentY stack.

    Sets up the five repos that make up agentY, prompts for the secrets it needs,
    and drops the chat UI into your ComfyUI:

      * agentY                (this repo)  - the Strands chat host / pipeline
      * agentY-core           (sibling)    - shared ComfyUI/HF/web/file tool layer
                                             (installed editable; required)
      * agentY-mcp            (sibling)    - the MCP-server / Claude-Desktop variant
                                             (optional; skip with -SkipMcp)
      * agentY-comfyuiConnect (into ComfyUI\custom_nodes) - the sidebar chat UI

    The UI is the "agentY" tab inside ComfyUI. There is no Chainlit, Docker,
    Postgres, or MinIO - conversations persist to a local SQLite file.

    Runs on Windows PowerShell 5.1+ and PowerShell 7+.

    On macOS use install_agent.sh instead: same eight stages and the same
    switches, but it checks for the Command Line Tools that insightface and
    sam3 need to compile there, and it does not offer a CUDA build of torch
    that macOS has no wheel for.

.PARAMETER ComfyUIPath
    Path to your ComfyUI install (the folder containing custom_nodes\). If omitted
    the script auto-detects common locations and otherwise asks.

.PARAMETER ParentDir
    Where the sibling repos (agentY-core, agentY-mcp) live / will be cloned.
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

.PARAMETER PythonVersion
    The Python a NEW .venv is made with (default 3.12, what agentY is tested on;
    uv downloads it if needed). An existing .venv keeps its interpreter.

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
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [string]$PythonVersion = "3.12"
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
    $parent = Split-Path -Parent $Path
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
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
    $entered = Read-Answer "    $Key$suffix"
    if ($entered.Trim() -ne "") {
        Set-EnvValue $File $Key $entered.Trim()
        Write-Success "$Key set"
        return $entered.Trim()
    }
    if ($isSet) { return $cur }
    return ""
}

function Read-Answer {
    # Read-Host, but never $null: with input redirected and used up (a piped
    # install, a closed console) it returns nothing, and .Trim() on that is a
    # terminating error halfway through an install.
    param([string]$Prompt)
    $v = Read-Host $Prompt
    if ($null -eq $v) { return "" }
    return [string]$v
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

# -- Release channel ----------------------------------------------------------
# "stable" (the default) follows each repository's `stable` branch, which only
# moves when a release is made - all five repositories together, so what runs is
# always a set that was tested together. "dev" follows the repository's `dev`
# branch, where new work lands first, commit by commit. Set update_channel in
# config/settings.local.json, or AGENTY_UPDATE_CHANNEL.
#
# Following is done by pointing the checked-out branch's upstream at
# origin/stable (or back at the default branch). Nothing is checked out and no
# file changes here; the fast-forward below then simply has a different target.
function Get-UpdateChannel {
    $c = $env:AGENTY_UPDATE_CHANNEL
    if (-not $c) {
        $f = Join-Path $ProjectRoot "config\settings.local.json"
        if (Test-Path $f) {
            try {
                $p = (Get-Content -LiteralPath $f -Raw | ConvertFrom-Json).PSObject.Properties['update_channel']
                if (($null -ne $p) -and $p.Value) { $c = [string]$p.Value }
            } catch { }
        }
    }
    if ($c -and ($c.Trim().ToLower() -eq "dev")) { return "dev" }
    return "stable"
}

function Set-ChannelUpstream {
    # Call after a fetch. Leaves a detached HEAD and a repository without a
    # `stable` branch exactly as they are.
    param([string]$Dir, [string]$Channel)
    $ErrorActionPreference = "Continue"
    $branch = (& git -C $Dir branch --show-current 2>$null)
    if (-not $branch) { return }
    $up = (& git -C $Dir rev-parse --abbrev-ref --symbolic-full-name "@{u}" 2>$null)
    if ($Channel -eq "stable") {
        & git -C $Dir rev-parse -q --verify "refs/remotes/origin/stable" 2>$null | Out-Null
        if (($LASTEXITCODE -eq 0) -and ($up -ne "origin/stable")) {
            & git -C $Dir branch -q --set-upstream-to=origin/stable 2>$null | Out-Null
        }
    } else {
        # dev: the repository's `dev` branch, where new work lands first. Only a
        # branch that was following a release line (stable, or the default
        # branch) is re-pointed - a feature branch keeps its own upstream.
        $head = (& git -C $Dir symbolic-ref -q --short "refs/remotes/origin/HEAD" 2>$null)
        $target = $head
        & git -C $Dir rev-parse -q --verify "refs/remotes/origin/dev" 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $target = "origin/dev" }
        if ($target -and ($up -ne $target) -and (($up -eq "origin/stable") -or ($up -eq $head))) {
            & git -C $Dir branch -q --set-upstream-to=$target 2>$null | Out-Null
        }
    }
    $global:LASTEXITCODE = 0
}

function Update-Checkout {
    # Fast-forward the checkout at $Dir to its upstream, and say what happened.
    # Returns $true when it is now at the remote's commit.
    #
    # The same rules as run_agent.ps1's Update-Repo, for the same reason: the agent
    # rewrites tracked files in its own checkouts (config/models.json on every
    # start, saved templates in agentY-core), so a plain `git pull` refuses as soon
    # as the remote touches one of them - and an installer that then says "up to
    # date" leaves an old agentY next to a new agentY-core. Only the local files
    # the incoming commits also change are parked in a stash; nothing is discarded.
    param([string]$Name, [string]$Dir)
    # git writes warnings to stderr; under "Stop" Windows PowerShell turns any such
    # line into a terminating error, even with 2>$null. Exit codes are checked.
    $ErrorActionPreference = "Continue"

    $upstream = & git -C $Dir rev-parse --abbrev-ref --symbolic-full-name "@{u}" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $upstream) {
        Write-Info "$Name has no upstream branch - left as it is"
        return $false
    }
    & git -C $Dir fetch --quiet 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "$Name - could not reach the remote. NOT updated; continuing with the local copy."
        return $false
    }
    Set-ChannelUpstream -Dir $Dir -Channel $Script:Channel
    $upstream = & git -C $Dir rev-parse --abbrev-ref --symbolic-full-name "@{u}" 2>$null
    $behind = 0; $ahead = 0
    [void][int]::TryParse([string](& git -C $Dir rev-list --count "HEAD..@{u}" 2>$null), [ref]$behind)
    [void][int]::TryParse([string](& git -C $Dir rev-list --count "@{u}..HEAD" 2>$null), [ref]$ahead)
    if ($behind -eq 0) {
        if (($ahead -gt 0) -and ($upstream -eq "origin/stable")) {
            Write-Info "$Name is $ahead commit(s) ahead of the stable release - left as it is"
        } else {
            Write-Success "$Name up to date"
        }
        return $true
    }
    if ($ahead -gt 0) {
        Write-Fail "$Name has $ahead local commit(s) and is $behind behind $upstream. NOT updated - merge it yourself."
        return $false
    }

    $incoming = @(& git -c core.quotepath=false -C $Dir diff --name-only "HEAD..@{u}" 2>$null)
    $dirty  = @(& git -c core.quotepath=false -C $Dir diff --name-only HEAD 2>$null)
    $dirty += @(& git -c core.quotepath=false -C $Dir ls-files --others --exclude-standard 2>$null)
    $inc = @{}
    foreach ($f in $incoming) { if ($f) { $inc[$f] = $true } }
    $collisions = @($dirty | Where-Object { $_ -and $inc.ContainsKey($_) } | Select-Object -Unique)

    $stashed = $false
    if ($collisions.Count -gt 0) {
        Write-Info "$Name - $($collisions.Count) locally changed file(s) also changed upstream; parking them in a stash:"
        foreach ($c in $collisions) { Write-Host "         $c" -ForegroundColor DarkYellow }
        $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        & git -C $Dir stash push --include-untracked -m "agentY installer $stamp" -- $collisions | Out-Host
        if ($LASTEXITCODE -ne 0) {
            Write-Fail "$Name - could not stash those files. NOT updated."
            return $false
        }
        $stashed = $true
    }

    Write-Info "$Name is $behind commit(s) behind $upstream - fast-forwarding"
    & git -C $Dir merge --ff-only "@{u}" | Out-Host
    if ($LASTEXITCODE -ne 0) {
        if ($stashed) { & git -C $Dir stash pop | Out-Host }
        Write-Fail "$Name could not be fast-forwarded. NOT updated - see git's message above."
        return $false
    }
    if ($stashed) {
        & git -C $Dir stash pop | Out-Host
        if ($LASTEXITCODE -ne 0) {
            # The restore conflicts with what just arrived. Conflict markers in
            # generated JSON the app parses at startup would stop it starting, so
            # those paths go back to the new HEAD and the stash entry is kept.
            & git -C $Dir reset -q -- $collisions 2>$null | Out-Null
            & git -C $Dir checkout -q --force -- $collisions 2>$null | Out-Null
            Write-Info "$Name - your version of those file(s) conflicts with the update. It is SAVED in"
            Write-Info "the stash (git -C `"$Dir`" stash list); the working tree now matches the remote."
        }
    }
    $now = (& git -C $Dir rev-parse --short HEAD 2>$null)
    Write-Success "$Name updated -> $now"
    return $true
}

function Stop-RunningHost {
    # Windows cannot replace a file a running process has open. With the agent
    # host up, `uv pip install` fails halfway through a package ("Access is
    # denied") and leaves the environment between two versions - an agent that no
    # longer starts. So nothing may be running from this venv while it changes.
    param([string]$Dir)
    if (-not $Script:OnWindows) { return }
    $venv = Join-Path $Dir ".venv"
    if (-not (Test-Path $venv)) { return }
    $root = (Resolve-Path $venv).Path.TrimEnd('\') + '\'
    $procs = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ExecutablePath -and $_.ExecutablePath.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)
    })
    if ($procs.Count -eq 0) { return }
    $ids = ($procs | ForEach-Object { $_.ProcessId }) -join ", "
    Write-Info "agentY is running from this environment (PID $ids)."
    Write-Info "Windows cannot replace files it has open, so it has to stop for the install."
    if ($NonInteractive) {
        Exit-WithError "Stop the agent (close its window, or 'Stop server' in the sidebar) and run the installer again."
    }
    $ans = Read-Answer "    Stop it now? [Y/n]"
    if ($ans.Trim() -match '^(n|no)$') {
        Exit-WithError "Stop the agent (close its window, or 'Stop server' in the sidebar) and run the installer again."
    }
    foreach ($p in $procs) {
        # /T: the venv's python.exe is a launcher; the interpreter holding the
        # files is its child.
        & taskkill /PID $p.ProcessId /T /F 2>$null | Out-Null
    }
    Start-Sleep -Milliseconds 800
    Write-Success "agentY stopped - start it again with .\run_agent.ps1 when the installer is done"
}

function Find-BepicViewer {
    # The bEpic viewer (szprivate/ComfyUI-ImageViewer) under ANY folder name: a git
    # checkout whose origin is that repo, or a registry/zip copy whose pyproject.toml
    # points at it. ComfyUI-Manager installs it as "comfyui-imageviewer", for one.
    # Returns @{ Dir; Git } or $null.
    param([string]$CustomNodes)
    if (-not (Test-Path $CustomNodes)) { return $null }
    foreach ($d in Get-ChildItem -LiteralPath $CustomNodes -Directory -ErrorAction SilentlyContinue) {
        if (Test-Path (Join-Path $d.FullName ".git")) {
            $origin = ""
            try { $origin = (& git -C $d.FullName remote get-url origin 2>$null | Select-Object -First 1) } catch {}
            if ($origin -match 'szprivate/ComfyUI-ImageViewer(\.git)?/?$') { return @{ Dir = $d.FullName; Git = $true } }
            continue
        }
        $toml = Join-Path $d.FullName "pyproject.toml"
        if ((Test-Path $toml) -and ((Get-Content -LiteralPath $toml -Raw -ErrorAction SilentlyContinue) -match 'szprivate/ComfyUI-ImageViewer')) {
            return @{ Dir = $d.FullName; Git = $false }
        }
    }
    return $null
}

function Install-BepicViewer {
    # Present (any folder name): update a git checkout, leave a registry copy to its
    # manager. Missing: clone it - into ComfyUI-ImageViewer, or beside a different
    # project that already holds that folder name.
    param([string]$CustomNodes)
    $url = "https://github.com/szprivate/ComfyUI-ImageViewer.git"
    $found = Find-BepicViewer -CustomNodes $CustomNodes
    if ($found) {
        if ($found.Git) { Ensure-Repo -Name "ComfyUI-ImageViewer (bEpic viewer)" -Url $url -Dir $found.Dir }
        else { Write-Success "bEpic viewer present at $($found.Dir) (not a git checkout - update it from ComfyUI-Manager)" }
        return
    }
    $dir = Join-Path $CustomNodes "ComfyUI-ImageViewer"
    if ((Test-Path $dir) -and (Get-ChildItem -LiteralPath $dir -Force -ErrorAction SilentlyContinue)) {
        Write-Info "A different ComfyUI-ImageViewer is at $dir (not szprivate's) - leaving it; installing the bEpic viewer beside it."
        Write-Info "If ComfyUI reports duplicate node names, disable one of the two."
        $dir = Join-Path $CustomNodes "ComfyUI-ImageViewer-bEpic"
    } else {
        Write-Info "bEpic viewer (szprivate/ComfyUI-ImageViewer) not found in $CustomNodes - installing it."
    }
    Ensure-Repo -Name "ComfyUI-ImageViewer (bEpic viewer)" -Url $url -Dir $dir
}

function Ensure-Repo {
    # Clone $Url into $Dir if missing; otherwise fast-forward it (Update-Checkout).
    param([string]$Name, [string]$Url, [string]$Dir, [switch]$Required)
    if (Test-Path (Join-Path $Dir ".git")) {
        Update-Checkout -Name $Name -Dir $Dir | Out-Null
        return
    }
    if ((Test-Path $Dir) -and (Get-ChildItem -LiteralPath $Dir -Force -ErrorAction SilentlyContinue)) {
        Write-Info "$Name exists at $Dir but is not a git checkout - leaving it untouched"
        return
    }
    Write-Info "Cloning $Name -> $Dir"
    # core.longpaths: git for Windows stops at 260 characters by itself, and the
    # template corpus in agentY-core has file names long enough to cross that in
    # a deep install folder. Set at clone, it stays in the checkout's own config.
    # A new clone starts on its channel's branch: `stable` (the release) or
    # `dev`. A repository without that branch is cloned as it is.
    $ok = $false
    if ($Script:Channel) {
        $ErrorActionPreference = "Continue"
        & git clone -q -c core.longpaths=true --branch $Script:Channel $Url $Dir 2>$null | Out-Null
        $ok = ($LASTEXITCODE -eq 0)
        $ErrorActionPreference = "Stop"
        if ((-not $ok) -and (Test-Path (Join-Path $Dir ".git"))) {
            Remove-Item -LiteralPath $Dir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    if (-not $ok) {
        $ok = Invoke-Native "git clone ($Name)" { git clone -c core.longpaths=true $Url $Dir } -AllowFail
    }
    if ($ok) { Write-Success "$Name cloned"; return }
    # A clone that failed at checkout leaves a .git behind; the next run would
    # take that for an install and report it up to date. The folder was missing
    # or empty before this call, so removing it loses nothing.
    if (Test-Path (Join-Path $Dir ".git")) {
        Remove-Item -LiteralPath $Dir -Recurse -Force -ErrorAction SilentlyContinue
    }
    if ($Required) { Exit-WithError "Could not clone required repo $Name from $Url - see git's message above." }
    Write-Fail "$Name could not be cloned - see git's message above. Continuing without it."
}

function Get-VenvPython {
    param([string]$Dir)
    $pyRel = if ($Script:OnWindows) { "Scripts\python.exe" } else { "bin/python" }
    return (Join-Path (Join-Path $Dir ".venv") $pyRel)
}

function Get-LockedSpec {
    # "torch==2.11.0" when requirements.lock pins it, else just "torch".
    param([string]$Dir, [string]$Package)
    $lock = Join-Path $Dir "requirements.lock"
    if (Test-Path $lock) {
        foreach ($line in (Get-Content -LiteralPath $lock)) {
            if ($line -match "^$([regex]::Escape($Package))==([0-9][^\s;]*)") { return "$Package==$($Matches[1])" }
        }
    }
    return $Package
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
    $ans = Read-Answer "    Install the CUDA build now? [Y/n]"
    if ($ans.Trim() -match '^(n|no)$') {
        Write-Info "Skipped - requirements.txt will pull the CPU build"
        return
    }
    # The CUDA build of the version the release pins: left to itself this
    # installs the newest torch, and the requirements install that follows would
    # then replace it with the pinned one - from PyPI, which is the CPU build.
    $torchSpec = Get-LockedSpec -Dir $Dir -Package "torch"
    $visionSpec = Get-LockedSpec -Dir $Dir -Package "torchvision"
    Invoke-Native "uv pip install (torch)" { uv pip install --python $py $torchSpec $visionSpec --index-url $TorchIndexUrl } -AllowFail | Out-Null
}

function Move-CoreDir {
    # requirements.txt names `-e ../agentY-core`. Where the folder still has its
    # old name (agenty_core), scripts/core_dir.py renames it and leaves a link
    # behind - or links the new name to it when Windows will not let go of it.
    param([string]$Dir)
    $py     = Get-VenvPython $Dir
    $script = Join-Path (Join-Path $ProjectRoot "scripts") "core_dir.py"
    if (-not ((Test-Path $py) -and (Test-Path $script))) { return }
    & $py $script | Out-Host
    $Script:CoreDir = Join-Path $ParentDir "agentY-core"
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
            # Named, not whatever `python` uv meets first on this machine (a
            # system 3.9, conda's base): 3.12 is what agentY is developed and
            # tested on. uv downloads it when it is not installed.
            Write-Info "Creating $Name .venv (uv venv, Python $PythonVersion)"
            $made = Invoke-Native "uv venv ($Name)" { uv venv .venv --python $PythonVersion } -AllowFail
            if (-not $made) {
                Write-Info "Python $PythonVersion is not available - using uv's default interpreter"
                Invoke-Native "uv venv ($Name)" { uv venv .venv } | Out-Null
            }
        } else {
            Write-Info "$Name .venv already exists"
            Stop-RunningHost -Dir $Dir
        }
        if ($WithTorch) { Install-Torch -Dir $Dir }
        Move-CoreDir -Dir $Dir
        $req = Join-Path $Dir "requirements.txt"
        if (-not (Test-Path $req)) { Exit-WithError "requirements.txt not found in $Dir." }
        Write-Info "Installing $Name dependencies (uv pip install -r requirements.txt)"
        # --python: name the target interpreter. With a conda environment active
        # (miniconda auto-activates `base`), uv installs into THAT rather than the
        # .venv we just made, and the whole dependency set lands somewhere agentY
        # never looks - an install that reports success and imports nothing.
        # requirements.lock: the versions the release was tested with, as
        # constraints. requirements.txt alone only sets minimums.
        if (Test-Path (Join-Path $Dir "requirements.lock")) {
            Write-Info "Using the versions pinned in requirements.lock"
            Invoke-Native "uv pip install ($Name)" { uv pip install --python $py -r requirements.txt -c requirements.lock } | Out-Null
        } else {
            Invoke-Native "uv pip install ($Name)" { uv pip install --python $py -r requirements.txt } | Out-Null
        }
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

function Test-LocalPort {
    param([int]$Port)
    $c = New-Object System.Net.Sockets.TcpClient
    try { return ($c.ConnectAsync("127.0.0.1", $Port).Wait(300) -and $c.Connected) }
    catch { return $false }
    finally { $c.Close() }
}

function Get-LocalSettings {
    # The settings object; an empty one when there is no file yet. $null when the
    # file is there but cannot be read as JSON - a caller must then leave it alone
    # instead of writing a fresh one over the user's settings.
    param([string]$File)
    if (-not (Test-Path $File)) { return [pscustomobject]@{} }
    $raw = Get-Content -LiteralPath $File -Raw
    if (-not $raw -or -not $raw.Trim()) { return [pscustomobject]@{} }
    try { $o = $raw | ConvertFrom-Json } catch { return $null }
    if ($o -is [System.Management.Automation.PSCustomObject]) { return $o }
    return $null
}

function Get-SettingValue {
    # One top-level value, or "" - asked for by name, because under StrictMode
    # reading a property the object does not have is a terminating error.
    param($Settings, [string]$Name)
    if ($null -eq $Settings) { return "" }
    $p = $Settings.PSObject.Properties[$Name]
    if (($null -eq $p) -or ($null -eq $p.Value)) { return "" }
    return [string]$p.Value
}

function Get-EmbedderPreset {
    param([string]$File)
    $o = Get-LocalSettings $File
    if ($null -eq $o) { return "" }
    if (($null -eq $o.PSObject.Properties['memory']) -or -not $o.memory) { return "" }
    if (($null -eq $o.memory.PSObject.Properties['embedder']) -or -not $o.memory.embedder) { return "" }
    if (($null -eq $o.memory.embedder.PSObject.Properties['preset'])) { return "" }
    return [string]$o.memory.embedder.preset
}

function Set-EmbedderPreset {
    param([string]$File, [string]$Preset)
    $o = Get-LocalSettings $File
    if ($null -eq $o) { return $false }
    if (($null -eq $o.PSObject.Properties['memory']) -or -not $o.memory) {
        $o | Add-Member -NotePropertyName memory -NotePropertyValue ([pscustomobject]@{}) -Force
    }
    if (($null -eq $o.memory.PSObject.Properties['embedder']) -or -not $o.memory.embedder) {
        $o.memory | Add-Member -NotePropertyName embedder -NotePropertyValue ([pscustomobject]@{}) -Force
    }
    $o.memory.embedder | Add-Member -NotePropertyName preset -NotePropertyValue $Preset -Force
    Write-TextNoBom $File ($o | ConvertTo-Json -Depth 20)
    return $true
}

function Select-Embedder {
    # Long-term memory needs an embedder: a model that turns text into vectors so
    # memories can be found by meaning. Asked here because the old default (Ollama)
    # is not on every machine, and without one memory silently does nothing.
    param([string]$EnvFile, [string]$LocalSettings)
    $ollamaUp = Test-LocalPort 11434
    $choices = @(
        @{ id = "local";     key = "";                  label = "Local - built into agentY, no service or key (downloads ~130 MB on first use)" },
        @{ id = "ollama";    key = "";                  label = "Ollama - nomic-embed-text" + $(if ($ollamaUp) { " (running)" } else { " (not running now - start it before agentY)" }) },
        @{ id = "dashscope"; key = "DASHSCOPE_API_KEY"; label = "Alibaba DashScope - text-embedding-v4" },
        @{ id = "gemini";    key = "GEMINI_API_KEY";    label = "Google Gemini - gemini-embedding-001" },
        @{ id = "openai";    key = "OPENAI_API_KEY";    label = "OpenAI - text-embedding-3-small" }
    )
    if ($null -eq (Get-LocalSettings $LocalSettings)) {
        Write-Fail "$LocalSettings is not valid JSON - left untouched, so no embedder was chosen."
        Write-Info "Fix or delete that file, then pick one in Settings > Models > Memory embedder."
        return
    }
    $current = Get-EmbedderPreset $LocalSettings
    $default = if ($current) { $current } elseif ($ollamaUp) { "ollama" } else { "local" }
    if ($NonInteractive) {
        if (-not $current) { Set-EmbedderPreset $LocalSettings $default | Out-Null; Write-Success "Memory embedder: $default" }
        else { Write-Success "Memory embedder: $current (kept)" }
        return
    }
    Write-Host "  Long-term memory turns what the agent learns into vectors so it can find it" -ForegroundColor DarkGray
    Write-Host "  again by meaning. Pick the model that does it (changeable later in Settings >" -ForegroundColor DarkGray
    Write-Host "  Models > Memory embedder; switching re-embeds what is stored)." -ForegroundColor DarkGray
    for ($i = 0; $i -lt $choices.Count; $i++) {
        $c = $choices[$i]
        $note = ""
        if ($c.key -and (Test-Placeholder (Get-EnvValue $EnvFile $c.key))) { $note = "  [needs $($c.key) - asked next]" }
        $mark = if ($c.id -eq $default) { "*" } else { " " }
        Write-Host ("   {0}{1}) {2}{3}" -f $mark, ($i + 1), $c.label, $note) -ForegroundColor White
    }
    $defIdx = 1 + [array]::IndexOf(@($choices | ForEach-Object { $_.id }), $default)
    $tries = 0
    while ($true) {
        $tries++
        if ($tries -gt 5) {
            Write-Info "No embedder chosen - pick one later in Settings > Models > Memory embedder."
            return
        }
        $ans = (Read-Answer "    Memory embedder [1-$($choices.Count), Enter = $defIdx]").Trim()
        if (-not $ans) { $ans = "$defIdx" }
        $n = 0
        if (-not [int]::TryParse($ans, [ref]$n) -or $n -lt 1 -or $n -gt $choices.Count) { Write-Fail "Type a number from the list."; continue }
        $pick = $choices[$n - 1]
        if ($pick.key -and (Test-Placeholder (Get-EnvValue $EnvFile $pick.key))) {
            $val = Read-Secret $EnvFile $pick.key "$($pick.key) for the memory embedder" ""
            if (-not $val) { Write-Fail "That embedder needs $($pick.key). Pick another, or enter the key."; continue }
        }
        Set-EmbedderPreset $LocalSettings $pick.id | Out-Null
        Write-Success "Memory embedder: $($pick.id) (settings.local.json)"
        if ($pick.id -eq "local") { Write-Info "Its model downloads into models\embeddings\ the first time memory is used." }
        return
    }
}

function Test-MenuConsole {
    # Arrow keys need a real console: input not piped, a window to redraw in.
    if ($NonInteractive) { return $false }
    try {
        if ([Console]::IsInputRedirected -or [Console]::IsOutputRedirected) { return $false }
        $null = [Console]::CursorTop
        $null = [Console]::WindowWidth
        return $true
    } catch { return $false }
}

function Select-Many {
    # A multi-select list: Up/Down move, Space ticks, Enter confirms ('a' ticks or
    # clears all). Returns the chosen indexes (an [int[]], possibly empty).
    # $Items: hashtables with Label, Hint (optional) and Selected.
    # Without an interactive console it asks for numbers instead ("1,3"; Enter =
    # keep the pre-ticked ones). With -NonInteractive it asks nothing and returns
    # the pre-ticked items, or none when $NoneWhenSilent is set.
    param([string]$Title, [object[]]$Items, [switch]$NoneWhenSilent)
    $n = $Items.Count
    $sel = New-Object 'bool[]' $n
    for ($i = 0; $i -lt $n; $i++) { $sel[$i] = [bool]$Items[$i].Selected }
    $result = { $o = @(); for ($i = 0; $i -lt $n; $i++) { if ($sel[$i]) { $o += $i } }; $o }
    if ($n -eq 0) { return @() }
    if ($NonInteractive) {
        if ($NoneWhenSilent) { return @() }
        return (& $result)
    }
    Write-Host ""
    Write-Host "  $Title" -ForegroundColor White
    if (-not (Test-MenuConsole)) {
        for ($i = 0; $i -lt $n; $i++) {
            $mark = if ($sel[$i]) { "*" } else { " " }
            Write-Host ("   {0}{1}) {2}" -f $mark, ($i + 1), $Items[$i].Label) -ForegroundColor White
            if ($Items[$i].Hint) { Write-Host "         $($Items[$i].Hint)" -ForegroundColor DarkGray }
        }
        $keep = (@(& $result) | ForEach-Object { $_ + 1 }) -join ","
        $tries = 0
        while ($true) {
            $tries++
            $ans = (Read-Answer "    Numbers separated by commas, 'none', or Enter = [$keep]").Trim()
            if (-not $ans) { return (& $result) }
            if ($ans -match '^(none|n|no)$') { return @() }
            $picked = @(); $bad = $false
            foreach ($t in ($ans -split '[,\s]+' | Where-Object { $_ })) {
                $k = 0
                if ([int]::TryParse($t, [ref]$k) -and $k -ge 1 -and $k -le $n) { $picked += ($k - 1) } else { $bad = $true }
            }
            if (-not $bad) { return @($picked | Sort-Object -Unique) }
            if ($tries -ge 5) { Write-Info "No valid answer - keeping [$keep]."; return (& $result) }
            Write-Fail "Type numbers from the list, e.g. 1,3"
        }
    }
    Write-Host "    Up/Down move, Space ticks, A ticks all, Enter confirms" -ForegroundColor DarkGray
    $cur = 0
    $rows = $n + 0
    $width = [Math]::Max(20, [Console]::WindowWidth - 1)
    $draw = {
        for ($i = 0; $i -lt $n; $i++) {
            $line = ("  {0} [{1}] {2}" -f $(if ($i -eq $cur) { ">" } else { " " }), $(if ($sel[$i]) { "x" } else { " " }), $Items[$i].Label)
            if ($Items[$i].Hint -and $i -eq $cur) { $line += "  - " + $Items[$i].Hint }
            if ($line.Length -gt $width) { $line = $line.Substring(0, $width) }
            Write-Host $line.PadRight($width) -ForegroundColor $(if ($i -eq $cur) { "Cyan" } else { "White" })
        }
    }
    $cursorWas = $true
    try { $cursorWas = [Console]::CursorVisible; [Console]::CursorVisible = $false } catch { }
    try {
        & $draw
        $top = [Console]::CursorTop - $n          # right even when drawing scrolled the window
        while ($true) {
            $key = [Console]::ReadKey($true)
            switch ($key.Key) {
                'UpArrow'   { $cur = ($cur + $n - 1) % $n }
                'DownArrow' { $cur = ($cur + 1) % $n }
                'Spacebar'  { $sel[$cur] = -not $sel[$cur] }
                'A'         { $all = $sel -contains $false; for ($i = 0; $i -lt $n; $i++) { $sel[$i] = $all } }
                'Enter'     { return (& $result) }
            }
            [Console]::SetCursorPosition(0, [Math]::Max(0, $top))
            & $draw
        }
    } finally {
        try { [Console]::CursorVisible = $cursorWas } catch { }
    }
}

# The LLM providers agentY can run its agents on: the provider token of a
# 'provider,model' setting (src/agent.py: ollama, anthropic, the DashScope
# aliases, openai, google/gemini) and the key each one reads. Ollama is local.
$Script:LlmProviders = @(
    @{ id = "anthropic"; label = "Anthropic (Claude) - recommended"; keys = @("ANTHROPIC_API_KEY");
       text = "Anthropic API key (Claude)"; help = "Create at https://platform.claude.com/" },
    @{ id = "dashscope"; label = "Alibaba Model Studio (DashScope, Qwen)"; keys = @("DASHSCOPE_API_KEY");
       text = "DashScope / Alibaba Model Studio key"; help = "For Qwen models: https://bailian.console.alibabacloud.com/" },
    @{ id = "openai";    label = "OpenAI (GPT)"; keys = @("OPENAI_API_KEY");
       text = "OpenAI API key"; help = "Create at https://platform.openai.com/api-keys" },
    @{ id = "google";    label = "Google (Gemini)"; keys = @("GEMINI_API_KEY", "GOOGLE_API_KEY");
       text = "Gemini API key"; help = "Create at https://aistudio.google.com/apikey" },
    @{ id = "ollama";    label = "Ollama (local models, no key)"; keys = @();
       text = ""; help = "" }
)

function Select-LlmProviders {
    # Which providers to ask for keys. Ticked from the start: the ones whose key is
    # already in .env (Anthropic when there is none yet). Asks the key of each
    # ticked one only; the rest of .env is left alone.
    param([string]$EnvFile)
    $items = @()
    $anySet = $false
    foreach ($p in $Script:LlmProviders) {
        $set = $false
        foreach ($k in $p.keys) { if (-not (Test-Placeholder (Get-EnvValue $EnvFile $k))) { $set = $true } }
        if ($set) { $anySet = $true }
        $items += @{ Label = $p.label; Selected = $set; Hint = $(if ($set) { "key already in .env" } else { "" }) }
    }
    if (-not $anySet) { $items[0].Selected = $true }
    $picked = @(Select-Many "Which LLM providers will you use? (a key is asked for each ticked one)" $items)
    foreach ($i in $picked) {
        $p = $Script:LlmProviders[$i]
        if (-not $p.keys.Count) {
            if (-not $NonInteractive) { Write-Info "Ollama needs no key. Install it from https://ollama.com/ and start it before agentY." }
            continue
        }
        $key = $p.keys[0]
        # Gemini also reads GOOGLE_API_KEY: an existing one counts as set.
        foreach ($k in $p.keys) { if (-not (Test-Placeholder (Get-EnvValue $EnvFile $k))) { $key = $k; break } }
        $null = Read-Secret $EnvFile $key $p.text $p.help
    }
}

function Install-McpPresets {
    # Offer the catalog in config/mcp_presets.json; scripts/mcp_presets.py does the
    # merge into config/mcp.json so the logic exists once. Nothing here asks for
    # an add-on to be installed into a desktop app: those steps are printed.
    param([string]$EnvFile)
    $py = Get-VenvPython $ProjectRoot
    $tool = Join-Path $ProjectRoot "scripts\mcp_presets.py"
    if (-not ((Test-Path $py) -and (Test-Path $tool))) { Write-Info "MCP presets skipped (no venv python or scripts\mcp_presets.py)"; return }
    if ($NonInteractive) { Write-Info "Non-interactive: no MCP presets added. Add them later in Settings > MCP servers."; return }
    $env:PYTHONIOENCODING = "utf-8"
    $presets = @()
    foreach ($line in @(& $py $tool list)) {
        $f = ([string]$line) -split '[|]', 5
        if ($f.Count -lt 5) { continue }
        $presets += [pscustomobject]@{ Id = $f[0]; Name = $f[1]; Category = $f[2]; Default = ($f[3] -eq "1"); Description = $f[4] }
    }
    if (-not $presets.Count) { return }
    $items = @($presets | ForEach-Object { @{ Label = "$($_.Name)  [$($_.Category)]"; Hint = $_.Description; Selected = $_.Default } })
    $picked = @(Select-Many "MCP servers to add (extra tools for the agent; Enter with none ticked adds nothing)" $items -NoneWhenSilent)
    if (-not $picked.Count) { Write-Info "No MCP presets chosen - add them later in Settings > MCP servers."; return }
    $ids = @($picked | ForEach-Object { $presets[$_].Id })
    foreach ($id in $ids) {
        foreach ($line in @(& $py $tool keys $id)) {
            $f = ([string]$line) -split '[|]', 4
            if ($f.Count -lt 4) { continue }
            $label = "$($f[1]) (for MCP preset $id)" + $(if ($f[3] -eq "1") { " - optional" } else { "" })
            $null = Read-Secret $EnvFile $f[0] $label $f[2]
        }
    }
    $later = @()
    foreach ($line in @(& $py $tool apply @ids)) {
        $f = ([string]$line) -split '[|]', 2
        if ($f.Count -lt 2) { continue }
        switch ($f[0]) {
            "added"   { Write-Success "MCP server '$($f[1])' added to config\mcp.json" }
            "skipped" { Write-Info "MCP server '$($f[1])' is already configured - kept as it is" }
            "needs_bundle" {
                Write-Info "MCP server '$($f[1])' needs its .mcpb bundle first - not written to config\mcp.json yet"
                $st = (@(& $py $tool steps $f[1]) -join "`n").Trim()
                if ($st) { $later += [pscustomobject]@{ Id = $f[1]; Steps = $st } }
            }
            default   { Write-Fail "Unknown MCP preset '$($f[1])'" }
        }
        if ($f[0] -in @("unknown", "needs_bundle")) { continue }
        $warm = (@(& $py $tool warmup $f[1]) | Select-Object -Last 1)
        if ($LASTEXITCODE -ne 0) { Write-Info "Could not pre-fetch '$($f[1])' ($warm) - it is fetched on first use" }
        elseif ($warm -and $warm -ne "nothing to fetch") { Write-Success "Pre-fetched '$($f[1])'" }
        $steps = (@(& $py $tool steps $f[1]) -join "`n").Trim()
        if ($steps) { $later += [pscustomobject]@{ Id = $f[1]; Steps = $steps } }
    }
    if ($later.Count) {
        Write-Host ""
        Write-Host "  Finish later (the installer does not touch other applications):" -ForegroundColor White
        foreach ($l in $later) {
            Write-Host "    $($l.Id):" -ForegroundColor Cyan
            foreach ($s in ($l.Steps -split "`n")) { Write-Host "      $s" -ForegroundColor DarkGray }
        }
        Write-Info "Or ask the agent: it can do these steps for you once it runs (Settings > MCP servers > Add from presets does it too)."
    }
}

# =============================================================================
Write-Host ""
Write-Host "  agentY stack installer" -ForegroundColor Cyan
Write-Host "  repo root: $ProjectRoot" -ForegroundColor DarkGray
Write-Host "  siblings : $ParentDir" -ForegroundColor DarkGray

# -- 1. Preflight -------------------------------------------------------------
Write-Header "1 / 8  Preflight"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Exit-WithError "'git' is not on PATH. Install Git and re-run."
}
Write-Success "git found"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Exit-WithError "'uv' is not on PATH. Install it: https://docs.astral.sh/uv/getting-started/installation/"
}
Write-Success "uv found: $(uv --version)"

# -- 2. Sibling repos (agentY-core, agentY-mcp) -------------------------------
Write-Header "2 / 8  Sibling repos"
$Script:Channel = Get-UpdateChannel
Write-Info "Release channel: $($Script:Channel)"
# agentY itself first: the siblings below are brought to the remote's newest
# commit, and a new agentY-core under an old agentY is an agent that does not
# start. When the pull replaced this very script, the new one takes over - the
# rest of an old installer would install what the old agentY needed.
if (Test-Path (Join-Path $ProjectRoot ".git")) {
    $selfBefore = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash
    Update-Checkout -Name "agentY" -Dir $ProjectRoot | Out-Null
    $selfAfter = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash
    if (($selfBefore -ne $selfAfter) -and -not $env:AGENTY_INSTALLER_RESTARTED) {
        Write-Info "The installer itself was updated - starting the new one."
        $env:AGENTY_INSTALLER_RESTARTED = "1"
        try { & $PSCommandPath @PSBoundParameters } finally { $env:AGENTY_INSTALLER_RESTARTED = $null }
        exit $LASTEXITCODE
    }
} else {
    Write-Info "agentY at $ProjectRoot is not a git checkout - it cannot be updated from here"
}
# The tool layer's repository is agentY-core. An install from before it was
# renamed has the folder as agenty_core: it is updated where it is here, and
# moved to the new name in stage 3 (scripts/core_dir.py), once there is a Python
# to run that with.
$CoreDir = Join-Path $ParentDir "agentY-core"
$LegacyCoreDir = Join-Path $ParentDir "agenty_core"
if ((-not (Test-Path $CoreDir)) -and (Test-Path (Join-Path $LegacyCoreDir ".git"))) { $CoreDir = $LegacyCoreDir }
Ensure-Repo -Name "agentY-core" -Url "https://github.com/szprivate/agentY-core.git" -Dir $CoreDir -Required
if (-not (Test-Path (Join-Path $CoreDir "pyproject.toml"))) {
    Exit-WithError "agentY-core looks incomplete at $CoreDir (no pyproject.toml). agentY's requirements.txt installs it editable via '-e ../agentY-core'."
}

$McpDir = Join-Path $ParentDir "agentY-mcp"
if (-not $SkipMcp) {
    Ensure-Repo -Name "agentY-mcp" -Url "https://github.com/szprivate/agentY-mcp.git" -Dir $McpDir
} else {
    Write-Info "Skipping agentY-mcp (-SkipMcp)"
}

# -- 3. agentY environment ----------------------------------------------------
Write-Header "3 / 8  agentY environment"
Setup-Venv -Name "agentY" -Dir $ProjectRoot -WithTorch

# -- 4. Secrets (.env) --------------------------------------------------------
Write-Header "4 / 8  Secrets (.env)"
$EnvFile = Ensure-EnvFile $ProjectRoot
if ($NonInteractive) {
    Write-Info "Non-interactive: leaving .env values as-is. Edit $EnvFile to set keys."
} else {
    Write-Host "  Press Enter to keep an existing value or skip an optional one." -ForegroundColor DarkGray
}
Select-LlmProviders -EnvFile $EnvFile
$hfToken  = Read-Secret $EnvFile "HF_TOKEN"          "Hugging Face token (gated model downloads)"       "Create at https://huggingface.co/  (account -> Access Tokens)"
$comfyKey = Read-Secret $EnvFile "COMFYUI_API_KEY"   "ComfyUI API key (optional - auth / API nodes)"    "https://platform.comfy.org/profile/api-keys  - blank for a local ComfyUI"
Install-McpPresets -EnvFile $EnvFile
Write-Success "agentY .env ready ($EnvFile)"

# -- 5. Long-term memory embedder ----------------------------------------------
Write-Header "5 / 8  Memory embedder"
Select-Embedder -EnvFile $EnvFile -LocalSettings (Join-Path $ProjectRoot "config\settings.local.json")

# -- 6. ComfyUI: locate + install the sidebar node ----------------------------
Write-Header "6 / 8  ComfyUI sidebar node"
$ResolvedComfy = $null
if ($SkipComfyNode) {
    Write-Info "Skipping ComfyUI node install (-SkipComfyNode)"
} else {
    $ResolvedComfy = Test-ComfyUIDir $ComfyUIPath
    if (-not $ResolvedComfy) { $ResolvedComfy = Find-ComfyUI $ComfyUIPath }
    if ($ResolvedComfy) {
        Write-Success "Found ComfyUI: $ResolvedComfy"
        if (-not $NonInteractive) {
            $ans = Read-Answer "    Use this ComfyUI? [Y/n] (or type another path)"
            if ($ans.Trim() -and $ans.Trim() -notmatch '^(y|yes)$') {
                if ($ans.Trim() -match '^(n|no)$') { $ResolvedComfy = $null }
                else { $ResolvedComfy = Test-ComfyUIDir $ans.Trim() }
            }
        }
    }
    if (-not $ResolvedComfy -and -not $NonInteractive) {
        $entered = Read-Answer "    ComfyUI folder (contains custom_nodes\), or Enter to skip"
        if ($entered.Trim()) {
            $ResolvedComfy = Test-ComfyUIDir $entered.Trim()
            if (-not $ResolvedComfy) { Write-Fail "That folder doesn't look like a ComfyUI install - skipping." }
        }
    }

    if ($ResolvedComfy) {
        $nodeDir = Join-Path (Join-Path $ResolvedComfy "custom_nodes") "agentY-comfyuiConnect"
        Ensure-Repo -Name "agentY-comfyuiConnect" -Url "https://github.com/szprivate/agentY-comfyuiConnect.git" -Dir $nodeDir
        Write-Success "Sidebar node installed under $ResolvedComfy\custom_nodes - restart ComfyUI once."

        # The bEpic viewer is agentY's standard viewer (Settings > Load & save nodes
        # saves into it by default). A failed clone only costs the viewer.
        Install-BepicViewer -CustomNodes (Join-Path $ResolvedComfy "custom_nodes")

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
        $obj = Get-LocalSettings $localSettings
        if ($null -eq $obj) {
            Write-Fail "$localSettings is not valid JSON - left untouched (the ComfyUI URL was not asked)."
        } elseif (-not $NonInteractive) {
            $curUrl = Get-SettingValue $obj "comfyui_url"
            if (-not $curUrl) { $curUrl = "http://127.0.0.1:8188" }
            $newUrl = Read-Answer "    ComfyUI URL for settings.local.json [Enter = keep $curUrl]"
            if ($newUrl.Trim() -and $newUrl.Trim() -ne $curUrl) {
                $obj | Add-Member -NotePropertyName comfyui_url -NotePropertyValue $newUrl.Trim() -Force
                Write-TextNoBom $localSettings ($obj | ConvertTo-Json -Depth 20)
                Write-Success "settings.local.json comfyui_url -> $($newUrl.Trim())"
            }
        }
    } else {
        Write-Info "No ComfyUI configured. Install the node later:"
        Write-Host  "       git clone https://github.com/szprivate/agentY-comfyuiConnect  <ComfyUI>\custom_nodes\agentY-comfyuiConnect" -ForegroundColor White
    }
}

# -- 7. agentY-mcp environment (optional) -------------------------------------
Write-Header "7 / 8  agentY-mcp environment"
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

# -- 8. Verify ----------------------------------------------------------------
Write-Header "8 / 8  Dependency check"
Test-Environment -Dir $ProjectRoot

# -- Done ---------------------------------------------------------------------
Write-Header "Setup complete"
Write-Host ""
Write-Host "  Installed:" -ForegroundColor Cyan
Write-Host "    - agentY        $ProjectRoot" -ForegroundColor White
Write-Host "    - agentY-core   $(Join-Path $ParentDir 'agentY-core')  (editable dependency)" -ForegroundColor White
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
exit 0
