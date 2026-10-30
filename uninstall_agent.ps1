<#
.SYNOPSIS
    Uninstaller for the agentY stack (the counterpart of install_agent.ps1).

    An install only creates folders - no registry keys, environment variables, services or
    shortcuts - so removing agentY means removing folders, and this script only ever removes
    folders it has IDENTIFIED as agentY's:

      * agentY, agentY-core (older: agenty_core), agentY-mcp   - the code and each .venv
      * your data inside them: conversations, memory, settings, .env keys, MCP tokens, output
      * agentY-comfyuiConnect (older: comfyui-agent-canvas, agentY_bridge) in custom_nodes
      * the bEpic viewer (ComfyUI-ImageViewer, any folder name) - kept unless you say so

    It shows a plan first (every path with its size, and what it KEEPS), then asks per group.
    It never removes ComfyUI, models, Ollama, Strata, Blender, or anything it cannot verify
    (git origin github.com/szprivate/<repo>, or agentY marker files).

    Works on old installs too (Chainlit-era layouts, agenty_core, copied panel folders) and
    standalone: download it anywhere and point it at the install with -Path, or let it look.

.PARAMETER Path
    The agentY folder to remove. Without it the script looks at its own folder, the current
    folder (and its parents), ComfyUI's .agenty_host.json and the usual places.

.PARAMETER ParentDir
    Where agentY-core / agentY-mcp live. Defaults to the parent of the agentY folder.

.PARAMETER ComfyUIPath
    Only look for the panel and viewer in this ComfyUI (default: every ComfyUI found).

.PARAMETER Yes
    Do not ask. Removes agentY, its siblings, your data (after a backup zip) and the panel;
    keeps the viewer and ComfyUI Desktop unless -IncludeViewer / -IncludeComfyDesktop.

.PARAMETER DryRun
    Only print the plan and the checks. Nothing is asked, stopped or removed.

.PARAMETER KeepUserData
    With -Yes: keep conversations, memory, settings and keys.

.PARAMETER NoBackup
    Do not offer / make the backup zip of your data.

.PARAMETER IncludeViewer
    With -Yes: also remove the bEpic viewer.

.PARAMETER IncludeComfyDesktop
    With -Yes: also run the winget uninstall of ComfyUI Desktop.

.PARAMETER Port
    The port the agentY host listens on (default 5000).

.EXAMPLE
    .\uninstall_agent.ps1 -DryRun
.EXAMPLE
    .\uninstall_agent.ps1 -Path D:\AI\agentY
#>

[CmdletBinding()]
param(
    [switch]$Help,
    [string]$Path = "",
    [string]$ParentDir = "",
    [string]$ComfyUIPath = "",
    [switch]$Yes,
    [switch]$DryRun,
    [switch]$KeepUserData,
    [switch]$NoBackup,
    [switch]$IncludeViewer,
    [switch]$IncludeComfyDesktop,
    [int]$Port = 5000,
    [string]$FromTemp = ""
)

Set-StrictMode -Version 3.0
$ErrorActionPreference = "Stop"
trap { Write-Host ""; Write-Host "  [!]  Unexpected error: $($_.Exception.Message)" -ForegroundColor Red; Write-Host $_.InvocationInfo.PositionMessage -ForegroundColor DarkGray; exit 1 }

function Write-Header  { param([string]$Text) Write-Host ""; Write-Host "===  $Text  ===" -ForegroundColor Cyan }
function Write-Success { param([string]$Text) Write-Host "  [ok] $Text" -ForegroundColor Green }
function Write-Info    { param([string]$Text) Write-Host "  [i]  $Text" -ForegroundColor Yellow }
function Write-Fail    { param([string]$Text) Write-Host "  [!]  $Text" -ForegroundColor Red }
function Write-Line    { param([string]$Text, [string]$Color = "White") Write-Host "       $Text" -ForegroundColor $Color }
function Exit-WithError { param([string]$Message, [int]$Code = 1) Write-Host ""; Write-Fail $Message; exit $Code }

if ($Help) { Get-Help $MyInvocation.MyCommand.Definition -Detailed; exit 0 }

$Script:OnWindows = $true
if (Get-Variable -Name IsWindows -Scope Global -ErrorAction SilentlyContinue) { $Script:OnWindows = [bool]$IsWindows }
if (-not $Script:OnWindows) { Exit-WithError "This is the Windows uninstaller. On macOS use uninstall_agent.sh." 2 }

# -- Native helper: sizes, lock tests and deletion that never follow a link ----------------
if (-not ("AgUn" -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
public static class AgUn {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
    static extern uint GetLongPathName(string s, StringBuilder b, uint n);
    // 8.3 short names (SEBAST~1.ZIL) -> long: two paths compared must be spelled one way.
    public static string Long83(string p) {
        try {
            if (p.Length < 3 || p.IndexOf('~') < 0) return p;
            StringBuilder b = new StringBuilder(32768);
            uint n = GetLongPathName(p, b, (uint)b.Capacity);
            return (n > 0 && n < b.Capacity) ? b.ToString() : p;
        } catch { return p; }
    }
    public static string L(string p) {
        if (p.StartsWith(@"\\?\")) return p;
        if (p.StartsWith(@"\\")) return @"\\?\UNC\" + p.Substring(2);
        return @"\\?\" + p;
    }
    public static bool IsLink(string p) {
        try { return (File.GetAttributes(L(p)) & FileAttributes.ReparsePoint) != 0; } catch { return false; }
    }
    // bytes, files - links are not followed.
    public static long[] Size(string root) {
        long bytes = 0, files = 0;
        if (IsLink(root)) return new long[] { 0, 0 };
        Stack<string> st = new Stack<string>();
        st.Push(L(root));
        while (st.Count > 0) {
            string d = st.Pop();
            try {
                foreach (FileSystemInfo fi in new DirectoryInfo(d).EnumerateFileSystemInfos()) {
                    if ((fi.Attributes & FileAttributes.ReparsePoint) != 0) continue;
                    if ((fi.Attributes & FileAttributes.Directory) != 0) st.Push(fi.FullName);
                    else { files++; try { bytes += ((FileInfo)fi).Length; } catch { } }
                }
            } catch { }
        }
        return new long[] { bytes, files };
    }
    public static string[] Files(string root) {
        List<string> r = new List<string>();
        Stack<string> st = new Stack<string>();
        st.Push(L(root));
        while (st.Count > 0) {
            string d = st.Pop();
            try {
                foreach (FileSystemInfo fi in new DirectoryInfo(d).EnumerateFileSystemInfos()) {
                    if ((fi.Attributes & FileAttributes.ReparsePoint) != 0) continue;
                    if ((fi.Attributes & FileAttributes.Directory) != 0) st.Push(fi.FullName);
                    else r.Add(fi.FullName);
                }
            } catch { }
        }
        return r.ToArray();
    }
    // Programs and libraries a running process holds open cannot be opened for writing.
    public static string[] Locked(string root, int max) {
        List<string> r = new List<string>();
        foreach (string f in Files(root)) {
            string e = Path.GetExtension(f).ToLowerInvariant();
            if (e != ".exe" && e != ".dll" && e != ".pyd" && e != ".node") continue;
            try { using (FileStream s = new FileStream(f, FileMode.Open, FileAccess.ReadWrite, FileShare.None)) { } }
            catch (IOException) { r.Add(f); if (r.Count >= max) break; }
            catch (UnauthorizedAccessException) { }
        }
        return r.ToArray();
    }
    // Removes the tree. A junction or symlink inside is removed as a link, never entered.
    public static string[] DeleteTree(string path) {
        List<string> fails = new List<string>();
        Del(L(path), fails);
        return fails.ToArray();
    }
    static void Del(string p, List<string> fails) {
        FileAttributes a;
        try { a = File.GetAttributes(p); } catch { return; }
        bool isDir = (a & FileAttributes.Directory) != 0;
        bool link = (a & FileAttributes.ReparsePoint) != 0;
        try {
            if (!isDir) {
                if ((a & FileAttributes.ReadOnly) != 0) File.SetAttributes(p, FileAttributes.Normal);
                File.Delete(p);
                return;
            }
            if (link) { Directory.Delete(p); return; }
            foreach (string e in Directory.GetFileSystemEntries(p)) Del(e, fails);
            if ((a & FileAttributes.ReadOnly) != 0) File.SetAttributes(p, FileAttributes.Directory);
            Directory.Delete(p);
        } catch (Exception ex) { fails.Add(p + " : " + ex.Message); }
    }
}
'@
}

# -- Small helpers -------------------------------------------------------------------------
function Format-Size {
    param([double]$Bytes)
    if ($Bytes -ge 1GB) { return ("{0:N1} GB" -f ($Bytes / 1GB)) }
    if ($Bytes -ge 1MB) { return ("{0:N1} MB" -f ($Bytes / 1MB)) }
    if ($Bytes -ge 1KB) { return ("{0:N0} KB" -f ($Bytes / 1KB)) }
    return ("{0:N0} B" -f $Bytes)
}

function Get-FullPath {
    param([string]$P)
    $f = ([IO.Path]::GetFullPath($P)).TrimEnd('\', '/')
    if ($f.Contains('~')) { $f = [AgUn]::Long83($f) }
    return $f
}

function Test-IsLink { param([string]$P) return [AgUn]::IsLink($P) }

function Get-TreeSize {
    param([string]$P)
    if (Test-Path -LiteralPath $P -PathType Leaf) { return [pscustomobject]@{ Bytes = [double](Get-Item -LiteralPath $P -Force).Length; Files = 1 } }
    $r = [AgUn]::Size($P)
    return [pscustomobject]@{ Bytes = [double]$r[0]; Files = [int64]$r[1] }
}

function Read-Answer {
    param([string]$Prompt)
    $v = Read-Host $Prompt
    if ($null -eq $v) { return "" }
    return [string]$v
}

function Confirm-Group {
    # y/N or Y/n. -Yes never reaches here (callers decide first). EOF = the default.
    param([string]$Question, [bool]$Default)
    $hint = if ($Default) { "Y/n" } else { "y/N" }
    $a = (Read-Answer "    $Question [$hint]").Trim().ToLower()
    if (-not $a) { return $Default }
    return ($a -match '^(y|yes)$')
}

function Test-UnderPath {
    # $true when $Child is $Parent or inside it.
    param([string]$Child, [string]$Parent)
    if (-not $Child -or -not $Parent) { return $false }
    $c = (Get-FullPath $Child).ToLowerInvariant(); $p = (Get-FullPath $Parent).ToLowerInvariant()
    return ($c -eq $p) -or $c.StartsWith($p + '\')
}

# -- Identification: git origin first, marker files as the fallback ------------------------
function Get-GitOrigin {
    # $null: not a git checkout. "": a checkout without an origin. Else the origin URL.
    # Read from .git\config directly (no dependency on git, none of its ownership checks).
    param([string]$Dir)
    $g = Join-Path $Dir ".git"
    if (-not (Test-Path -LiteralPath $g)) { return $null }
    if (Test-Path -LiteralPath $g -PathType Container) {
        $cfg = Join-Path $g "config"
        if (Test-Path -LiteralPath $cfg) {
            $in = $false
            foreach ($line in (Get-Content -LiteralPath $cfg -ErrorAction SilentlyContinue)) {
                if ($line -match '^\s*\[') { $in = ($line -match '^\s*\[remote\s+"origin"\]') ; continue }
                if ($in -and $line -match '^\s*url\s*=\s*(.+?)\s*$') { return $Matches[1] }
            }
            return ""
        }
    }
    $o = ""
    if (Get-Command git -ErrorAction SilentlyContinue) {
        $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        try { $o = [string](& git -C $Dir remote get-url origin 2>$null | Select-Object -First 1) } catch { }
        $ErrorActionPreference = $prev
    }
    return $o.Trim()
}

# repo kind -> origin pattern and marker files (all markers of one row must exist).
$Script:Kinds = @{
    agentY  = @{ Origin = 'github\.com[:/]szprivate/agenty(\.git)?/?$'
                 Markers = @(@("requirements.txt", "src\pipeline.py"), @("requirements.txt", "src\agenty_ui_server.py"), @("requirements.txt", "src\chainlit_app.py")) }
    core    = @{ Origin = 'github\.com[:/]szprivate/(agenty-core|agenty_core)(\.git)?/?$'
                 Markers = @(@("pyproject.toml", "agenty_core\__init__.py")) }
    mcp     = @{ Origin = 'github\.com[:/]szprivate/agenty-mcp(\.git)?/?$'
                 Markers = @(@("manifest.json", "claude_desktop_config.example.json"), @("requirements.txt", "agents", "src")) }
    connect = @{ Origin = 'github\.com[:/]szprivate/(agenty-comfyuiconnect|comfyui-agent-canvas)(\.git)?/?$'
                 Markers = @(@("__init__.py", "web\agent_canvas.js"), @("__init__.py", "web\agent_chat.js"), @("__init__.py", "js\agentY_bridge.js")) }
    viewer  = @{ Origin = 'github\.com[:/]szprivate/comfyui-imageviewer(\.git)?/?$'
                 Markers = @() }
    datalayer = @{ Origin = 'github\.com[:/]chainlit/chainlit-datalayer(\.git)?/?$'; Markers = @() }
}

function Test-Identity {
    # Returns @{ Ok; How; Reason }. A foreign git origin is a refusal whatever files the folder holds.
    param([string]$Dir, [string]$Kind)
    $k = $Script:Kinds[$Kind]
    $origin = Get-GitOrigin $Dir
    if ($origin) {
        if ($origin -match $k.Origin) { return @{ Ok = $true; How = "git origin $origin"; Reason = "" } }
        return @{ Ok = $false; How = ""; Reason = "its git origin is '$origin', not an agentY repository" }
    }
    foreach ($row in $k.Markers) {
        $all = $true
        foreach ($m in $row) { if (-not (Test-Path -LiteralPath (Join-Path $Dir $m))) { $all = $false; break } }
        if ($all) { return @{ Ok = $true; How = "marker files ($($row -join ', '))"; Reason = "" } }
    }
    if ($Kind -eq "viewer") {
        $toml = Join-Path $Dir "pyproject.toml"
        if ((Test-Path -LiteralPath $toml) -and ((Get-Content -LiteralPath $toml -Raw -ErrorAction SilentlyContinue) -match 'szprivate/ComfyUI-ImageViewer')) {
            return @{ Ok = $true; How = "pyproject.toml names szprivate/ComfyUI-ImageViewer"; Reason = "" }
        }
    }
    return @{ Ok = $false; How = ""; Reason = "no git origin from szprivate and none of the agentY marker files" }
}

# == ComfyUI discovery (copied from install_agent.ps1 - the installer is not edited) =========
function Test-ComfyUIDir {
    param([string]$P)
    if (-not $P -or -not (Test-Path -LiteralPath $P)) { return $null }
    foreach ($cand in @($P, (Join-Path $P "ComfyUI"), (Join-Path $P "ComfyUI_windows_portable\ComfyUI"))) {
        if (Test-Path -LiteralPath (Join-Path $cand "custom_nodes")) { return (Resolve-Path -LiteralPath $cand).Path }
    }
    return $null
}

function Get-ComfyCandidatePaths {
    $c = @()
    foreach ($drive in @("D:", "E:", "C:")) {
        $c += @("$drive\ai\ComfyUI", "$drive\AI\ComfyUI", "$drive\ComfyUI", "$drive\ai\comfyui__",
                "$drive\ai\ComfyUI_windows_portable\ComfyUI", "$drive\ComfyUI_windows_portable\ComfyUI")
    }
    if ($env:USERPROFILE) {
        $c += @("$env:USERPROFILE\ComfyUI", "$env:USERPROFILE\Documents\ComfyUI", (Join-Path $env:USERPROFILE "ComfyUI-Installs"))
    }
    if ($env:LOCALAPPDATA) { $c += (Join-Path $env:LOCALAPPDATA "Comfy-Desktop\ComfyUI-Installs") }
    if ($env:AGENTY_COMFYUI_DIR) { $c += $env:AGENTY_COMFYUI_DIR }
    return $c
}

function Get-ComfyDesktopBasePaths {
    $out = @()
    if ($env:APPDATA) {
        $cfg = Join-Path $env:APPDATA "ComfyUI\config.json"
        if (Test-Path -LiteralPath $cfg) {
            try {
                $j = Get-Content -LiteralPath $cfg -Raw -ErrorAction Stop | ConvertFrom-Json
                if ($j.PSObject.Properties['basePath'] -and $j.basePath) { $out += [string]$j.basePath }
            } catch { }
        }
    }
    return $out
}

function Find-AllComfyUI {
    param([string]$Hint, $Procs)
    $found = New-Object System.Collections.Generic.List[string]
    $seen = @{}
    $add = {
        param($p)
        $hit = Test-ComfyUIDir $p
        if (-not $hit) { return }
        $key = $hit.TrimEnd('\', '/').ToLowerInvariant()
        if ($seen.ContainsKey($key)) { return }
        $seen[$key] = $true
        $found.Add($hit)
    }
    if ($Hint) { & $add $Hint; return @($found | ForEach-Object { Get-FullPath $_ }) }       # an explicit folder is the only one looked at
    foreach ($c in (Get-ComfyCandidatePaths)) {
        if (-not $c -or -not (Test-Path -LiteralPath $c)) { continue }
        & $add $c
        if ($c -match 'Installs') {
            foreach ($d in (Get-ChildItem -LiteralPath $c -Directory -ErrorAction SilentlyContinue)) { & $add $d.FullName }
        }
    }
    foreach ($b in (Get-ComfyDesktopBasePaths)) { & $add $b }
    foreach ($p in $Procs) {
        $line = [string]$p.CommandLine
        if ($line -notmatch 'comfy') { continue }
        if ($line -match '"([^"]+?)[\\/]main\.py"' -or $line -match '(?:^|\s)((?:[A-Za-z]:)?[\\/][^\s"]*?)[\\/]main\.py') { & $add $Matches[1] }
    }
    return @($found | ForEach-Object { Get-FullPath $_ })
}
# == end ComfyUI discovery ====================================================================

function Get-ProcessList {
    try { return @(Get-CimInstance Win32_Process -ErrorAction Stop) } catch { return @() }
}

function Get-OwnPids {
    # This process and its ancestors: the shell that started the uninstaller, and the copy it re-ran from %TEMP%.
    param($Procs)
    $ids = @{}; $cur = $PID
    for ($i = 0; $i -lt 12 -and $cur; $i++) {
        $ids[[int]$cur] = $true
        $p = $Procs | Where-Object { $_.ProcessId -eq $cur } | Select-Object -First 1
        if (-not $p) { break }
        $cur = $p.ParentProcessId
    }
    return $ids
}

# -- Where is agentY? ----------------------------------------------------------------------
function Find-AgentyInstalls {
    param($ComfyRoots)
    $tiers = @()
    if ($Path) { $tiers += , @($Path, (Join-Path $Path "agentY")) }
    else {
        # Tier 1: where the script is and where the user stands. Tier 2: where the panel says
        # agentY lives. Tier 3: the usual places. The first tier that holds an agentY wins.
        $t1 = @()
        if ($PSScriptRoot) { $t1 += $PSScriptRoot }
        $cur = (Get-Location).Path
        for ($i = 0; $i -lt 6 -and $cur; $i++) { $t1 += $cur; $t1 += (Join-Path $cur "agentY"); $cur = Split-Path -Parent $cur }
        $tiers += , $t1
        $t2 = @()
        foreach ($root in $ComfyRoots) {
            $nodes = Join-Path $root "custom_nodes"
            foreach ($d in (Get-ChildItem -LiteralPath $nodes -Directory -ErrorAction SilentlyContinue)) {
                $hostCfg = Join-Path $d.FullName ".agenty_host.json"
                if (Test-Path -LiteralPath $hostCfg) {
                    try { $j = Get-Content -LiteralPath $hostCfg -Raw | ConvertFrom-Json
                          if ($j.PSObject.Properties['project_root'] -and $j.project_root) { $t2 += [string]$j.project_root } } catch { }
                }
            }
        }
        $tiers += , $t2
        $t3 = @()
        foreach ($drive in @("D:", "E:", "C:")) { foreach ($rel in @("\ai\agentY", "\AI\agentY", "\agentY")) { $t3 += "$drive$rel" } }
        if ($env:USERPROFILE) { foreach ($rel in @("agentY", "Documents\agentY", "source\repos\agentY", "dev\agentY")) { $t3 += (Join-Path $env:USERPROFILE $rel) } }
        $tiers += , $t3
    }
    $seen = @{}; $ok = @(); $refused = @()
    foreach ($tier in $tiers) {
        foreach ($c in $tier) {
            if (-not $c -or -not (Test-Path -LiteralPath $c -PathType Container)) { continue }
            $full = Get-FullPath $c
            $key = $full.ToLowerInvariant()
            if ($seen.ContainsKey($key)) { continue }
            $seen[$key] = $true
            $id = Test-Identity $full "agentY"
            if ($id.Ok) { $ok += [pscustomobject]@{ Dir = $full; How = $id.How } } else { $refused += [pscustomobject]@{ Dir = $full; Reason = $id.Reason } }
        }
        if ($ok.Count) { break }
    }
    return [pscustomobject]@{ Ok = $ok; Refused = $refused }
}

# -- Safety: paths that are never removed --------------------------------------------------
function Get-ForbiddenReason {
    param([string]$P, $ComfyRoots, [string]$Parent)
    $full = Get-FullPath $P
    $root = [IO.Path]::GetPathRoot($full)
    if (-not $root -or ($full + "\").ToLowerInvariant() -eq $root.ToLowerInvariant()) { return "it is a drive root" }
    $sys = @($env:SystemRoot, $env:ProgramFiles, ${env:ProgramFiles(x86)}, $env:ProgramData, (Join-Path $env:SystemDrive "Users"))
    foreach ($s in $sys) { if ($s -and (Test-UnderPath $s $full)) { return "it is, or contains, a Windows system folder ($s)" } }
    if ($env:USERPROFILE -and (Test-UnderPath $env:USERPROFILE $full)) { return "it is, or contains, your user profile" }
    foreach ($c in $ComfyRoots) { if (Test-UnderPath $c $full) { return "it is, or contains, the ComfyUI folder $c" } }
    if ($Parent -and (Test-UnderPath $Parent $full)) { return "it is, or contains, the folder the siblings live in ($Parent)" }
    return $null
}

# == Plan =====================================================================================
$Removed = New-Object System.Collections.Generic.List[string]
$Kept    = New-Object System.Collections.Generic.List[string]
$Failed  = New-Object System.Collections.Generic.List[string]
$Refused = New-Object System.Collections.Generic.List[string]

Write-Host ""
Write-Host "  agentY uninstaller" -ForegroundColor Cyan
if ($DryRun) { Write-Host "  DRY RUN - nothing is asked, stopped or removed." -ForegroundColor Yellow }

$Procs = Get-ProcessList
$ComfyRoots = @(Find-AllComfyUI $ComfyUIPath $Procs)
if ($ComfyUIPath -and -not $ComfyRoots.Count) { Write-Fail "-ComfyUIPath $ComfyUIPath does not look like a ComfyUI install (no custom_nodes\ in it)." }

$found = Find-AgentyInstalls $ComfyRoots
$ProjectRoot = $null
if ($found.Ok.Count -eq 1) { $ProjectRoot = $found.Ok[0].Dir }
elseif ($found.Ok.Count -gt 1) {
    Write-Info "More than one agentY folder was found:"
    for ($i = 0; $i -lt $found.Ok.Count; $i++) { Write-Line ("{0}) {1}" -f ($i + 1), $found.Ok[$i].Dir) }
    if ($Yes -or $DryRun) { Exit-WithError "Pass -Path <folder> to say which one." 2 }
    $n = Read-Answer "    Which one? [1-$($found.Ok.Count)]"
    $k = 0; [void][int]::TryParse($n.Trim(), [ref]$k)
    if ($k -lt 1 -or $k -gt $found.Ok.Count) { Exit-WithError "No choice made - nothing removed." 2 }
    $ProjectRoot = $found.Ok[$k - 1].Dir
}
if (-not $ProjectRoot) {
    foreach ($r in $found.Refused) { if ($Path) { Write-Fail "$($r.Dir): refused - $($r.Reason)" } }
    Exit-WithError "No agentY folder found. Run this from inside it, or pass -Path <the agentY folder>." 2
}
if (-not $ParentDir) { $ParentDir = Split-Path -Parent $ProjectRoot }
$ParentDir = Get-FullPath $ParentDir

# From inside the folder it is about to delete: re-run from a copy in %TEMP% (a running script's
# folder cannot be removed from under it, and the shell may be standing in it).
$selfPath = $PSCommandPath
if ($selfPath -and -not $FromTemp -and -not $DryRun -and (Test-UnderPath $selfPath $ProjectRoot)) {
    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("agenty_uninstall_{0}.ps1" -f $PID)
    Copy-Item -LiteralPath $selfPath -Destination $tmp -Force
    $argv = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $tmp, "-FromTemp", $selfPath, "-Path", $ProjectRoot, "-ParentDir", $ParentDir, "-Port", "$Port")
    if ($ComfyUIPath) { $argv += @("-ComfyUIPath", $ComfyUIPath) }
    foreach ($s in @("Yes", "KeepUserData", "NoBackup", "IncludeViewer", "IncludeComfyDesktop")) { if ($PSBoundParameters.ContainsKey($s)) { $argv += "-$s" } }
    Write-Info "Running from a copy in %TEMP% so this folder can be removed."
    $exe = (Get-Process -Id $PID).Path
    Push-Location ([IO.Path]::GetTempPath())
    try { & $exe @argv; $code = $LASTEXITCODE } finally { Pop-Location }
    Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
    exit $code
}
if ($FromTemp) { try { [Environment]::CurrentDirectory = [IO.Path]::GetTempPath() } catch { } }

Write-Host "  agentY   : $ProjectRoot" -ForegroundColor DarkGray
Write-Host "  siblings : $ParentDir" -ForegroundColor DarkGray
if ($ComfyRoots.Count) { foreach ($c in $ComfyRoots) { Write-Host "  ComfyUI  : $c" -ForegroundColor DarkGray } }
else { Write-Host "  ComfyUI  : none found (the panel and viewer are not looked for)" -ForegroundColor DarkGray }

# One entry per thing that can be removed: Path, Kind (dir|link), Size, Label, Group.
$Items = New-Object System.Collections.Generic.List[object]
function Add-Item {
    param([string]$P, [string]$Group, [string]$Label, [string]$Kind = "dir", [string]$Note = "")
    $link = Test-IsLink $P
    $size = if ($link) { [pscustomobject]@{ Bytes = 0.0; Files = 0 } } else { Get-TreeSize $P }
    $Items.Add([pscustomobject]@{ Path = $P; Group = $Group; Label = $Label; Link = $link; Bytes = $size.Bytes; Files = $size.Files; Note = $Note })
}

function Add-Verified {
    # Identity + forbidden-path checks; refusals are listed, never removed.
    param([string]$P, [string]$Kind, [string]$Group, [string]$Label)
    $full = Get-FullPath $P
    $why = Get-ForbiddenReason $full $ComfyRoots $ParentDir
    if ($why -and $Kind -ne "agentY") { $Refused.Add("$full - refused: $why"); return $false }
    if ($why -and $Kind -eq "agentY") { $Refused.Add("$full - refused: $why"); return $false }
    $isLink = Test-IsLink $full
    $check = $full
    if ($isLink) {
        $t = $null; try { $t = [string](Get-Item -LiteralPath $full -Force).Target } catch { }
        if (-not $t) { $Refused.Add("$full - refused: a link whose target cannot be read"); return $false }
        $check = $t
    }
    $id = Test-Identity $check $Kind
    if (-not $id.Ok) { $Refused.Add("$full - refused: $($id.Reason)"); return $false }
    if ($isLink) { Add-Item $full $Group "$Label (link to $check - only the link is removed)" "link"; return $true }
    Add-Item $full $Group $Label "dir" $id.How
    return $true
}

# Group 2: code and venvs. ---------------------------------------------------------------
$null = Add-Verified $ProjectRoot "agentY" "code" "agentY"
$sibKinds = @()
if (Test-Path -LiteralPath $ParentDir) {
    foreach ($d in (Get-ChildItem -LiteralPath $ParentDir -Directory -Force -ErrorAction SilentlyContinue)) {
        if ((Get-FullPath $d.FullName) -eq $ProjectRoot) { continue }
        $origin = Get-GitOrigin $d.FullName
        $isLinkD = Test-IsLink $d.FullName
        $kind = ""
        if ($d.Name -in @("agentY-core", "agenty_core")) { $kind = "core" }
        elseif ($d.Name -eq "agentY-mcp") { $kind = "mcp" }
        elseif ($origin -and $origin -match $Script:Kinds.core.Origin) { $kind = "core" }
        elseif ($origin -and $origin -match $Script:Kinds.mcp.Origin) { $kind = "mcp" }
        if (-not $kind) { continue }
        if (-not $isLinkD -and $origin -and $origin -notmatch $Script:Kinds[$kind].Origin) {
            $Refused.Add("$($d.FullName) - refused: its git origin is '$origin', not an agentY repository")
            continue
        }
        $lbl = if ($kind -eq "core") { "agentY-core (the tool layer)" } else { "agentY-mcp" }
        $null = Add-Verified $d.FullName $kind "code" $lbl
    }
}

# Group 3: user data inside the repos. ---------------------------------------------------
$DataRel = @("memory", "output", "output_images", "output_workflows", "logs", ".logs", ".files", ".chainlit",
             ".env", ".env.*", "keys*.txt", ".agenty_token*", "config\settings.local.json*", "config\.mcp_tokens",
             "config\fitness_weights.json", "comfyui_workflow_templates_custom\templates\local")
$DataPaths = New-Object System.Collections.Generic.List[string]
foreach ($it in @($Items | Where-Object { $_.Group -eq "code" -and -not $_.Link })) {
    foreach ($rel in $DataRel) {
        $parentOfRel = Split-Path -Parent (Join-Path $it.Path $rel)
        $leaf = Split-Path -Leaf $rel
        if (-not (Test-Path -LiteralPath $parentOfRel)) { continue }
        if ($leaf -match '[*?]') {
            foreach ($m in (Get-ChildItem -LiteralPath $parentOfRel -Filter $leaf -Force -ErrorAction SilentlyContinue)) {
                if ($m.Name -in @(".env_example", ".env.example")) { continue }
                if (-not (Test-IsLink $m.FullName)) { $DataPaths.Add($m.FullName) }
            }
        } else {
            $full = Join-Path $it.Path $rel
            if ((Test-Path -LiteralPath $full) -and -not (Test-IsLink $full)) { $DataPaths.Add($full) }
        }
    }
    $dl = Join-Path $it.Path "chainlit-datalayer"
    if (Test-Path -LiteralPath $dl) {
        $id = Test-Identity $dl "datalayer"
        if ($id.Ok) { $DataPaths.Add($dl) } else { $Refused.Add("$dl - refused: $($id.Reason)") }
    }
}
foreach ($p in ($DataPaths | Select-Object -Unique)) {
    $tracked = $false
    Add-Item $p "data" ("data: " + $p.Substring($ParentDir.Length + 1))
}
$dataKeep = @($Items | Where-Object { $_.Group -eq "data" } | ForEach-Object { $_.Path.ToLowerInvariant() })

# Groups 4 and 5: the panel and the viewer in every ComfyUI found. ----------------------------
$panelNames = @("agentY-comfyuiConnect", "comfyui-agent-canvas", "agentY_bridge")
foreach ($root in $ComfyRoots) {
    $nodes = Join-Path $root "custom_nodes"
    foreach ($d in (Get-ChildItem -LiteralPath $nodes -Directory -Force -ErrorAction SilentlyContinue)) {
        $origin = Get-GitOrigin $d.FullName
        $isPanel = ($d.Name -in $panelNames) -or ($origin -and $origin -match $Script:Kinds.connect.Origin)
        $isViewer = ($origin -and $origin -match $Script:Kinds.viewer.Origin)
        if (-not $isViewer -and -not $origin -and -not $isPanel) {
            $toml = Join-Path $d.FullName "pyproject.toml"
            if ((Test-Path -LiteralPath $toml) -and ((Get-Content -LiteralPath $toml -Raw -ErrorAction SilentlyContinue) -match 'szprivate/ComfyUI-ImageViewer')) { $isViewer = $true }
        }
        if ($isViewer) { $null = Add-Verified $d.FullName "viewer" "viewer" "bEpic viewer ($($d.Name))" }
        elseif ($isPanel) { $null = Add-Verified $d.FullName "connect" "panel" "sidebar panel ($($d.Name))" }
    }
}

# Plan output -----------------------------------------------------------------------------
function Show-Group {
    param([string]$Title, [string]$Group, [string]$Default)
    $its = @($Items | Where-Object { $_.Group -eq $Group })
    Write-Host ""
    Write-Host "  $Title  (default answer: $Default)" -ForegroundColor White
    if (-not $its.Count) { Write-Line "(nothing found)" "DarkGray"; return }
    foreach ($i in $its) {
        $sz = if ($i.Link) { "link" } else { (Format-Size $i.Bytes) }
        if ($Group -eq "code" -and -not $i.Link) {
            $dataIn = 0.0
            foreach ($d in @($Items | Where-Object { $_.Group -eq "data" -and (Test-UnderPath $_.Path $i.Path) })) { $dataIn += $d.Bytes }
            if ($dataIn -gt 0) { $sz = "{0} in all, of which {1} is your data (item 3)" -f (Format-Size $i.Bytes), (Format-Size $dataIn) }
        }
        Write-Line ("{0}  [{1}]" -f $i.Path, $sz) "Gray"
        if ($i.Note) { Write-Line "    verified: $($i.Note)" "DarkGray" }
    }
}

Write-Header "Plan"
$hostProcs = @()
$ownPids = Get-OwnPids $Procs
function Get-HostProcesses {
    $dirs = @($Items | Where-Object { $_.Group -eq "code" -and -not $_.Link } | ForEach-Object { $_.Path })
    $res = @()
    $listen = @{}
    try {
        foreach ($l in (& netstat -ano -p tcp 2>$null)) {
            if ($l -match "^\s*TCP\s+\S+:$Port\s+\S+\s+LISTENING\s+(\d+)") { $listen[[int]$Matches[1]] = $true }
        }
    } catch { }
    $others = @()
    foreach ($p in $Procs) {
        if ($ownPids.ContainsKey([int]$p.ProcessId)) { continue }
        $cl = [string]$p.CommandLine; $exe = [string]$p.ExecutablePath
        $ours = $false
        foreach ($d in $dirs) { if ((Test-UnderPath $exe $d) -or ($cl -and $cl.ToLowerInvariant().Contains($d.ToLowerInvariant()))) { $ours = $true } }
        $isHost = ($cl -match 'agenty_ui_server|run_agent\.(ps1|sh)') -and $ours
        if ($listen.ContainsKey([int]$p.ProcessId)) { if ($ours) { $isHost = $true } else { $others += $p } }
        if ($isHost) { $res += $p }
    }
    return [pscustomobject]@{ Host = $res; Others = $others }
}
$hp = Get-HostProcesses
$hostProcs = @($hp.Host)
Write-Host ""
Write-Host "  1. The running agentY host  (default answer: Y)" -ForegroundColor White
if ($hostProcs.Count) { foreach ($p in $hostProcs) { Write-Line ("PID {0}  {1}" -f $p.ProcessId, $p.Name) "Gray" } }
else { Write-Line "(not running)" "DarkGray" }
foreach ($o in $hp.Others) { Write-Line ("port $Port is used by PID {0} ({1}) - not agentY's, left alone" -f $o.ProcessId, $o.Name) "DarkYellow" }

Show-Group "2. agentY, agentY-core and agentY-mcp: the code and the venvs" "code" "Y"
$codeItems = @($Items | Where-Object { $_.Group -eq "code" })
Show-Group "3. Your data (conversations, memory, settings, .env keys, MCP tokens, output)" "data" "N - asked separately, with a backup zip first"
Show-Group "4. The agentY panel in ComfyUI's custom_nodes" "panel" "Y"
Show-Group "5. The bEpic viewer (works without agentY)" "viewer" "N - kept"
foreach ($r in $Refused) { Write-Host ""; Write-Fail "NOT TOUCHED  $r" }

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$BackupZip = Join-Path $ParentDir "agentY-userdata-backup-$stamp.zip"

Write-Host ""
Write-Host "  6. ComfyUI Desktop  (default answer: N)" -ForegroundColor White
$desktop = $false
if (Get-Command winget -ErrorAction SilentlyContinue) {
    try { & winget list --id Comfy.ComfyUI-Desktop --exact --accept-source-agreements 2>$null | Out-Null; $desktop = ($LASTEXITCODE -eq 0) } catch { }
}
if ($desktop) { Write-Line "installed via winget. Remove it with:  winget uninstall --id Comfy.ComfyUI-Desktop --exact" "Gray" }
else { Write-Line "not installed via winget (nothing to remove)" "DarkGray" }

Write-Host ""
Write-Host "  KEPT, always:" -ForegroundColor White
foreach ($c in $ComfyRoots) { Write-Line "ComfyUI itself and its models: $c" "Gray" }
Write-Line "Ollama, Strata, Blender, and everything outside the folders listed above" "Gray"
Write-Line "uv's cache and the Python builds uv downloaded (see the hints at the end)" "Gray"

# Locks ---------------------------------------------------------------------------------------
function Get-Locks {
    param($Selected)
    $msgs = @()
    $dirs = @($Selected | Where-Object { -not $_.Link } | ForEach-Object { $_.Path })
    $procs = Get-ProcessList
    $own = Get-OwnPids $procs
    foreach ($p in $procs) {
        if ($own.ContainsKey([int]$p.ProcessId)) { continue }
        $exe = [string]$p.ExecutablePath; $cl = [string]$p.CommandLine
        foreach ($d in $dirs) {
            if ($exe -and (Test-UnderPath $exe $d)) { $msgs += ("PID {0} ({1}) runs from {2}" -f $p.ProcessId, $p.Name, $d); break }
        }
        if ($cl -match 'main\.py' -and $cl -match 'comfy') {
            foreach ($root in $ComfyRoots) {
                if ($cl.ToLowerInvariant().Contains($root.ToLowerInvariant()) -and @($Selected | Where-Object { Test-UnderPath $_.Path (Join-Path $root "custom_nodes") }).Count) {
                    $msgs += ("ComfyUI is running (PID {0}) from {1} - close it first" -f $p.ProcessId, $root); break
                }
            }
        }
    }
    foreach ($d in $dirs) {
        foreach ($f in [AgUn]::Locked($d, 5)) { $msgs += "file in use: " + $f.Replace('\\?\', '') }
    }
    return @($msgs | Select-Object -Unique)
}

# Hints (never automatic) ---------------------------------------------------------------------
function Show-Hints {
    Write-Header "Things this script never does for you"
    $cd = if ($env:APPDATA) { Join-Path $env:APPDATA "Claude\claude_desktop_config.json" } else { "" }
    if ($cd -and (Test-Path -LiteralPath $cd)) {
        $hit = (Get-Content -LiteralPath $cd -Raw -ErrorAction SilentlyContinue) -match '(?i)agenty'
        if ($hit) { Write-Info "Claude Desktop still lists agentY-mcp: open $cd and remove the `"agentY`" entry under `"mcpServers`"." }
        else { Write-Info "Claude Desktop config: $cd (no agentY entry found)." }
    } else { Write-Info "If you registered agentY-mcp with Claude Desktop: remove the `"agentY`" key under `"mcpServers`" in %APPDATA%\Claude\claude_desktop_config.json." }
    Write-Info "uv keeps a download cache of every package (shared with your other Python projects). To free it:  uv cache clean"
    Write-Info "The Deadline plugin is removed by its own script, BEFORE agentY is deleted:  python scripts\install_deadline_plugin.py --uninstall"
    if (@($Items | Where-Object { $_.Path -like '*chainlit-datalayer*' }).Count) {
        Write-Info "The old Chainlit data layer ran Postgres in Docker. Stop it and drop its volumes first:  docker compose -f <chainlit-datalayer>\docker-compose.yml down -v"
    }
}

if ($DryRun) {
    $sel = @($Items | Where-Object { $_.Group -in @("code", "data", "panel") })
    Write-Header "Checks"
    $locks = @(Get-Locks $sel)
    if ($locks.Count) { foreach ($l in $locks) { Write-Fail "would block the removal: $l" } } else { Write-Success "nothing holds the folders open" }
    Write-Info "A real run asks before every group. -Yes answers: stop the host, remove agentY + siblings, back up and remove data, remove the panel; keeps the viewer."
    Show-Hints
    Write-Host ""
    Write-Host "  Dry run finished. Nothing was changed." -ForegroundColor Cyan
    exit 0
}

# == Questions ================================================================================
Write-Header "Questions"
$doHost = $false; $doCode = $false; $doData = $false; $doBackup = $false; $doPanel = $false; $doViewer = $false; $doDesktop = $false
$hasCode = (@($Items | Where-Object { $_.Group -eq "code" }).Count -gt 0)
$hasData = (@($Items | Where-Object { $_.Group -eq "data" }).Count -gt 0)
$hasPanel = (@($Items | Where-Object { $_.Group -eq "panel" }).Count -gt 0)
$hasViewer = (@($Items | Where-Object { $_.Group -eq "viewer" }).Count -gt 0)
if ($Yes) {
    $doHost = $true; $doCode = $hasCode; $doData = ($hasData -and -not $KeepUserData); $doBackup = ($doData -and -not $NoBackup)
    $doPanel = $hasPanel; $doViewer = ($hasViewer -and $IncludeViewer); $doDesktop = ($desktop -and $IncludeComfyDesktop)
} else {
    if ($hostProcs.Count) { $doHost = Confirm-Group "1. Stop the running agentY host?" $true }
    if ($hasCode) { $doCode = Confirm-Group "2. Remove agentY, agentY-core and agentY-mcp (code and venvs)?" $true }
    if ($hasData) {
        $doData = Confirm-Group "3. Also DELETE your data (conversations, memory, settings, keys, tokens)? This cannot be undone." $false
        if ($doData -and -not $NoBackup) { $doBackup = Confirm-Group "   Back it up as a zip next to the folder first?" $true }
    }
    if ($hasPanel) { $doPanel = Confirm-Group "4. Remove the agentY panel from ComfyUI's custom_nodes?" $true }
    if ($hasViewer) { $doViewer = Confirm-Group "5. Also remove the bEpic viewer (it works without agentY)?" $false }
    if ($desktop) { $doDesktop = Confirm-Group "6. Run 'winget uninstall' for ComfyUI Desktop now?" $false }
}
if (-not ($doHost -or $doCode -or $doData -or $doPanel -or $doViewer -or $doDesktop)) {
    Write-Info "Nothing selected - nothing changed."
    Show-Hints
    exit 0
}

# Final selection -------------------------------------------------------------------------
$selected = @()
if ($doCode)   { $selected += @($Items | Where-Object { $_.Group -eq "code" }) }
if ($doData -and -not $doCode) { $selected += @($Items | Where-Object { $_.Group -eq "data" }) }
if ($doPanel)  { $selected += @($Items | Where-Object { $_.Group -eq "panel" }) }
if ($doViewer) { $selected += @($Items | Where-Object { $_.Group -eq "viewer" }) }

# 1. Stop the host ------------------------------------------------------------------------
if ($doHost -and $hostProcs.Count) {
    foreach ($p in $hostProcs) { & taskkill /PID $p.ProcessId /T /F 2>$null | Out-Null }
    Start-Sleep -Milliseconds 800
    Write-Success "agentY host stopped"
    $Removed.Add("stopped the running agentY host")
}

# Locks: before anything is removed, so a refusal leaves everything as it was.
if ($selected.Count) {
    $locks = @(Get-Locks $selected)
    if ($locks.Count) {
        Write-Host ""
        foreach ($l in $locks) { Write-Fail $l }
        Write-Host ""
        Exit-WithError "Close what is listed above (ComfyUI, Claude Desktop, a terminal in these folders, the agent window) and run the uninstaller again. NOTHING was removed." 3
    }
}

# Backup ----------------------------------------------------------------------------------
$BackupDone = ""
if ($doData -and $doBackup) {
    Write-Header "Backup"
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $expected = 0; $written = 0
    $fs = $null; $zip = $null
    try {
        $fs = New-Object System.IO.FileStream($BackupZip, [IO.FileMode]::CreateNew)
        $zip = New-Object System.IO.Compression.ZipArchive($fs, [IO.Compression.ZipArchiveMode]::Create)
        foreach ($it in @($Items | Where-Object { $_.Group -eq "data" })) {
            $base = [AgUn]::L($ParentDir) + '\'
            $files = @()
            if (Test-Path -LiteralPath $it.Path -PathType Container) { $files = [AgUn]::Files($it.Path) } else { $files = @([AgUn]::L($it.Path)) }
            foreach ($f in $files) {
                $expected++
                $entry = $f.Substring($base.Length).Replace('\', '/')
                $in = $null
                try {
                    $in = New-Object System.IO.FileStream($f, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
                    $e = $zip.CreateEntry($entry, [IO.Compression.CompressionLevel]::Fastest)
                    $o = $e.Open(); try { $in.CopyTo($o) } finally { $o.Dispose() }
                    $written++
                } catch { Write-Fail "could not back up $entry : $($_.Exception.Message)" }
                finally { if ($in) { $in.Dispose() } }
            }
        }
    } catch { Write-Fail "backup failed: $($_.Exception.Message)" }
    finally { if ($zip) { $zip.Dispose() }; if ($fs) { $fs.Dispose() } }
    if ($expected -eq $written -and (Test-Path -LiteralPath $BackupZip)) {
        $BackupDone = $BackupZip
        Write-Success ("{0} files backed up -> {1} ({2})" -f $written, $BackupZip, (Format-Size (Get-Item -LiteralPath $BackupZip).Length))
    } else {
        Write-Fail "The backup is incomplete ($written of $expected files). Your data is NOT being deleted."
        $doData = $false
        if (-not $doCode) { $selected = @($selected | Where-Object { $_.Group -ne "data" }) }
    }
}

# Removal ---------------------------------------------------------------------------------
Write-Header "Removing"
Set-Location ([IO.Path]::GetTempPath())

function Remove-One {
    param([string]$P, [string]$Label)
    $why = Get-ForbiddenReason $P $ComfyRoots $ParentDir
    if ($why) { $Failed.Add("$P - refused at the last check: $why"); return }
    $fails = @([AgUn]::DeleteTree($P))
    # Everything inside is gone and only the folder itself is held (a terminal stands in it): done.
    if ($fails.Count -eq 1 -and (Test-Path -LiteralPath $P -PathType Container) -and -not @(Get-ChildItem -LiteralPath $P -Force -ErrorAction SilentlyContinue).Count) {
        $Removed.Add("$Label - emptied; the empty folder $P is in use by a terminal (leave it, then rmdir it)")
        Write-Success "removed $Label (the empty folder itself is in use)"
        return
    }
    if ($fails.Count) {
        foreach ($f in $fails) { $Failed.Add($f.Replace('\\?\', '')) }
        Write-Fail "$Label - not completely removed"
    } else { $Removed.Add("$Label  ($P)"); Write-Success "removed $Label" }
}

function Remove-Keeping {
    param([string]$Dir, [string[]]$Keep)
    foreach ($e in [IO.Directory]::GetFileSystemEntries($Dir)) {
        $k = $e.ToLowerInvariant()
        if ($Keep -contains $k) { continue }
        $anc = $false
        foreach ($x in $Keep) { if ($x.StartsWith($k + '\')) { $anc = $true; break } }
        if ($anc -and -not (Test-IsLink $e)) { Remove-Keeping $e $Keep; continue }
        foreach ($f in @([AgUn]::DeleteTree($e))) { $Failed.Add($f.Replace('\\?\', '')) }
    }
}

# Panel and viewer first, then the code, the agentY folder (with this script's copy) last.
foreach ($it in @($selected | Where-Object { $_.Group -in @("panel", "viewer") })) { Remove-One $it.Path $it.Label }

if ($doCode) {
    $codeSel = @($Items | Where-Object { $_.Group -eq "code" })
    # Links first (agenty_core -> agentY-core): the link goes, never the folder through it.
    foreach ($it in @($codeSel | Where-Object { $_.Link })) { Remove-One $it.Path $it.Label }
    foreach ($it in @($codeSel | Where-Object { -not $_.Link -and $_.Path -ne $ProjectRoot })) {
        if ($doData) { Remove-One $it.Path $it.Label }
        else { Remove-Keeping $it.Path $dataKeep; $Removed.Add("$($it.Label) code and venv ($($it.Path))") }
    }
    $self = @($codeSel | Where-Object { $_.Path -eq $ProjectRoot })
    foreach ($it in $self) {
        if ($doData) { Remove-One $it.Path $it.Label }
        else { Remove-Keeping $it.Path $dataKeep; $Removed.Add("agentY code and venv ($($it.Path))") }
    }
} elseif ($doData) {
    foreach ($it in @($Items | Where-Object { $_.Group -eq "data" })) { Remove-One $it.Path $it.Label }
}

if ($doDesktop) {
    Write-Info "winget uninstall --id Comfy.ComfyUI-Desktop --exact"
    & winget uninstall --id Comfy.ComfyUI-Desktop --exact
    if ($LASTEXITCODE -eq 0) { $Removed.Add("ComfyUI Desktop (winget)") } else { $Failed.Add("ComfyUI Desktop - winget returned $LASTEXITCODE") }
}

# Summary ---------------------------------------------------------------------------------
foreach ($it in $Items.ToArray()) {
    $gone = -not (Test-Path -LiteralPath $it.Path)
    if ($gone) { continue }
    if ($it.Group -eq "viewer" -and -not $doViewer) { $Kept.Add("the bEpic viewer - works without agentY: $($it.Path)") }
    elseif ($it.Group -eq "data" -and -not ($doData -and $doCode)) { if (-not $doData) { $Kept.Add("your data: $($it.Path)") } }
    elseif ($it.Group -eq "panel" -and -not $doPanel) { $Kept.Add("the panel: $($it.Path)") }
    elseif ($it.Group -eq "code" -and -not $doCode) { $Kept.Add("$($it.Label): $($it.Path)") }
}
foreach ($c in $ComfyRoots) { $Kept.Add("ComfyUI and its models: $c") }
$Kept.Add("Ollama, Strata, Blender and every folder outside the ones listed")

Write-Header "Summary"
Write-Host "  Removed:" -ForegroundColor Cyan
if ($Removed.Count) { foreach ($r in $Removed) { Write-Line $r "White" } } else { Write-Line "(nothing)" "DarkGray" }
Write-Host "  Kept:" -ForegroundColor Cyan
foreach ($k in ($Kept | Select-Object -Unique)) { Write-Line $k "White" }
if ($Refused.Count) { Write-Host "  Not touched (not verified as agentY's):" -ForegroundColor Cyan; foreach ($r in $Refused) { Write-Line $r "Yellow" } }
if ($BackupDone) { Write-Host "  Backup of your data:" -ForegroundColor Cyan; Write-Line $BackupDone "White" }
if ($Failed.Count) { Write-Host "  Could not remove:" -ForegroundColor Red; foreach ($f in $Failed) { Write-Line $f "Red" } }
Show-Hints
Write-Host ""
if ($Failed.Count) { exit 1 }
exit 0
