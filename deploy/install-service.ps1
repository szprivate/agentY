# install-service.ps1 - run agentY as a Windows service with NSSM (https://nssm.cc).
#
# You run this, once, in an elevated PowerShell on the server. It does not start
# anything on its own and changes nothing but the service it creates.
#
#   .\deploy\install-service.ps1 -EnvFile C:\agenty\agenty.env
#   .\deploy\install-service.ps1 -EnvFile C:\agenty\agenty.env -Port 5000 -Remove   # uninstall
#
# EnvFile: KEY=value lines (see deploy\agenty.env.example). The service gets those
# as its environment; API keys can stay in the checkout's .env as usual.
#
# The service runs the same host the launcher starts:
#   <checkout>\.venv\Scripts\python.exe -m src.agenty_ui_server --port <Port>
# so run_agent.ps1's pre-flight (update check, model refresh, dependency sync) is
# NOT repeated at every service start. Run run_agent.ps1 by hand once after an
# update, or add `-NoUpdate` knowledge to your own deployment routine.
param(
    [string]$EnvFile = "",
    [int]$Port = 5000,
    [string]$ServiceName = "agentY",
    [string]$Nssm = "nssm",
    [switch]$Remove
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition)
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Get-Command $Nssm -ErrorAction SilentlyContinue)) {
    throw "nssm not found. Install it (winget install nssm, or https://nssm.cc) or pass -Nssm <path>."
}

if ($Remove) {
    & $Nssm stop $ServiceName confirm 2>$null
    & $Nssm remove $ServiceName confirm
    Write-Host "Service $ServiceName removed."
    exit 0
}

if (-not (Test-Path $Python)) { throw "No virtualenv at $Python. Run install_agent.ps1 first." }

$extra = @()
if ($EnvFile) {
    if (-not (Test-Path $EnvFile)) { throw "EnvFile not found: $EnvFile" }
    foreach ($line in Get-Content $EnvFile) {
        $t = $line.Trim()
        if ($t -eq "" -or $t.StartsWith("#")) { continue }
        if ($t -notmatch "^[A-Za-z_][A-Za-z0-9_]*=") { Write-Warning "skipped: $t"; continue }
        $extra += $t
    }
}

$logs = Join-Path $Root ".logs"
New-Item -ItemType Directory -Force $logs | Out-Null

& $Nssm install $ServiceName $Python "-m src.agenty_ui_server --port $Port"
& $Nssm set $ServiceName AppDirectory $Root
& $Nssm set $ServiceName DisplayName "agentY host"
& $Nssm set $ServiceName Description "agentY chat host (Slack-first server)"
& $Nssm set $ServiceName Start SERVICE_AUTO_START
& $Nssm set $ServiceName AppStdout (Join-Path $logs "service.out.log")
& $Nssm set $ServiceName AppStderr (Join-Path $logs "service.err.log")
& $Nssm set $ServiceName AppRotateFiles 1
& $Nssm set $ServiceName AppRotateBytes 10485760
# Restart after a crash, but not in a tight loop.
& $Nssm set $ServiceName AppExit Default Restart
& $Nssm set $ServiceName AppRestartDelay 5000
# Ctrl+C first (the host drains and stops cleanly), then kill after 20 s.
& $Nssm set $ServiceName AppStopMethodConsole 20000
if ($extra.Count -gt 0) {
    & $Nssm set $ServiceName AppEnvironmentExtra @extra
}

Write-Host ""
Write-Host "Service '$ServiceName' installed (not started)." -ForegroundColor Green
Write-Host "  Start:   nssm start $ServiceName"
Write-Host "  Status:  nssm status $ServiceName    and    curl http://127.0.0.1:$Port/agentY/health?deep=1"
Write-Host "  Logs:    $logs\service.out.log"
Write-Host "Run the service as a normal account (nssm edit $ServiceName > Log on), not LocalSystem,"
Write-Host "so it can reach the shared storage with that account's rights."
