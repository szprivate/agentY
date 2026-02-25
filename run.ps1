# run.ps1
# Utility script for launching the agent.  It verifies that the Ollama
# server and ComfyUI are running before handing control to the Python driver.

function Start-IfNotRunning {
    param(
        [string]$ProcessName,
        [ScriptBlock]$StartAction
    )

    $p = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
    if (-not $p) {
        Write-Host "[$ProcessName] not running, starting..." -ForegroundColor Cyan
        & $StartAction
        # give the process a moment to come up
        Start-Sleep -Seconds 2
    }
    else {
        Write-Host "[$ProcessName] already running (PID=$($p.Id))." -ForegroundColor Green
    }
}

# check comfyui: look for the default web port or process name
function Test-ComfyUI {
    # first, see if any process explicitly named "comfyui" exists
    $p = Get-Process -Name "comfyui" -ErrorAction SilentlyContinue
    if ($p) { return $true }

    # if not, check for a process listening on the typical port
    try {
        $listening = Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue
        if ($listening) { return $true }
    } catch {
        # older systems may not have Get-NetTCPConnection; ignore
    }

    # as a last resort inspect command line for "comfyui"
    $py = Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.Path -and (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)").CommandLine -match "comfyui" }
    return $py -ne $null
}

if (-not (Test-ComfyUI)) {
    Write-Warning "ComfyUI does not appear to be running.  Please start it if needed."
} else {
    Write-Host "ComfyUI is running." -ForegroundColor Green
}

# ensure ollama server is running - start it if missing
Start-IfNotRunning -ProcessName "ollama" -StartAction { Start-Process -NoNewWindow -FilePath "ollama" -ArgumentList "serve" }

# finally invoke the Python entrypoint
Write-Host "Launching agent" -ForegroundColor Yellow
python .\src\main.py