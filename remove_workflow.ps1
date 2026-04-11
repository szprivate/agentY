param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Name,
    [Parameter(Mandatory=$false, Position=1)]
    [string]$IndexPath = ""
)

# Run from the repository root so relative config paths resolve correctly.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (-not $Name) {
    Write-Error "Template name is required"
    exit 2
}

# Create a temporary python file that will call workflow_remove
$tmp = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), [System.IO.Path]::GetRandomFileName() + '.py')
$RepoRoot = $ScriptDir -replace '\\','\\\\'
$py = @"
import sys
from pathlib import Path
sys.path.insert(0, r"$ScriptDir")
from src.utils.workflow_parser import workflow_remove
name = sys.argv[1]
index_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
result = workflow_remove(name, index_path=index_path)
print(result)
"@

$py | Out-File -FilePath $tmp -Encoding utf8

$pyArgs = @($tmp, $Name)
if ($IndexPath -ne "") { $pyArgs += $IndexPath }

Write-Host "Running: python $($pyArgs -join ' ')"
& python @pyArgs
$exit = $LASTEXITCODE
Remove-Item -Force -ErrorAction SilentlyContinue $tmp
if ($exit -ne 0) {
    Write-Error "remove_workflow.ps1: python exited with code $exit"
}
exit $exit
