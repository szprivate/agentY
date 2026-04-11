param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$WorkflowFile,
    [Parameter(Mandatory=$false, Position=1)]
    [string]$IndexPath = ""
)

# Run from the repository root so relative config paths resolve correctly.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (-not (Test-Path $WorkflowFile)) {
    Write-Error "Workflow file not found: $WorkflowFile"
    exit 2
}

$pyArgs = @('-m', 'src.utils.workflow_parser', $WorkflowFile)
if ($IndexPath -ne "") {
    $pyArgs += '--index-path'
    $pyArgs += $IndexPath
}

Write-Host "Running: python $($pyArgs -join ' ')"
& python @pyArgs
$exit = $LASTEXITCODE
if ($exit -ne 0) {
    Write-Error "add_workflow.ps1: parser exited with code $exit"
}
exit $exit
