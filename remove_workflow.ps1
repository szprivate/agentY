param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Name,
    [Parameter(Mandatory=$false, Position=1)]
    [string]$IndexPath = ""
)

# Resolve repository root (script is expected to live in repo root)
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-And-RunPython {
    param($pySource, $args)
    $tmp = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), [System.IO.Path]::GetRandomFileName() + '.py')
    $pySource | Out-File -FilePath $tmp -Encoding utf8
    try {
        $cmd = @('python', $tmp) + $args
        $proc = & $cmd 2>&1
        $exit = $LASTEXITCODE
        return @{ Exit = $exit; Output = $proc }
    } finally {
        Remove-Item -Force -ErrorAction SilentlyContinue $tmp
    }
}

# Python snippet to import and call workflow_remove
$py = @"
import sys
from pathlib import Path
# Ensure repo root is on sys.path so 'src' can be imported
sys.path.insert(0, r"$RepoRoot")
from src.utils.workflow_parser import workflow_remove
name = sys.argv[1]
index_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
result = workflow_remove(name, index_path=index_path)
print(result)
"@

$args = @($Name)
if ($IndexPath -ne "") { $args += $IndexPath }

$res = Write-And-RunPython -pySource $py -args $args
if ($res.Exit -ne 0) {
    Write-Error "Python exited with code $($res.Exit). Output:`n$($res.Output -join "`n")"
    exit $res.Exit
} else {
    Write-Output $res.Output
}
