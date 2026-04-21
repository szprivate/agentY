# Moves JSON templates from comfyui_workflow_templates_official/templates
# to comfyui_workflow_templates_official_deprecated if not listed in config/workflow_templates.json

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
# repository root is parent of scripts directory
$repoRoot = Resolve-Path (Join-Path $scriptDir '..')

$configPath = Join-Path $repoRoot "config\workflow_templates.json"
$templatesDir = Join-Path $repoRoot "comfyui_workflow_templates_official\templates"
$deprecatedDir = Join-Path $repoRoot "comfyui_workflow_templates_official_deprecated"

if (-not (Test-Path $configPath)) { Write-Error "Config not found: $configPath"; exit 1 }
if (-not (Test-Path $templatesDir)) { Write-Error "Templates dir not found: $templatesDir"; exit 1 }
if (-not (Test-Path $deprecatedDir)) { New-Item -ItemType Directory -Path $deprecatedDir | Out-Null }

$allowed = (Get-Content -Raw -Path $configPath | ConvertFrom-Json) | Get-Member -MemberType NoteProperty | Select-Object -ExpandProperty Name

$moved = @()
$skipped = @()

Get-ChildItem -Path $templatesDir -Filter *.json | ForEach-Object {
    $file = $_
    $base = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
    if ($allowed -contains $base) {
        $skipped += $file.Name
    } else {
        $dest = Join-Path $deprecatedDir $file.Name
        Move-Item -Path $file.FullName -Destination $dest -Force
        $moved += $file.Name
    }
}

Write-Output "Moved files: $($moved.Count)"
if ($moved.Count -gt 0) { $moved | ForEach-Object { Write-Output " - $_" } }
Write-Output "Kept files: $($skipped.Count)"
if ($skipped.Count -gt 0) { $skipped | ForEach-Object { Write-Output " - $_" } }

# Save log
$log = @{ moved = $moved; kept = $skipped; timestamp = (Get-Date).ToString("o") }
$logPath = Join-Path $deprecatedDir "move_log_$(Get-Date -Format yyyyMMdd_HHmmss).json"
$log | ConvertTo-Json | Out-File -FilePath $logPath -Encoding UTF8

Write-Output "Log saved to: $logPath"
