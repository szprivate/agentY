#!/usr/bin/env pwsh
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$templatesDir = Join-Path $scriptDir 'comfyui_workflow_templates_custom\templates'

if (-not (Test-Path -Path $templatesDir)) {
    Write-Error "Templates directory not found: $templatesDir"
    exit 1
}

Get-ChildItem -Path $templatesDir -File | Where-Object { $_.Name -notmatch '^index' } | ForEach-Object {
    $template = $_.FullName
    Write-Host "Adding workflow from: $template"
    & "$scriptDir\add_workflow.ps1" $template
    if (-not $?) {
        Write-Warning "add_workflow.ps1 failed for: $template"
    }
}

Write-Host 'All templates processed.'
