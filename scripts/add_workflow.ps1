param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$WorkflowFile
)

# Resolve repo root (this script lives in <repo>/scripts/)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

if (-not (Test-Path $WorkflowFile)) {
    Write-Error "Workflow file not found: $WorkflowFile"
    exit 2
}

# Add the workflow to the canonical custom-template corpus (in agenty_core) and
# regenerate the recipe database. This mirrors the /add_workflow slash command:
# it copies the JSON into the templates folder, registers it in index.json,
# writes a description, and rebuilds workflow_recipes.json.
$resolved = (Resolve-Path $WorkflowFile).Path
$py = @"
import json, sys
from pathlib import Path
from src.utils.workflow_admin import register_workflow, format_recipe_counts
p = Path(sys.argv[1])
res = register_workflow(json.loads(p.read_text(encoding='utf-8')), p.stem, source_path=p)
print(f"Added '{res['name']}' - {format_recipe_counts(res['recipes'])}")
if res['description']:
    print('Description: ' + res['description'])
"@

& python -X utf8 -c $py $resolved
exit $LASTEXITCODE
