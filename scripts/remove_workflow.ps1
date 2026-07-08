param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Name
)

# Resolve repo root (this script lives in <repo>/scripts/)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

if (-not $Name) {
    Write-Error "Template name is required"
    exit 2
}

# Remove the workflow from the canonical custom-template corpus (in agenty_core)
# and regenerate the recipe database. Mirrors the /remove_workflow slash command:
# drops the index.json entry, the template JSON file, its description, and the
# derived skill dir, then rebuilds workflow_recipes.json.
$py = @"
import sys
from src.utils.workflow_admin import remove_workflow, format_recipe_counts
res = remove_workflow(sys.argv[1])
note = '' if res['removed_file'] else ' (no template file was on disk)'
print(f"Removed '{res['name']}'{note} - {format_recipe_counts(res['recipes'])}")
"@

& python -X utf8 -c $py $Name
exit $LASTEXITCODE
