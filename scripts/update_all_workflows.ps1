#!/usr/bin/env pwsh
# Rebuild the custom-template index.json for every workflow already present in
# the canonical corpus templates folder (in agenty_core), then regenerate the
# recipe database once. Use after dropping template JSON files into the folder
# directly. Anchors on the canonical corpus, not a per-app path.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot  = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

$py = @"
from src.utils.workflow_admin import reindex_all, format_recipe_counts
res = reindex_all()
print(f"Reindexed {len(res['registered'])} template(s) - {format_recipe_counts(res['recipes'])}")
if res['failed']:
    print('Failed: ' + ', '.join(res['failed']))
"@

& python -X utf8 -c $py
exit $LASTEXITCODE
