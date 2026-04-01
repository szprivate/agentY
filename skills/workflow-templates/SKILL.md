---
name: workflow-templates
description: Select ComfyUI workflow templates based on user requests. Read the template registry, pick the best match, and load the template for execution.
allowed-tools: file_read list_workflow_templates search_workflow_templates get_workflow_template
---

# Workflow Template Selection

Activate this skill when you need to choose a ComfyUI workflow template.

## Step 1 — Read the template registry
Call `file_read` on `./config/workflow_templates.json`. This file is a flat `{"template_name": "description", ...}` map of all available templates. Read it once. 

## Step 2 — Pick the best match
Compare the user intent to the `descriptions` in the file. Find a description that matches the user's request, find the workflow that maps to that description.

## Step 3 — Load the template
Call `get_workflow_template(template_name)` with the exact name from the registry. This returns the full node-graph JSON plus a node summary.

Proceed to patch and assemble the workflow as instructed in your main system prompt.