---
name: workflow-templates
description: Select ComfyUI workflow templates based on user requests. Retrieves metadata about the workflow, such as used models and input / output nodes, to assist with workflow assembly.
allowed-tools: file_read, get_workflow_template
---

# Workflow Template Selection
Activate this skill when you need to choose a ComfyUI workflow template.

1. Call `file_read` on `./config/workflow_templates.json`. This file is a flat `{"template_name": "description", ...}` map of all available templates. Read it once. 
2. Compare the user intent to the `descriptions` in the file. Find a description that matches the user's request, find the workflow that maps to that description.
3. Call `get_workflow_template(template_name)` with the exact name from the registry. This returns the workflow, and metadata about the template (workflow name, used models, input and output nodes).
4. Proceed to patch and assemble the workflow as instructed in your main system prompt.