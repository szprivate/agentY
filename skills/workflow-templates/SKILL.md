---
name: workflow-templates
description: Select ComfyUI workflow templates based on user requests. Retrieves metadata about the workflow, such as used models and input / output nodes, to assist with workflow assembly.
allowed-tools: file_read, get_workflow_template
---

# Workflow Template Selection
Activate this skill when you need to choose a ComfyUI workflow template.

1. Call `file_read` on `./config/workflow_templates.json`. This file is a flat `{"template_name": "description", ...}` map of all available templates. Read it once. 
2. Compare the user intent to the `descriptions` in the file. Find a description that matches the user's request, find the workflow that maps to that description.
3. Call `get_workflow_template(template_name)` with the exact name from the registry. This returns a **summary** (node list, models, I/O metadata) and a **`workflow_path`** pointing to the full workflow JSON on disk.
4. Use `file_read(workflow_path)` to load the full workflow JSON when you need to patch nodes.
5. After patching, use `save_workflow(modified_json)` to persist the changes and get a new file path.
6. Proceed to validate and submit using the file path.