````skill
---
name: workflow-templates
description: Browse and select from 186 ComfyUI workflow templates covering text-to-image, img2img, image editing, inpainting, outpainting, ControlNet, video generation (wan, hunyuan, kling, seedance), upscaling, 3D generation, audio, and API-backed pipelines (BFL, ElevenLabs, Kling, ByteDance, Google). Activate this skill whenever you need to pick a workflow template for a user request.
allowed-tools: file_read list_workflow_templates search_workflow_templates get_workflow_template
---

# Workflow Template Selection

Activate this skill when you need to choose a ComfyUI workflow template.

## Step 1 — Read the template registry

Call `file_read` on `./config/workflow_templates.json`.

This file is a flat `{"template_name": "description", ...}` map of all available templates. Read it once and reason over it using your own judgment — do not ask the user to pick a name.

## Step 2 — Pick the best match

Select the template whose description best fits the user's request. Prefer:
- Exact task type (txt2img, img2img, video, 3D, audio, etc.)
- Requested model or API (flux, qwen, wan, kling, elevenlabs, etc.)
- Requested editing mode (inpaint, outpaint, controlnet, style transfer, etc.)

If the registry doesn't give you enough detail to decide, call `list_workflow_templates` (returns name + description) or `search_workflow_templates(query)` to narrow down candidates.

For disambiguation by model or node requirements, call `list_workflow_templates(verbose=True)` — this adds `models` and `requires_custom_nodes` fields per template.

## Step 3 — Load the template

Call `get_workflow_template(template_name)` with the exact name from the registry. This returns the full node-graph JSON plus a node summary.

Proceed to patch and assemble the workflow as instructed in your main system prompt.
````
