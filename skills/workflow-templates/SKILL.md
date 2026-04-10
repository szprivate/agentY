---
name: workflow-templates
description: Select ComfyUI workflow templates based on user requests. Retrieves metadata about the workflow, such as used models and input / output nodes, to assist with workflow assembly.
allowed-tools: get_workflow_catalog, get_workflow_template
---

# Workflow Template Selection
Activate this skill when you need to choose a ComfyUI workflow template.

1. Call `get_workflow_catalog()`. This returns a flat `{"template_name": "description", ...}` map of all available templates. Read it once.
2. Find the best matching template using this priority order:
   - **Name match first**: normalise the user's phrasing to snake_case and check if a template key contains those words. Example: "Nano Banana Pro API" → `api_nano_banana_pro`. "Kling image to video" → `api_kling_i2v`. This catches most cases even when the user doesn't use the exact name.
   - **Description match**: if no name matches, look into the descriptions in `./config/workflow_templates.json` - see if they mention the user's requested features (e.g. "video interpolation", "uses Runway Gen-2", "text-to-video with LTX") or model names.
   - **Task-type fallback**: if still ambiguous, infer the task type (text-to-image, image-to-video, audio, 3D, etc.) and pick the most capable template for that type.
3. Call `get_workflow_template` with the exact name from the registry. This returns a **summary** (node list, models, I/O metadata) and a **`workflow_path`** pointing to the full workflow JSON on disk.

## Matching tips
- if you don't find a matching template, check if there's a template with a similar name (e.g. "Nano Banana Pro API" → api_nano_banana_pro). DO NEVER INVENT NEW TEMPLATES. 
- Template names encode the task: `t2i` = text-to-image, `i2v` = image-to-video, `t2v` = text-to-video, `flf2v` = first-last-frame-to-video, `v2v` = video-to-video, `s2v` = sound-to-video.
- `api_` prefix = cloud API (no local VRAM needed). No prefix = runs locally. Prefer local unless the user asks for a specific cloud service or speed.
- When multiple templates match, prefer the model family the user mentioned (Wan, Kling, LTX, Flux, Qwen, Runway, etc.).
- If still unsure, pick the closest match and add a WARNING in the brainbriefing.