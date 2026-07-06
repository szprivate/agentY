---
name: assemble-from-template
description: Assembles a workflow on the basis of a brainbriefing JSON and a pre-selected template.
allowed-tools: apply_brainbriefing, update_workflow, replace_node, get_node_schema, get_workflow_template
---

This skill is used whenever the assemble workflow agent needs to assemble and patch a workflow from a workflow template pre-selected by the query templates agent. Uses the brainbriefing JSON to retrieve input- and output nodes.

**General constraints:**
- Before every tool call, state what you are doing and why.

Follow these steps:

### 1. Load template
Load the workflow template specified in the brainbriefing.
**Constraints:**
- You MUST call `get_workflow_template(brainbriefing.template_name)` and record the returned file path.
- You MUST NOT proceed if the template fails to load — report the error with `task_id` and stop.
- If the template is **`Kling3_multiShot`**: you MUST activate the `kling-multishot` skill and follow its Brain assembly steps instead of the standard step 2 procedure below.
- For a **Nano Banana / Nano Banana Pro** or **z-Image** variant: if a dedicated `nano-banana` / `zimage-turbo` skill is available, activate it for its special handling; otherwise proceed with step 2 — `apply_brainbriefing` assembles these correctly.

### 2. Apply the brainbriefing, then fix any errors
Assemble the workflow by applying the whole brainbriefing in ONE call, then correct only what it could not do mechanically.

**Constraints:**
- You MUST call `apply_brainbriefing(workflow_path, brainbriefing_json)` with the **full** brainbriefing JSON. This single call patches every input node, the positive/negative prompt (into the exact `prompt_nodes` the briefing names, using each node's **real** input slot — e.g. `prompt` for `GeminiNanoBanana2`, `text` for `CLIPTextEncode`), the output node paths, and the resolution. Do **NOT** hand-build these patches yourself, and do NOT assume the prompt input is always `text`.
- You MUST NOT call `save_workflow()` — that is only for building workflows from scratch.
- If `apply_brainbriefing` returns `status: "ready"` (or `"ok"`): the workflow is assembled — inspect its `applied` list, then hand off. Do not re-patch anything it already applied.
- If it returns `status: "error"`: read `problems` and `server_errors` and fix each with the **smallest** correction, then let it re-validate:
  - a wrong/missing node input or value → `update_workflow(workflow_path, patches)`, each patch `{"node_id": "...", "input_name": "...", "value": ...}`. If you are unsure of the exact input name, inspect the node first with `get_node_schema`.
  - a node that must be swapped → `replace_node(...)`.
  Repeat at most twice, then hand off.
- **Excess input nodes**: if `input_image_count` < the number of image-load nodes → remove the extras via `update_workflow(workflow_path, remove_nodes=[...])`.
- **Missing input nodes**: if `input_image_count` > the number of image-load nodes → add them via `update_workflow(workflow_path, add_nodes=[...])`.
- If the workflow has a **ModelSamplingFlux** node and `apply_brainbriefing` left its inputs incomplete: activate the `flux-sampling` skill and set all four required inputs via `update_workflow`.
- If `count_iter > 1` AND `variations == true`: activate the `image-batch` skill first (a `batch_request`: the same template run N times with substituted parameters — the structure does not change).
- If you find a `BatchImagesNode`: call `replace_node(workflow_path, <node_id>, "ImageBatch")` immediately (it preserves all connections).
