---
name: brain-learnings
description: Auto-populated learnings from past high-iteration problem-solving sessions. Activate this skill when you notice you are making repeated tool calls to solve the same sub-problem, or when the same error keeps appearing. The entries below document past problems and proven solutions — consult them before retrying a failing pattern.
allowed-tools: 
---

# Brain Self-Learnings

> **This file is automatically maintained by the learnings agent.**
> It is appended after any session where the Brain used more than 5 tool calls.
> Do **not** edit the "Learnings Log" section manually.

## When to activate this skill

Activate and consult this skill when you observe any of the following:
- You have already made **3 or more tool calls** attempting to fix the same issue.
- A tool call fails and you are about to retry with the same approach.
- You are uncertain how to proceed and the task feels repetitive.

Scan the learnings log below for entries that match your current situation.
If a matching entry exists, **apply the documented solution directly** instead of re-discovering it.

---

## Learnings Log

<!-- The learnings agent automatically appends new entries below this line. -->
<!-- Format: date | problem summary | solution (1–2 sentences) -->

2026-04-15 | update_workflow fails validation if reference images not in ComfyUI input directory | Upload images using upload_image before patching workflow inputs to ensure files exist for node validation steps.
2026-04-15 | Template aspect_ratio defaults to 'auto' conflicting with specific user ratio requests | Always patch generator node aspect_ratio input to specific value like '16:9' to override 'auto' default.
2026-04-15 | Nano Banana 2 node requires resolution strings like '2K' instead of raw pixel dimensions | Map requested pixel dimensions to standard resolution strings like '2K' before patching generator resolution input.

2026-04-16 | workflow validation fails when reference images not uploaded to ComfyUI input | Always upload reference images before calling update_workflow; images must be in ComfyUI input directory for validation to pass.

2026-04-16 | Template node KlingOmniProImageToVideoNode not found in ComfyUI install | Search for available nodes using search_nodes and use KlingCameraControlI2VNode instead to avoid missing_node_type errors.
2026-04-16 | Unused BatchImagesNode and Note nodes cause validation errors immediately after template load | Remove unused nodes like BatchImagesNode and Note immediately after loading template if they are not required.
2026-04-16 | Kling i2v workflow fails without connecting LoadImage output to reference_images input | Connect LoadImage node output explicitly to the reference_images input of the Kling video generation node to avoid missing input errors.

2026-04-17 | Missing reference_images input caused validation failure on Kling video node | Add explicit reference_images patch linking LoadImage output to the video node input.
2026-04-17 | LoadImage node rejected local file path during validation | Upload input images to ComfyUI input directory before assembling the workflow.
2026-04-17 | Template contained unknown Note node type causing validation failure | Remove unknown node types from template before workflow validation to avoid type errors.

2026-04-17 | ModelSamplingFlux validation fails with missing shift inputs | Always include base_shift (0.5), max_shift (1.15), width, height when patching workflows containing ModelSamplingFlux nodes. Omitting any causes error.
2026-04-17 | Template references custom node class not in ComfyUI installation | Verify all required custom nodes (e.g., LTXVideo extensions) are installed before using templates. Missing node class causes validation error.

2026-04-17 | BatchImagesNode and Note nodes cause validation errors in Kling O3 templates | Always remove BatchImagesNode and Note nodes from Kling O3 templates before validation; they are not needed for single-run workflows and cause required input errors.
2026-04-17 | KlingOmniProImageToVideoNode requires reference_images connection not value assignment | Connect reference_images input to LoadImage output using node ID and slot format, not as a literal value patch to avoid missing input errors.
2026-04-17 | Input image paths in brainbriefing may reference non-existent files | Verify input files exist at specified paths before workflow assembly; do not attempt upload if file not found at brainbriefing location.

2026-04-20 | update_workflow validation fails when images are not in ComfyUI input directory | Upload images to ComfyUI input directory first using upload_image before applying patches with image references. This avoids "image - Invalid" validation errors.

2026-04-20 | BatchImagesNode COMFY_AUTOGROW_V3 rejects array format for 'images' input | Use dotted notation keys (images.image0, images.image1) at top-level inputs instead of array or nested dict wrapper.
2026-04-20 | WAS Image Batch fails with TypeError unhashable type list when inputs have different dimensions | Resize reference image to match master image dimensions before batch processing; use ImageScale node with target width/height from master image.

2026-04-20 | get_workflow_template failed for 'img2img_basic', hint suggests using get_workflow_catalog | Call get_workflow_catalog before attempting to load specific template names to avoid 'not found' assembly errors.
2026-04-20 | search_nodes for 'qwen sampler' and 'qwen infer' failed repeatedly | Use standard samplers like EasyKSampler with QWEN encoders; QWEN inference pipelines do not require unique branded sampler nodes.

2026-04-21 | workflow validation fails with prompt_outputs_failed_validation when input images not in ComfyUI folder | Upload all reference/master images to ComfyUI input directory before calling update_workflow to avoid validation errors.
2026-04-21 | template shows node 23 but patches target node 26 which isn't visible initially | Verify all target node IDs exist in loaded template before applying patches; node IDs may differ from initial template view.

2026-04-21 | Workflow validation fails when referenced images not in ComfyUI input directory | Upload all referenced images to ComfyUI's input directory before patching or validating. Use image_type: 'input' when calling upload_image to ensure proper placement.
