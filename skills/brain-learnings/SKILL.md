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

2026-04-21 | KlingVideoNode prompts stored in multi_shot.storyboard_N_prompt fields, not separate nodes | Before patching prompts, read the template structure first to confirm prompt storage location. Prompts are embedded directly in the KlingVideoNode, not separate Text nodes.
2026-04-21 | reference image must be uploaded separately before workflow validation | Call upload_image with file_path and type='input' before signal_workflow_ready to ensure image is available to ComfyUI.
2026-04-21 | multi_shot.storyboard_N_duration should be set as integer 1, not string "1" | When patching duration inputs, use integer value (1) not string representation to match workflow schema expectations.
2026-04-21 | template may have fewer nodes than assumed, causing node not found errors | Always inspect actual workflow structure via read_text_file before removing or patching nodes with assumed IDs.

2026-04-21 | Removing non-existent nodes from workflow caused errors | Verify node IDs exist before removing. Omit remove_nodes if template structure is unverified.
2026-04-21 | Kling3 model node rejects custom pixel resolution strings | Use preset resolution values like '1080p' or '720p' instead of '1024x1024' in model inputs.
2026-04-21 | Workflow inputs failed to reference file until uploaded first | Use upload_image tool to register files before patching workflow inputs with their file paths.

2026-04-21 | LoadImage validation fails with filename only if image not uploaded to ComfyUI inputs | Always upload images to ComfyUI input directory via upload_image before setting custom filenames in LoadImage node; validation will reject filenames referencing local paths that dont exist in the input folder.
2026-04-21 | multi_shot.storyboard_x_duration fields pre-populate with default count (6), must override only needed ones | When setting multi_shot durations for fewer shots than default, patch all fields including unused ones to a minimal value like 0 or keep default; validation may reject partial duration overrides.

2026-04-21 | Kling3_multiShot validation fails until input images are uploaded to ComfyUI input directory | Upload input images to ComfyUI input directory before running update_workflow to ensure validation passes.

2026-04-21 | Input images specified by path are inaccessible to ComfyUI nodes unless uploaded first. | Invoke upload_image tool for each input image path before validation or execution to ensure accessibility.

2026-04-21 | update_workflow returns error status for Kling3_multiShot until input images are uploaded | Upload input images to ComfyUI input directory before calling update_workflow to ensure path validation
