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
2026-04-15 | Nano Banana 2 node requires resolution strings like '2K' instead of raw pixel dimensions | Map requested pixel dimensions to standard resolution strings like '2K' before patching generator resolution input.
2026-04-17 | Template contained unknown Note node type causing validation failure | Remove unknown node types from template before workflow validation to avoid type errors.

2026-04-17 | ModelSamplingFlux validation fails with missing shift inputs | Always include base_shift (0.5), max_shift (1.15), width, height when patching workflows containing ModelSamplingFlux nodes. Omitting any causes error.
2026-04-17 | Template references custom node class not in ComfyUI installation | Verify all required custom nodes (e.g., LTXVideo extensions) are installed before using templates. Missing node class causes validation error.

2026-04-17 | BatchImagesNode and Note nodes cause validation errors in Kling O3 templates | Always remove BatchImagesNode and Note nodes from Kling O3 templates before validation; they are not needed for single-run workflows and cause required input errors.
2026-04-17 | KlingOmniProImageToVideoNode requires reference_images connection not value assignment | Connect reference_images input to LoadImage output using node ID and slot format, not as a literal value patch to avoid missing input errors.
2026-04-17 | Input image paths in brainbriefing may reference non-existent files | Verify input files exist at specified paths before workflow assembly; do not attempt upload if file not found at brainbriefing location.

2026-04-20 | BatchImagesNode COMFY_AUTOGROW_V3 rejects array format for 'images' input | Use dotted notation keys (images.image0, images.image1) at top-level inputs instead of array or nested dict wrapper.
2026-04-20 | WAS Image Batch fails with TypeError unhashable type list when inputs have different dimensions | Resize reference image to match master image dimensions before batch processing; use ImageScale node with target width/height from master image.

2026-04-20 | get_workflow_template failed for 'img2img_basic', hint suggests using get_workflow_catalog | Call get_workflow_catalog before attempting to load specific template names to avoid 'not found' assembly errors.
2026-04-20 | search_nodes for 'qwen sampler' and 'qwen infer' failed repeatedly | Use standard samplers like EasyKSampler with QWEN encoders; QWEN inference pipelines do not require unique branded sampler nodes.

2026-04-21 | LoadImage validation fails with filename only if image not uploaded to ComfyUI inputs | Always upload images to ComfyUI input directory via upload_image before setting custom filenames in LoadImage node; validation will reject filenames referencing local paths that dont exist in the input folder.
2026-04-21 | KlingVideoNode prompts stored in multi_shot.storyboard_N_prompt fields, not separate nodes | Before patching prompts, read the template structure first to confirm prompt storage location. Prompts are embedded directly in the KlingVideoNode, not separate Text nodes.
2026-04-21 | multi_shot.storyboard_N_duration should be set as integer 1, not string "1" | When patching duration inputs, use integer value (1) not string representation to match workflow schema expectations.
2026-04-21 | multi_shot.storyboard_x_duration fields pre-populate with default count (6), must override only needed ones | When setting multi_shot durations for fewer shots than default, patch all fields including unused ones to a minimal value like 0 or keep default; validation may reject partial duration overrides.
2026-04-21 | template may have fewer nodes than assumed, causing node not found errors | Always inspect actual workflow structure via read_text_file before removing or patching nodes with assumed IDs.
2026-04-21 | Kling3 model node rejects custom pixel resolution strings | Use preset resolution values like '1080p' or '720p' instead of '1024x1024' in model inputs.

2026-04-23 | update_workflow fails validation after patching with prompt_outputs_failed_validation error | When validation fails after patching, persist workflow JSON via write_text_file then call update_workflow with empty patches to reset state.
2026-04-23 | GeminiNanoBanana2 images input requires mixed format: string for loaded image, int for generation index | Format images as [\"node_id_string\", 0] for image-to-generate pattern. Avoid pure string or pure int arrays to prevent validation KeyError.
2026-04-23 | Single-image workflows require no multi-image batching nodes | For single input image, wire LoadImage directly to generator node. Remove ImageToList and ImageListToImageBatch nodes that are unnecessary.
2026-04-26 | Workflow failed to signal ready due to incorrect LoadImage node input paths | Ensure correct paths to input images are added to LoadImage nodes before signaling workflow ready to avoid errors.

2026-04-26 | Validator flags missing ModelSamplingFlux inputs for ModelSamplingAuraFlow workflow nodes | Include required Flux inputs (base_shift, max_shift, width, height) in update_workflow patches to satisfy the validator error.
