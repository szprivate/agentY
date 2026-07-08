---
name: assemble-workflow-learnings
description: Auto-populated learnings from past workflow-assembly sessions. Activate this skill when assembling or patching a ComfyUI workflow, especially if you notice repeated tool calls to fix the same assembly sub-problem or the same validation error recurring. The entries below document past problems and proven solutions — consult them before retrying a failing pattern.
allowed-tools: 
---

2026-07-06 | LoadImage node cannot load image files from subdirectories in ComfyUI | LoadImage requires filenames at the root level of the input directory. Files in subdirectories (e.g., agent/, 3d/) will fail validation. Use root-level filenames only.
2026-07-06 | Brainbriefing specifies nonexistent placeholder.png causing workflow validation to fail | Verify that input image files referenced in brainbriefing actually exist in ComfyUI input directory before assembling workflow. Use get_comfyui_dirs and file_read to validate file existence.
2026-07-06 | signal_workflow_ready succeeds but downstream validation still reports invalid image error | signal_workflow_ready queues the workflow even if validation errors exist in apply_brainbriefing result. Ensure workflow passes validation before calling signal_workflow_ready to prevent failed submissions.
