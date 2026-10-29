---
name: video-gemini-motionpromptgeneration
description: "Analyse video input using Gemini, create motion prompts. Activate when brainbriefing template name contains 'video_gemini_motionPromptGeneration'. Patchable: Input video, Output; Locked: Structural nodes, Sampler recipe, and Resolution-driven logic."
allowed-tools: patch_workflow check_model
---

# Video videp_gemini_motionPromptGeneration workflow assembly

Assembles motion prompt generation workflows using Gemini. NEVER analyse a video yourself -- always use this workflow to create a description/prompt from the video, which can then be used for generation. This workflow is designed to extract motion information and scene details from a video input and convert them into a structured prompt format.

---

## Workflow shape

- **Inputs:** video
- **Output:** text file (TXT)

---

## Patchable nodes — the ONLY things you may change

All patches go through `patch_workflow({node_id, input_name, value})`.

### Input images / video

| Node | Type | Input | Notes |
|---|---|---|---|
| `4` | `LoadVideo` (Load First Frame) | `image` | Do not touch node `124` — reads from this automatically. |


### Prompts

| Node | Type | Input | Notes |
|---|---|---|---|
| `1` | `GeminiNode` | `prompt` | 

### Output

| Node | Type | Input | Notes |
|---|---|---|---|
| `3` | `easy saveText`  | `output_file_path` | Change to a matching file path, should match to the filename of the input video.|

Always upload input files via `LoadVideo` before patching.


## Validation checklist

1. Image video uploaded and patched into nodes `4`.

If all pass → `signal_workflow_ready(workflow_path)`.

---

