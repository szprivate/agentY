skill
---
name: image-batch
description: Use ONLY when count_iter > 1 AND variations is true in the brainbriefing. Generate N distinct variation prompts and write them to multiprompt.json so the pipeline can split them across individual workflow copies.
allowed-tools: write_text_file
---

# image-batch

Activate this skill when the brainbriefing contains **both** `count_iter > 1` **and** `variations: true`.

## Your responsibility

Generate `count_iter` distinct, creative variations of the core prompt from the brainbriefing
and write them to `output/_workflows/multiprompt.json` using the `write_text_file` tool.

Each prompt should be a meaningfully different creative take on the same subject — vary style,
mood, lighting, composition, camera angle, colour palette, etc. Do **not** simply copy the
original prompt or make trivial word swaps.

## How to write the file

Call `write_text_file` exactly once:

- `path`: `output/_workflows/multiprompt.json`
- `content`: a JSON string with keys `prompt1` ... `promptN` (count equals `count_iter`)

Example for count_iter=3:

```json
{
  "prompt1": "full positive prompt for variation 1",
  "prompt2": "full positive prompt for variation 2",
  "prompt3": "full positive prompt for variation 3"
}
```

The number of keys MUST equal `count_iter`. Keys must be named `prompt1` ... `promptN` in order.

## What happens next (pipeline takes over)

After you write `multiprompt.json` and hand back to the Brain:

1. The **Brain** patches the base workflow with `prompt1` using `positive_prompt_node_id` from
   the brainbriefing, then calls `signal_workflow_ready(base_workflow_path)` **exactly once**.
2. The **pipeline** detects `variations: true`, finds `multiprompt.json`, and automatically:
   - Re-patches the base workflow with `prompt1` (ensuring consistency).
   - Creates one workflow copy per remaining prompt (`prompt2` ... `promptN`), patching the
     positive-prompt node (identified by `positive_prompt_node_id`) in each copy.
   - Passes all N workflows to the executor batch for submission to ComfyUI.
3. `multiprompt.json` is deleted by the pipeline after expansion to prevent bleed-over.

## Important rules

- Use `write_text_file` with `path=output/_workflows/multiprompt.json` — the tool resolves
  this relative to the workspace root automatically.
- Do **not** call `signal_workflow_ready` yourself — that is the Brain's responsibility.
- Do **not** create or duplicate workflow files — the pipeline handles that.
- Prompts must be complete, self-contained positive-prompt strings (no placeholders).