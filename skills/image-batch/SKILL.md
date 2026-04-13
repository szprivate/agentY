---
name: image-batch
description: Use ONLY when count_iter > 1 AND variations is true in the brainbriefing. Generate N distinct variation prompts and write them to multiprompt.json so the pipeline can split them across individual workflow copies.
allowed-tools: write_text_file
---

# image-batch

Activate this skill when the brainbriefing contains **both** `count_iter > 1` **and** `variations: true`.

## Your responsibility

Generate `count_iter` distinct, creative variations if the `prompt` from brainbriefing and write them to `output_workflows/multiprompt.json` using the `write_text_file` tool.


## How to write the file

Call `write_text_file` exactly once:

- `path`: `output_workflows/multiprompt.json`
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