---
name: NanoBanana
description: Follow these detailed instructions to build and patch NanoBanana-based workflows according to the `brainbriefing` specifications. This skill is responsible for all interactions with NanoBanana templates, including structural changes, node edits, and validation fixes.
allowed-tools: get_workflow_template patch_workflow get_workflow_node_info add_workflow_node remove_workflow_node validate_workflow signal_workflow_ready duplicate_workflow read_text_file
---


# NanoBanana Workflow Assembly Instructions


***GeminiImage2Node / GeminiNanoBanana2 / Nano Banana 2 / Nano Banana Pro template with more than one input image:***

- use `add_workflow_node` to add an `LoadImage` node for every input image
- use `add_workflow_node` to add an `BatchImagesNode`, 
- connect the outputs of all LoadImage nodes to the BatchImagesNode inputs,
- then connect the BatchImagesNode output to the generator's input.


***GeminiImage2Node / GeminiNanoBanana2 / Nano Banana 2 / Nano Banana Pro template with exactly 1 input image and a `BatchImagesNode` already present in the template:***

- remove the `BatchImagesNode` immediately
- wire the LoadImage node directly to the generator's `images` input. 
- Do not attempt to keep or reconfigure it — it requires multiple inputs and will always fail validation with a single image.