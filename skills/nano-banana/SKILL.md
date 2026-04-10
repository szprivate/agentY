---
name: nano-banana
description: Follow these detailed instructions to build and patch NanoBanana-based workflows according to the `brainbriefing` specifications. This skill is responsible for all interactions with NanoBanana templates, including structural changes, node edits, and validation fixes.
allowed-tools: add_workflow_node, remove_workflow_node
---


# ALWAYS APPLY: NanoBanana Workflow Assembly Instructions
- unless requested otherwise by the user, ALWAYS set `resolution` in the generator node to "1K" 


***GeminiImage2Node / GeminiNanoBanana2 / Nano Banana 2 / Nano Banana Pro template WITH MORE THAN ONE INPUT IMAGE:***

- use `add_workflow_node` to add a `LoadImage` node for every input image
- use `add_workflow_node` to add a `BatchImagesNode`, 
- connect the output of each LoadImage node to the correct BatchImagesNode inputs (output of 1st LoadImage → `images.image0` input of BatchImagesNode, output of 2nd LoadImage → `images.image1` input of BatchImagesNode, etc.). DONT CONNECT AN ARRAY INPUT TO THE BATCHIMAGESNODE (e.g. `images`) — ALWAYS CONNECT TO THE EXPLICITLY NUMBERED INPUTS (e.g. `images.image0`, `images.image1` etc.)
- then connect the BatchImagesNode output to the generator's input.


***GeminiImage2Node / GeminiNanoBanana2 / Nano Banana 2 / Nano Banana Pro template with exactly 1 input image and a `BatchImagesNode` already present in the template:***

- remove the `BatchImagesNode` using `remove_workflow_node`
- wire the LoadImage node directly to the generator's `images` input. 
- Do not attempt to keep or reconfigure it — it requires multiple inputs and will always fail validation with a single image.