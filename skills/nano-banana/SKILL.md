---
name: nano-banana
description: Whenever researcher selected a Nano Banana / Nano Banana 2 / Nano Banana Pro template: Follow these detailed instructions to build and patch NanoBanana-based workflows according to the `brainbriefing` specifications. 
allowed-tools: add_workflow_node, remove_workflow_node
---


# ALWAYS APPLY: NanoBanana Workflow Assembly Instructions
- **RESOLUTION** set `resolution` in the generator / Nano Banana / Nano Banana Pro / Nano Banana 2 node to `1K` 
- **ASPECT RATIO** set `aspect_ratio` in the generator / Nano Banana / Nano Banana Pro / Nano Banana 2 node to `16:9`


***GeminiImage2Node / GeminiNanoBanana2 / Nano Banana 2 / Nano Banana Pro template WITH MORE THAN ONE INPUT IMAGE:***

- use `add_workflow_node` to add a `LoadImage` node for every input image
- connect the new LoadImage nodes to the AILab_ImageToList node in the template
- in the prompt, ALWAYS refer to the input images as @img1 for the first image, @img2 for the second, etc.


***GeminiImage2Node / GeminiNanoBanana2 / Nano Banana 2 / Nano Banana Pro template with exactly 1 input image and a `BatchImagesNode` already present in the template:***

- wire the LoadImage node directly to the generator's `images` input. 
- Do not attempt to keep or reconfigure it — it requires multiple inputs and will always fail validation with a single image.
- in the prompt, ALWAYS refer to the input images as @img1 for the first image, @img2 for the second, etc.
