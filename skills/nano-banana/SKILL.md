---
name: nano-banana
description: Whenever researcher selected a Nano Banana / Nano Banana 2 / Nano Banana Pro template: Follow these detailed instructions to build and patch NanoBanana-based workflows according to the `brainbriefing` specifications. 
allowed-tools: add_workflow_node, remove_workflow_node
---


# ALWAYS APPLY: NanoBanana Workflow Assembly Instructions
- **RESOLUTION** set `resolution` in the generator / Nano Banana / Nano Banana Pro / Nano Banana 2 node to `1K` 


***GeminiImage2Node / GeminiNanoBanana2 / Nano Banana 2 / Nano Banana Pro template WITH MORE THAN ONE INPUT IMAGE:***

- use `add_workflow_node` to add a `LoadImage` node for every input image
- use `add_workflow_node` to add a `BatchImagesNode`, 
- use this JSON schema to connect
- IMPORTANT! connect the output of each LoadImage node to the correct BatchImagesNode inputs, use this JSON schema to connect:
```json
{
    "node_id_BatchImagesNode": {
    "inputs": {
      "images.image0": [
        "node_id_LoadImage1",
        0
      ],
      "images.image1": [
        "node_id_LoadImage2",
        0
      ],
      "images.image2": [
        "node_id_LoadImage3",
        0
      ]
    },
  }, 
}
```

- then connect the BatchImagesNode output to the generator's input.


***GeminiImage2Node / GeminiNanoBanana2 / Nano Banana 2 / Nano Banana Pro template with exactly 1 input image and a `BatchImagesNode` already present in the template:***

- remove the `BatchImagesNode` using `remove_workflow_node`
- wire the LoadImage node directly to the generator's `images` input. 
- Do not attempt to keep or reconfigure it — it requires multiple inputs and will always fail validation with a single image.