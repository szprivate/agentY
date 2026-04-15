---
name: flux-sampling
description: ModelSamplingFlux node patch requirements. Activate in Brain step 2 (Patch and validate) whenever the workflow contains a ModelSamplingFlux node.
allowed-tools: update_workflow
---

# Flux Sampling â€” ModelSamplingFlux Patch Requirements

When a workflow contains a `ModelSamplingFlux` node, all **four inputs** are required. Omitting any one will cause a ComfyUI validation failure.

---

## Required inputs

| input_name   | required value                                         |
|--------------|--------------------------------------------------------|
| `max_shift`  | `1.15`                                                 |
| `base_shift` | `0.5`                                                  |
| `width`      | from `brainbriefing.resolution_width`                  |
| `height`     | from `brainbriefing.resolution_height`                 |

---

## Patch template

Include these four patches in the `patches` array passed to `update_workflow`:

```json
[
  { "node_id": "<ModelSamplingFlux_node_id>", "input_name": "max_shift",  "value": 1.15 },
  { "node_id": "<ModelSamplingFlux_node_id>", "input_name": "base_shift", "value": 0.5  },
  { "node_id": "<ModelSamplingFlux_node_id>", "input_name": "width",      "value": <resolution_width>  },
  { "node_id": "<ModelSamplingFlux_node_id>", "input_name": "height",     "value": <resolution_height> }
]
```

Replace `<ModelSamplingFlux_node_id>` with the actual node ID from the workflow template.  
Replace `<resolution_width>` and `<resolution_height>` with the integer values from the brainbriefing.

---

## How to find the node ID

The `ModelSamplingFlux` node is typically listed in the `io.nodes` array returned by `get_workflow_template`. Look for `class_type: "ModelSamplingFlux"` and note its `nodeId`.

If it is not in `io.nodes`, inspect the full workflow JSON for a node with `class_type: "ModelSamplingFlux"`.

---

## Rules

- You MUST include all four inputs in every `update_workflow` call that patches a ModelSamplingFlux node.
- Values are **not optional** â€” ComfyUI will reject the workflow if any of the four are absent.
- `width` and `height` MUST come from `brainbriefing.resolution_width` / `brainbriefing.resolution_height` â€” never hard-code or guess them.
