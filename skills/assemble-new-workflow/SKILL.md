---
name: assemble-new-workflow
description: Build a ComfyUI workflow from scratch when no template matches (template.name == "build_new"). Derives the node graph from brainbriefing data using comfyui-core pipeline patterns.
allowed-tools: get_workflow_catalog, get_workflow_template, update_workflow, get_node_schema, search_nodes, get_models_in_folder, get_model_types
---

# Assemble New Workflow from Scratch

Activate when `brainbriefing.template.name == "build_new"`. The goal is to construct a valid, fully-wired ComfyUI workflow using the data in `brainbriefing` and the pipeline patterns in the `comfyui-core` skill.

---

## Step 1 — Select a scaffold template

Call `get_workflow_catalog()` and pick the **closest generic template** to the requested task type. Use it as a structural starting point — you will reshape it completely in step 4.

| brainbriefing task type | Preferred scaffold |
|-------------------------|--------------------|
| `image_generation`      | any `txt2img` template |
| `image_edit`            | any `img2img` template |
| `video_i2v`             | any `i2v` or `video` template |
| `video_flf`             | any `flf` or `video` template |
| `video_v2v`             | any `v2v` or `video` template |
| `audio`                 | any `audio` template |
| `3d`                    | any `3d` template |

If no close match exists, pick the simplest `txt2img` template available.

Call `get_workflow_template(scaffold_name)` → record `workflow_path` and the existing node IDs in `io.nodes`.

---

## Step 2 — Plan the node graph

Using the `comfyui-core` pipeline patterns and the brainbriefing, plan:

- **Which scaffold nodes to keep** (checkpoint loader, sampler, output node — anything that fits the target pipeline)
- **Which scaffold nodes to remove** (anything that does not fit the target pipeline)
- **Which new nodes to add** (any node class needed by the target pipeline that is absent from the scaffold)

Map each brainbriefing value to the correct node input:

| brainbriefing field | Target node input |
|---------------------|-------------------|
| `prompt.positive` | `CLIPTextEncode.text` (positive node) |
| `prompt.negative` | `CLIPTextEncode.text` (negative node), or skip if `null` |
| `resolution_width` | `EmptyLatentImage.width` (or equivalent) |
| `resolution_height` | `EmptyLatentImage.height` (or equivalent) |
| `input_nodes[].path` | `LoadImage.image` / `VHS_LoadImagePath.image` (per `input_nodes[].node`) |
| `output_nodes[].output_path` | `SaveImage.filename_prefix` (or equivalent output node) |
| model from `brainbriefing` | `CheckpointLoaderSimple.ckpt_name` (or UNETLoader / CLIPLoader as needed) |

Assign sequential string node IDs (`"1"`, `"2"`, ...) to all nodes you plan to add. Avoid reusing IDs from nodes you plan to remove.

---

## Step 3 — Inspect unfamiliar nodes

For any node class you are not certain about:

- **`get_node_schema(node_class)`** — returns required inputs, types, defaults, and output slots. Use to verify input names and connection indices before wiring.
- **`search_nodes(query)`** — use when you need to find the right `class_type` for a capability (e.g., `"video combine"`, `"load image path"`).

Verify model paths:
- If the brainbriefing does not specify a model file, call `get_models_in_folder(folder)` to list available files and pick the most appropriate one.

---

## Step 4 — Assemble the update_workflow call

Build three arrays:

**`patches`** — set scalar inputs on nodes that already exist in the scaffold (nodes you are keeping):
```json
[
  { "node_id": "<existing_id>", "input_name": "text", "value": "<positive_prompt>" },
  { "node_id": "<existing_id>", "input_name": "ckpt_name", "value": "<model_file>" }
]
```

**`add_nodes`** — full node definitions for every new node (use connection format `["sourceId", outputIndex]` for linked inputs):
```json
[
  {
    "id": "10",
    "class_type": "EmptyLatentImage",
    "inputs": {
      "width": 1024,
      "height": 1024,
      "batch_size": 1
    },
    "_meta": { "title": "Empty Latent" }
  },
  {
    "id": "11",
    "class_type": "KSampler",
    "inputs": {
      "model": ["1", 0],
      "positive": ["2", 0],
      "negative": ["3", 0],
      "latent_image": ["10", 0],
      "seed": 0,
      "steps": 20,
      "cfg": 7.0,
      "sampler_name": "euler",
      "scheduler": "karras",
      "denoise": 1.0
    },
    "_meta": { "title": "KSampler" }
  }
]
```

**`remove_nodes`** — IDs of scaffold nodes that do not belong in the new pipeline:
```json
["5", "8"]
```

Call:
```
update_workflow(workflow_path, patches=<patches_json>, add_nodes=<add_nodes_json>, remove_nodes=<remove_nodes_json>)
```

---

## Step 5 — Handle errors and retry

- If `update_workflow` returns `status: "error"`:
  - Read the error message carefully.
  - Common causes: wrong connection index, missing required input, duplicate node ID, removed node still referenced.
  - Fix the specific issue and call `update_workflow` again immediately.
  - Do not ask the user — retry up to 3 times before reporting with `task_id` and stopping.

- If a required node class is not found by `get_node_schema` (node does not exist in the ComfyUI instance):
  - Report with `task_id` and stop. Do not substitute an incompatible node class.

---

## Rules

- **Never guess model file names** — always verify via `get_models_in_folder` if not listed in `brainbriefing`.
- **Never reuse node IDs** that appear in `remove_nodes` — assign fresh IDs to all added nodes.
- **Always wire output nodes** (`SaveImage`, `VHS_VideoCombine`, etc.) to the final IMAGE/LATENT/AUDIO output of the pipeline. An unconnected output node will cause a validation failure.
- **All connections must be type-safe** — verify output index and type via `get_node_schema` when unsure. See `comfyui-core` references/common-nodes.md for standard node types.
- **`signal_workflow_ready` is NOT called in this skill** — return `workflow_path` to the Brain, which handles handoff per its step 2 constraints.
