---
name: comfyui-core
description: Core ComfyUI knowledge — workflow JSON format, node connection semantics, data types, pipeline patterns, and agentY tool usage. Activate whenever building or patching a workflow.
allowed-tools: get_workflow_catalog, get_workflow_template, update_workflow, signal_workflow_ready, duplicate_workflow, get_node_schema, search_nodes, get_model_types, get_models_in_folder, interrupt_execution, free_memory
---

# ComfyUI Core Knowledge
# Adapted from artokun/comfyui-mcp
# Copyright (c) 2024 Arthur R Longbottom
# MIT License - https://github.com/artokun/comfyui-mcp/LICENSE

> See also: [references/common-nodes.md](references/common-nodes.md) — quick-reference tables for built-in node inputs, outputs, and types.

## Workflow JSON Format (API Format)

ComfyUI workflows are JSON objects mapping **string node IDs** to node definitions:

```json
{
  "1": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": { "ckpt_name": "sd_xl_base_1.0.safetensors" },
    "_meta": { "title": "Load Checkpoint" }
  },
  "2": {
    "class_type": "CLIPTextEncode",
    "inputs": { "text": "a cat", "clip": ["1", 1] },
    "_meta": { "title": "Positive Prompt" }
  }
}
```

### Key Rules

- **Node IDs** are strings of integers (`"1"`, `"2"`, etc.)
- **`class_type`** is the exact Python class name of the node
- **`inputs`** contains both widget values (scalars) and connections (arrays)
- **Connections** use the format `["sourceNodeId", outputIndex]` — a 2-element array where:
  - First element: string node ID of the source node
  - Second element: integer index into the source node's `output` list (0-based)
- **`_meta`** is optional, used for display titles only

### Connection Examples

```json
"model": ["1", 0]       // Connect to node 1's first output (MODEL)
"clip": ["1", 1]        // Connect to node 1's second output (CLIP)
"vae": ["1", 2]         // Connect to node 1's third output (VAE)
"positive": ["2", 0]    // Connect to node 2's first output (CONDITIONING)
"samples": ["5", 0]     // Connect to node 5's first output (LATENT)
"images": ["6", 0]      // Connect to node 6's first output (IMAGE)
```

### Important: API Format vs Web UI Format

- **API format** (what we use): `{ "1": { class_type, inputs }, "2": { ... } }`
- **Web UI format** (saved workflows): `{ "nodes": [...], "links": [...] }` — includes layout positions, visual metadata
- All agentY tools expect and return API format
- Muted/bypassed nodes are preserved with `_meta.mode: "muted"` — inactive but visible for understanding the workflow

---

## Data Types

ComfyUI nodes pass typed data through connections:

| Type | Description | Common Source |
|------|-------------|---------------|
| `MODEL` | Diffusion model weights | CheckpointLoaderSimple (output 0) |
| `CLIP` | Text encoder | CheckpointLoaderSimple (output 1) |
| `VAE` | Variational autoencoder | CheckpointLoaderSimple (output 2) |
| `CONDITIONING` | Encoded text prompt | CLIPTextEncode (output 0) |
| `LATENT` | Latent space tensor | EmptyLatentImage, KSampler, VAEEncode |
| `IMAGE` | Pixel image tensor (BHWC) | VAEDecode, LoadImage |
| `MASK` | Single-channel mask | LoadImage (output 1) |
| `UPSCALE_MODEL` | Upscaling model | UpscaleModelLoader |

---

## Standard Pipeline Patterns

### Text-to-Image (txt2img)

```
CheckpointLoaderSimple → MODEL, CLIP, VAE
  ├─ CLIP → CLIPTextEncode (positive) → CONDITIONING
  ├─ CLIP → CLIPTextEncode (negative) → CONDITIONING
  │
EmptyLatentImage → LATENT
  │
KSampler (model, positive, negative, latent_image) → LATENT
  │
VAEDecode (samples, vae) → IMAGE
  │
SaveImage (images)
```

Node IDs typically: 1=Checkpoint, 2=Positive, 3=Negative, 4=EmptyLatent, 5=KSampler, 6=VAEDecode, 7=SaveImage

### Image-to-Image (img2img)

Same as txt2img but replace `EmptyLatentImage` with:
```
LoadImage → IMAGE
VAEEncode (pixels, vae) → LATENT → KSampler.latent_image
```
Set `KSampler.denoise` to 0.5–0.8 (lower = closer to input image).

### Upscale

```
LoadImage → IMAGE
UpscaleModelLoader → UPSCALE_MODEL
ImageUpscaleWithModel (upscale_model, image) → IMAGE
SaveImage (images)
```

### Inpaint

```
LoadImage (image) → IMAGE → VAEEncode → LATENT
LoadImage (mask) → MASK
SetLatentNoiseMask (samples, mask) → LATENT → KSampler.latent_image
```

---

## agentY Tool Usage Guide

### Workflow Discovery and Loading

- **`get_workflow_catalog()`** — returns a flat `{name: description}` dict of all available templates. Call first when choosing a template.
- **`get_workflow_template(template_name)`** — loads a template by name. Saves the workflow to a file and returns a compact summary with `io.nodes` (patchable inputs) and `workflow_path`. Use `workflow_path` in all subsequent `update_workflow` calls.

### Patching Workflows

**`update_workflow(workflow_path, patches, add_nodes, remove_nodes)`** — atomic patch + validate in one call.

- `patches` — JSON string array of `{ "node_id", "input_name", "value" }` objects:

```json
[
  { "node_id": "4", "input_name": "width",  "value": 1024 },
  { "node_id": "4", "input_name": "height", "value": 1024 },
  { "node_id": "2", "input_name": "text",   "value": "a cat" }
]
```

- `add_nodes` — JSON string array of full node definitions to insert:

```json
[
  {
    "id": "10",
    "class_type": "LoraLoader",
    "inputs": {
      "model": ["1", 0],
      "clip": ["1", 1],
      "lora_name": "my_lora.safetensors",
      "strength_model": 0.8,
      "strength_clip": 0.8
    }
  }
]
```

- `remove_nodes` — JSON string array of node IDs to delete: `'["10", "11"]'`

**If `update_workflow` returns an error:** read the message, fix the patches, and retry immediately.

### Signalling Completion

- **`signal_workflow_ready(workflow_path)`** — call as the **final** tool call once the workflow is fully assembled and validated. This hands off to the Executor. Never call this before all patches are applied.
- **`duplicate_workflow(source_path)`** — create a copy of a workflow with a fresh random seed. Use for batch/variation runs before patching each copy independently.

### Node Inspection

- **`get_node_schema(node_class)`** — returns required/optional inputs, types, defaults, and outputs for a node class. Use when you need to verify connection types or discover required inputs before patching.
- **`search_nodes(query, limit=10)`** — search available nodes by keyword across names, descriptions, and categories. Use when you need to find the right node class name.

### Model Inspection

- **`get_model_types()`** — list available model folder types (checkpoints, loras, unet, vae, clip, etc.).
- **`get_models_in_folder(folder)`** — list model files in a specific folder. Use to verify a model exists before referencing it in a patch.

### Memory and Execution Control

- **`free_memory(unload_models=True, free_memory_flag=True)`** — free GPU/system memory by unloading models and clearing caches. Use if the system is under memory pressure before submitting a workflow.
- **`interrupt_execution()`** — immediately stop the currently running ComfyUI workflow. Use only when explicitly requested or in an error recovery scenario.

---

## KSampler Parameters

| Parameter | Type | Common Values |
|-----------|------|---------------|
| `seed` | int | Any integer; use a fixed value for reproducibility |
| `steps` | int | 20 (standard), 4–8 (turbo/lightning models) |
| `cfg` | float | 7–8 (SD 1.5/SDXL), 1.0 (Flux), 3.5 (turbo) |
| `sampler_name` | string | `"euler"`, `"euler_ancestral"`, `"dpmpp_2m"`, `"dpmpp_sde"` |
| `scheduler` | string | `"normal"`, `"karras"`, `"sgm_uniform"` |
| `denoise` | float | 1.0 (txt2img), 0.5–0.8 (img2img), 0.75–0.9 (inpaint) |

---

## Common Mistakes to Avoid

1. **Wrong connection format**: Use `["1", 0]` not `[1, 0]` — node IDs are strings
2. **Web UI format**: Never pass `{ nodes: [], links: [] }` — use API format only
3. **Missing VAE**: CheckpointLoaderSimple has 3 outputs — MODEL(0), CLIP(1), VAE(2)
4. **Wrong output index**: Verify the node's output order with `get_node_schema` before wiring
5. **Patching unknown node IDs**: Cross-reference node IDs against `io.nodes` from `get_workflow_template` — never guess IDs
6. **Calling `signal_workflow_ready` too early**: All patches must be applied and `update_workflow` must return success before signalling
