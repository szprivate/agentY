---
name: assemble-new-workflow
description: Build a ComfyUI workflow from scratch when no template matches (template.name == "build_new"). Builds to the standard in the recipe database (task -> model -> node clusters), starting from a recipe member template.
allowed-tools: get_workflow_recipe, list_workflow_recipes, get_workflow_catalog, get_workflow_template, update_workflow, get_node_schema, search_nodes
---

# Assemble New Workflow from Scratch

Activate when `brainbriefing.template.name == "build_new"`. Build the workflow to the **standard described by the recipe database** (`task -> model -> node clusters`). The recipe tells you which nodes are required, how they connect, the boundary ports, and which existing templates implement this task+model so you can start from one rather than from nothing.

**Trust the recipe — never refuse or substitute.** The recipe's `required_nodes` are the correct, **standard** ComfyUI node classes for this exact task+model, and its `node_defaults` are the template-verified widget params. Every model in the recipe DB (Qwen Image, Z-Image, Flux, LTX, WAN, ...) runs on these standard nodes — do NOT claim it "needs a custom node / diffusers integration", do NOT say it is "unsupported", and do NOT substitute a different model (Flux/SDXL/etc.). Build exactly what the recipe specifies. The only legitimate stop is when `get_node_schema` reports a required class truly absent from this ComfyUI instance.

---

## Step 1 — Fetch the recipe

Call `get_workflow_recipe(task=<brainbriefing task type>, model=<brainbriefing model>)`.

- **task**: the brainbriefing task type — a shorthand (`video_i2v`, `video_flf`, `video_v2v`, `image_generation`, `image_edit`, `controlnet`, `inpaint`, `upscale`, `audio`, `3d`) or a canonical name ("Image to Video", "Image Edit with ControlNet").
- **model**: the requested model family (`WAN 2.2`, `LTX-2`, `Flux`, `Qwen Image`, `Z-Image`, ...). If you do not know the exact model, omit it — the response includes `models_in_task`; pick the closest and call again with that `model`.

The returned `recipe` gives you:

| field | use |
|-------|-----|
| `execution` | `local` (local models), `api` (remote partner-node generation), or `hybrid` (local + a remote helper). See note below. |
| `member_workflows` | concrete templates that already implement this task+model — your scaffold (Step 2) |
| `required_nodes` | node classes that MUST be present, with `min_instances` (e.g. `UNETLoader` x2 for a high/low-noise model pair) |
| `node_clusters` | the required structure grouped by function (model loading, conditioning, sampling, decoding, output) |
| `connection_patterns` | the invariant role-level wiring (e.g. `model_loader -> sampler [MODEL]`) |
| `boundary_ports` | the inputs/outputs the finished workflow must expose |

**Local vs remote (`execution`)** determines what the workflow needs:
- `execution: "api"` — generation runs **remotely** on a partner service (Kling, Veo, Magnific, ...). There are no local model files to set; configure the partner node's parameters (model name, prompt, input image) instead. Local model checks/downloads do not apply.
- `execution: "local"` — runs on the local ComfyUI; wire the local model loaders (the brainbriefing model paths are Query Templates-verified).
- `execution: "hybrid"` — local generation plus a remote helper node (e.g. a Gemini prompt expander); set up both.

If `get_workflow_recipe` returns `{"error": "recipe database not found"}`, skip to the **Fallback** section at the end of this skill.

---

## Step 2 — Load a scaffold from the recipe

Pick the **closest entry in `member_workflows`** — it already matches the task and model — and call `get_workflow_template(<member_name>)`. Record `workflow_path` and the existing node IDs in `io.nodes`.

- Prefer the member whose inputs match the brainbriefing (e.g. an i2v member when the user supplied a start image; a multi-image member when `input_image_count > 1`).
- If `member_workflows` is empty, use the **Fallback** generic-scaffold table.

This scaffold is already close to correct — you reshape it in Step 3, not rebuild it.

---

## Step 3 — Conform the scaffold to the recipe

Treat the recipe as a build checklist against the scaffold:

- **Required nodes**: every `required_nodes` entry must be present with at least `min_instances` copies. Add any that are missing. Pay attention to paired nodes (`min_instances >= 2`, e.g. two model loaders / two samplers) — both must exist.
- **Connections**: every `connection_patterns` edge must be wired. Verify exact input names / slot indices with `get_node_schema` when unsure.
- **Boundary ports**: feed each `boundary_ports.inputs` from the brainbriefing inputs; wire each `boundary_ports.outputs` to a save/output node (`SaveImage`, `VHS_VideoCombine`, `CreateVideo`, ...).
- **Remove** scaffold nodes that are not part of this recipe.

Map each brainbriefing value to the correct node input:

| brainbriefing field | Target node input |
|---------------------|-------------------|
| `prompt.positive` | `CLIPTextEncode.text` (positive node) |
| `prompt.negative` | `CLIPTextEncode.text` (negative node), or skip if `null` |
| `resolution_width` | `EmptyLatentImage.width` (or equivalent) |
| `resolution_height` | `EmptyLatentImage.height` (or equivalent) |
| `input_nodes[].path` | `LoadImage.image` / `VHS_LoadImagePath.image` (per `input_nodes[].node`) |
| `output_nodes[].output_path` | `SaveImage.filename_prefix` (or equivalent output node) |
| model from `brainbriefing` | `CheckpointLoaderSimple.ckpt_name` (or `UNETLoader` / `CLIPLoader` as needed) |

Assign fresh sequential string node IDs (`"1"`, `"2"`, ...) to any nodes you add. Never reuse an ID you are removing.

---

## Step 4 — Inspect unfamiliar nodes

For any node class you are not certain about:

- **`get_node_schema(node_class)`** — required inputs, types, defaults, output slots. Use to verify input names and connection indices before wiring.
- **`search_nodes(query)`** — find the right `class_type` for a capability (e.g. `"video combine"`, `"load image path"`).

Model paths come from the brainbriefing (Query Templates-verified via `check_model`). Do NOT look up models here.

---

## Step 5 — Assemble the update_workflow call

Build three arrays:

**`patches`** — set scalar inputs on nodes you are keeping:
```json
[
  { "node_id": "<existing_id>", "input_name": "text", "value": "<positive_prompt>" },
  { "node_id": "<existing_id>", "input_name": "ckpt_name", "value": "<model_file>" }
]
```

**`add_nodes`** — full node definitions for every new node (use connection format `["sourceId", outputIndex]` for linked inputs):
```json
[
  { "id": "10", "class_type": "EmptyLatentImage",
    "inputs": { "width": 1024, "height": 1024, "batch_size": 1 },
    "_meta": { "title": "Empty Latent" } },
  { "id": "11", "class_type": "KSampler",
    "inputs": { "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                "latent_image": ["10", 0], "seed": 0, "steps": 20, "cfg": 7.0,
                "sampler_name": "euler", "scheduler": "karras", "denoise": 1.0 },
    "_meta": { "title": "KSampler" } }
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

## Step 6 — Handle errors and retry

- If `update_workflow` returns `status: "error"`: read the message, fix the specific issue (wrong connection index, missing required input, duplicate node ID, removed node still referenced), and call `update_workflow` again immediately. Retry up to 3 times, then report with `task_id` and stop. Do not ask the user.
- If a required node class is not found by `get_node_schema` (does not exist in the ComfyUI instance): report with `task_id` and stop. Do not substitute an incompatible node class.

---

## Fallback — no recipe available

If `get_workflow_recipe` reports the database is missing, select a generic scaffold via `get_workflow_catalog()` instead:

| brainbriefing task type | Preferred scaffold |
|-------------------------|--------------------|
| `image_generation`      | any `txt2img` template |
| `image_edit`            | any `img2img` template |
| `video_i2v`             | any `i2v` or `video` template |
| `video_flf`             | any `flf` or `video` template |
| `video_v2v`             | any `v2v` or `video` template |
| `audio`                 | any `audio` template |
| `3d`                    | any `3d` template |

If no close match exists, pick the simplest `txt2img` template, then continue from Step 3 (planning the node graph yourself from the standard pipeline shape: model loader → conditioning → sampler → VAE decode → save/output node).

---

## Rules

- **Never guess model file names** — model paths come from the brainbriefing (Query Templates-verified). Never look them up here.
- **Never reuse node IDs** that appear in `remove_nodes` — assign fresh IDs to all added nodes.
- **Preserve paired nodes** — when the recipe lists a node with `min_instances >= 2`, both/all copies must be present and wired to their distinct roles.
- **Always wire output nodes** (`SaveImage`, `VHS_VideoCombine`, `CreateVideo`, ...) to the final IMAGE/LATENT/AUDIO output. An unconnected output node fails validation.
- **All connections must be type-safe** — verify input names, output slot indices, and types via `get_node_schema` when unsure (it is the authoritative, always-current source for this ComfyUI instance).
- **`signal_workflow_ready` is NOT called in this skill** — return `workflow_path` to the Brain, which handles handoff per its step 2 constraints.
