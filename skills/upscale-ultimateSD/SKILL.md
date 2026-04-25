---
name: upscale-ultimatesd
description: Ultimate SD upscaler on Flux1-dev fp8 (template upscale_ultimateSD). Activate in the Researcher when the selected template is upscale_ultimateSD. Activate in the Brain during assembly of the upscale_ultimateSD template to patch the UltimateSDUpscale node, ensure the Flux1-dev fp8 checkpoint is present (download from HuggingFace if missing), and wire the LoadImage input + SaveImage output correctly.
allowed-tools: update_workflow, get_workflow_template, check_local_model, download_hf_model, search_huggingface_models, get_model_info, get_models_in_folder
---

# Ultimate SD Upscale — Flux1-dev fp8 Skill

## When to activate
- **Researcher** (step 2): user asks for image upscaling with creative detail / re-generation / tile-based upscaling / "fix textures" / higher factor than a plain ESRGAN pass can deliver → select template `upscale_ultimateSD`. For a plain model-only upscale (no diffusion pass) use `upscale_using_model` instead.
- **Brain** (step 1.1 assembly): loaded template is `upscale_ultimateSD`.
- Do **not** activate for API upscalers (Magnific, Topaz) — those have their own templates.

---

## Researcher — Template selection and brainbriefing

Set `task.type` to `image_edit` (UltimateSD re-samples the image, it is not a pure generation).

Input / output contract:
- `input_nodes`: exactly **1** image (node `8`, `LoadImage`). If the user supplies more than one image, keep only the first and add a WARNING blocker.
- `output_nodes`: exactly **1** (node `7`, `SaveImage`), `output_path` = `./agentOut/image_edit` (per `output-paths` skill).
- `prompt.positive`: a short descriptor of the **target look** — what the upscaler should reinforce (materials, textures, lighting), NOT a re-description of the subject. Default template prompt is a photoreal detail prompt — keep it unless the user asks for a specific style (anime / painterly / matte).
- `prompt.negative`: keep the template default (blur / artifacts / plastic skin / cartoon etc.) unless the user requests otherwise.
- `resolution_width` / `resolution_height`: **do not set** — UltimateSD derives output size from `upscale_by` × input image size. Leave out of the brainbriefing.

Note any non-default upscale factor, denoise, or tile size as a WARNING in `blockers` so the Brain picks it up.

---

## Brain — Template patching (step 1.1)

### Template node map (upscale_ultimateSD)

| Node ID | Class                   | Role                                                                         |
|---------|-------------------------|------------------------------------------------------------------------------|
| `1`     | `UltimateSDUpscale`     | Main upscaler — holds all sampling + tiling + seam-fix parameters            |
| `3`     | `UpscaleModelLoader`    | ESRGAN-style upscale model (default `2xNomosUni_compact_otf_medium.pth`)     |
| `4`     | `CheckpointLoaderSimple`| Diffusion checkpoint (default `FLUX1\flux1-dev-fp8.safetensors`)             |
| `5`     | `CLIPTextEncode`        | Positive prompt                                                               |
| `6`     | `CLIPTextEncode`        | Negative prompt                                                               |
| `7`     | `SaveImage`             | Output — patched via `output_nodes`                                           |
| `8`     | `LoadImage`             | Input — patched via `input_nodes`                                             |

### Assembly steps

**1. Ensure the diffusion checkpoint is on disk**

Before patching, verify `flux1-dev-fp8.safetensors` exists. The model ships under `FLUX1/`:

```
check_local_model(filename="flux1-dev-fp8.safetensors")
```

If the response reports the file is missing, download it from HuggingFace:

```
download_hf_model(
    model_id="Kijai/flux-fp8",
    filename="flux1-dev-fp8.safetensors",
    destination_folder="FLUX1"
)
```

If `check_local_model` is ambiguous, fall back to `get_models_in_folder(folder="checkpoints")` and look for `FLUX1/flux1-dev-fp8.safetensors`.

Do not proceed to `update_workflow` until the file is confirmed present — LoadCheckpoint validation will fail otherwise.

**2. Patch input node (node 8 LoadImage)**

Standard `input_nodes` patch from the brainbriefing — set `image` to the uploaded input filename. The input image MUST already be uploaded to the ComfyUI input directory (see `brain-learnings` — LoadImage validation fails on local paths).

If `input_image_count` > 1 in the brainbriefing: keep only the first image, drop the rest. This template has no batching node and a second LoadImage would not be wired to anything.

**3. Patch output node (node 7 SaveImage)**

Standard `output_nodes` patch. Set `filename_prefix` from the brainbriefing output entry; do not touch the `images` wiring — it is already connected to node `1` output slot `0`.

**4. Patch prompts**

- Node `5` (`text`): positive prompt from `brainbriefing.prompt.positive`.
- Node `6` (`text`): negative prompt from `brainbriefing.prompt.negative`.

**5. Patch UltimateSDUpscale parameters (node 1)**

Apply only the parameters the user / brainbriefing overrides. The template defaults are tuned for a photoreal 2× pass on Flux fp8 — keep them unless there is a reason to change.

### Default parameters (node 1)

| Parameter              | Default                     | Override condition                                                                                  |
|------------------------|-----------------------------|-----------------------------------------------------------------------------------------------------|
| `upscale_by`           | `2`                         | Respect user value (1.5–4 typical). Above 4 is slow and rarely improves quality — warn user.        |
| `seed`                 | template value              | Set to `brainbriefing.seed` if present, otherwise keep. For `variations=true` batches, vary seed.   |
| `steps`                | `20`                        | Raise to `25`–`30` for hard-to-reconstruct textures (skin, fabric weave). Rarely below 15.          |
| `cfg`                  | `1`                         | Keep `1` for Flux-dev fp8 (guidance distilled). Do NOT raise — it breaks Flux output.               |
| `sampler_name`         | `euler`                     | Keep for Flux. Never DDIM/UniPC on Flux fp8.                                                        |
| `scheduler`            | `simple`                    | Keep for Flux. `beta` also valid; `karras` is not recommended on Flux.                              |
| `denoise`              | `0.1`                       | Key creativity dial. `0.1`–`0.2` preserves the source; `0.25`–`0.35` adds detail; `>0.4` drifts.   |
| `mode_type`            | `Linear`                    | Use `Chess` only if seams are visible with Linear.                                                  |
| `tile_width`           | `512`                       | Keep `512` for Flux (VRAM-friendly). `768` possible on 24GB+ GPUs.                                  |
| `tile_height`          | `512`                       | Match `tile_width`.                                                                                 |
| `mask_blur`            | `8`                         | Increase to `16` if tile edges are visible.                                                         |
| `tile_padding`         | `32`                        | Increase to `64` for sharper seam blending at the cost of speed.                                    |
| `seam_fix_mode`        | `None`                      | Switch to `Band Pass` if seams persist after mask_blur / padding adjustments.                       |
| `seam_fix_denoise`     | `1`                         | Only relevant if `seam_fix_mode` != `None`.                                                         |
| `seam_fix_width`       | `64`                        | Only relevant if `seam_fix_mode` != `None`.                                                         |
| `seam_fix_mask_blur`   | `8`                         | Only relevant if `seam_fix_mode` != `None`.                                                         |
| `seam_fix_padding`     | `16`                        | Only relevant if `seam_fix_mode` != `None`.                                                         |
| `force_uniform_tiles`  | `true`                      | Keep `true` — uneven tiles cause seam artifacts on Flux.                                            |
| `tiled_decode`         | `false`                     | Set `true` only if you hit VAE decode OOM at large `upscale_by`.                                    |
| `batch_size`           | `1`                         | Keep `1`. UltimateSD does not parallelise tiles across batches.                                     |

### Upscale model (node 3)

| `model_name`                        | When to use                                                          |
|-------------------------------------|----------------------------------------------------------------------|
| `2xNomosUni_compact_otf_medium.pth` | Default — photoreal, all-round                                       |
| `4x-UltraSharp.pth`                 | If user wants a 4× pass driven by the ESRGAN model itself            |
| `4x_foolhardy_Remacri.pth`          | Illustration / anime source                                          |

If the user requests a different upscale model, verify it exists via `get_models_in_folder(folder="upscale_models")` before patching. Do not fabricate a filename.

---

## Example `update_workflow` patches

Minimal patch set (2× upscale, default prompts, denoise 0.15):

```json
[
  { "node_id": "1", "input_name": "upscale_by", "value": 2 },
  { "node_id": "1", "input_name": "denoise",    "value": 0.15 },
  { "node_id": "1", "input_name": "seed",       "value": 518529199445041 },
  { "node_id": "5", "input_name": "text",       "value": "<positive_prompt>" },
  { "node_id": "6", "input_name": "text",       "value": "<negative_prompt>" },
  { "node_id": "8", "input_name": "image",      "value": "<uploaded_input_filename>" },
  { "node_id": "7", "input_name": "filename_prefix", "value": "<output_prefix>" }
]
```

`add_nodes` / `remove_nodes` should both be empty for the standard single-image case.

---

## Troubleshooting

| Problem                                                    | Fix                                                                                               |
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `checkpoint not found: flux1-dev-fp8.safetensors`          | Run `check_local_model` → `download_hf_model(Kijai/flux-fp8, flux1-dev-fp8.safetensors, FLUX1)`.  |
| Output looks identical to input                            | `denoise` too low — raise to `0.15`–`0.25`.                                                       |
| Output drifts / faces change                               | `denoise` too high — drop to `0.1`–`0.15`; Flux is very sensitive at the upscaler stage.          |
| Visible tile seams                                         | Increase `mask_blur` to `16`, `tile_padding` to `64`, or switch `mode_type` to `Chess`.           |
| Plastic skin / waxy faces                                  | Lower `denoise` and enrich positive prompt with `skin pores, subsurface scattering`.              |
| VAE decode OOM on 4× pass                                  | Set `tiled_decode` to `true`; optionally drop `tile_width`/`tile_height` to `384`.                |
| Wrong colours / washed out                                 | `cfg` was raised above `1` — Flux fp8 requires `cfg = 1`. Reset.                                  |
| `LoadImage` validation fails on input filename             | Input image was not uploaded to ComfyUI input directory. Upload first, then re-patch.             |

---

## Checklist

**Researcher:**
- [ ] Template set to `upscale_ultimateSD`
- [ ] `task.type` set to `image_edit`
- [ ] Exactly 1 input image in `input_nodes`
- [ ] Exactly 1 output node (SaveImage, node `7`, path `./agentOut/image_edit`)
- [ ] `prompt.positive` describes target texture/look, not subject
- [ ] No `resolution_width` / `resolution_height` set in brainbriefing
- [ ] Non-default upscale factor / denoise / tile size flagged as WARNING

**Brain:**
- [ ] `check_local_model` confirmed `flux1-dev-fp8.safetensors` present (downloaded via `download_hf_model` if missing)
- [ ] Input image uploaded to ComfyUI input directory
- [ ] Node `8` `image` patched to uploaded filename
- [ ] Node `7` `filename_prefix` patched from brainbriefing output
- [ ] Node `5` / `6` text patched from brainbriefing prompts
- [ ] Node `1` `cfg` left at `1` (Flux requirement)
- [ ] Node `1` `sampler_name` = `euler`, `scheduler` = `simple`
- [ ] Node `1` `denoise` in range `0.1`–`0.35` unless user explicitly asked for more creative re-gen
- [ ] Only one `LoadImage` node remains — extras removed
